"""FaultLadder perturbations as mjlab event terms.

Each function has mjlab's event signature ``(env, env_ids, **params)`` and is
idempotent: it first restores the model defaults for ``env_ids`` and then
applies its perturbation, so calling it with the nominal value undoes it.
"""

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_yaw, quat_from_matrix

MOTOR_TYPES = ("hip", "thigh", "calf")
FOOT_GEOMS = ("^[FR][LR]_foot_collision$",)
ROBOT = "robot"


def _env_ids(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids.to(env.device)


def motor_ctrl_ids(asset: Entity, joint_type: str) -> torch.Tensor:
    """Global control indices of the four joints of one type, e.g. the hips."""
    if joint_type not in MOTOR_TYPES:
        raise ValueError(f"joint_type must be one of {MOTOR_TYPES}, got {joint_type!r}")
    ids = [
        int(ctrl_id)
        for actuator in asset.actuators
        for name, ctrl_id in zip(
            actuator.target_names, actuator.global_ctrl_ids.tolist(), strict=True
        )
        if name.endswith(f"_{joint_type}_joint")
    ]
    return torch.tensor(ids, device=asset.data.device)


@requires_model_fields("actuator_gainprm", "actuator_biasprm", "actuator_forcerange")
def motor_strength(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    joint_type: str | None,
    strength: float,
) -> None:
    """Weaken one randomly chosen joint of ``joint_type`` in each environment.

    ``strength`` scales that motor's stiffness, damping and torque limit; 0.0 is
    a dead (free-swinging) motor and 1.0, or ``joint_type=None``, is healthy.
    """
    ids = _env_ids(env, env_ids)
    asset: Entity = env.scene[ROBOT]
    model = env.sim.model
    gain = env.sim.get_default_field("actuator_gainprm")
    bias = env.sim.get_default_field("actuator_biasprm")
    force = env.sim.get_default_field("actuator_forcerange")
    every = asset.indexing.ctrl_ids
    rows = ids[:, None]
    model.actuator_gainprm[rows, every] = gain[every]
    model.actuator_biasprm[rows, every] = bias[every]
    model.actuator_forcerange[rows, every] = force[every]
    if joint_type is None or strength == 1.0:
        return
    candidates = motor_ctrl_ids(asset, joint_type)
    pick = torch.randint(len(candidates), (len(ids),), device=env.device)
    _weaken(env, ids, candidates[pick], torch.full((len(ids),), strength))


def _weaken(
    env: ManagerBasedRlEnv,
    ids: torch.Tensor,
    ctrl: torch.Tensor,
    strength: torch.Tensor,
) -> None:
    """Scale motor ``ctrl[i]`` of environment ``ids[i]`` by ``strength[i]``."""
    model = env.sim.model
    gain = env.sim.get_default_field("actuator_gainprm")
    bias = env.sim.get_default_field("actuator_biasprm")
    force = env.sim.get_default_field("actuator_forcerange")
    strength = strength.to(env.device)
    # Position actuator: gainprm[0] = kp, biasprm[1] = -kp, biasprm[2] = -kd.
    model.actuator_gainprm[ids, ctrl, 0] = gain[ctrl, 0] * strength
    model.actuator_biasprm[ids, ctrl, 1] = bias[ctrl, 1] * strength
    model.actuator_biasprm[ids, ctrl, 2] = bias[ctrl, 2] * strength
    model.actuator_forcerange[ids, ctrl] = force[ctrl] * strength[:, None]


@requires_model_fields(
    "body_mass",
    "body_ipos",
    "body_inertia",
    "body_iquat",
    recompute=RecomputeLevel.set_const,
)
def payload(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    mass_fraction: float | tuple[float, float],
    max_offset: float = 0.1,
) -> None:
    """Attach a point mass to the trunk, offset sideways by up to ``max_offset``.

    A ``(low, high)`` ``mass_fraction`` draws one value per environment.

    The mass is ``mass_fraction`` of the whole robot's mass and sits at the
    trunk's centre of mass shifted along the body y axis. The trunk's mass,
    centre of mass and inertia are updated together.
    """
    ids = _env_ids(env, env_ids)
    asset: Entity = env.scene[ROBOT]
    model = env.sim.model
    bodies = asset.indexing.body_ids
    trunk = int(asset.indexing.root_body_id)
    mass0 = env.sim.get_default_field("body_mass")
    ipos0 = env.sim.get_default_field("body_ipos")[trunk]
    inertia0 = env.sim.get_default_field("body_inertia")[trunk]
    iquat0 = env.sim.get_default_field("body_iquat")[trunk]
    n = len(ids)
    if mass_fraction == 0.0:
        model.body_mass[ids, trunk] = mass0[trunk]
        model.body_ipos[ids, trunk] = ipos0
        model.body_inertia[ids, trunk] = inertia0
        model.body_iquat[ids, trunk] = iquat0
        return
    if isinstance(mass_fraction, tuple):
        low, high = mass_fraction
        fraction = low + torch.rand(n, device=env.device) * (high - low)
    else:
        fraction = torch.full((n,), mass_fraction, device=env.device)
    added = fraction * mass0[bodies].sum()
    total = mass0[trunk] + added
    offset = (torch.rand(n, device=env.device) * 2 - 1) * max_offset
    ipos = ipos0.repeat(n, 1)
    ipos[:, 1] += offset * added / total
    # Inertia of trunk plus point mass about their joint centre of mass, in the
    # body frame: two bodies a distance |offset| apart along y add
    # reduced_mass * offset^2 about the x and z axes.
    rot0 = matrix_from_quat(iquat0)
    tensor = (rot0 * inertia0) @ rot0.T
    reduced = mass0[trunk] * added / total
    extra = torch.zeros(n, 3, 3, device=env.device, dtype=tensor.dtype)
    extra[:, 0, 0] = extra[:, 2, 2] = reduced * offset**2
    values, vectors = torch.linalg.eigh(tensor + extra)
    # eigh may return a reflection; flip one axis to keep a proper rotation.
    flip = torch.linalg.det(vectors) < 0
    vectors[flip, :, 0] *= -1
    model.body_mass[ids, trunk] = total
    model.body_ipos[ids, trunk] = ipos
    model.body_inertia[ids, trunk] = values
    model.body_iquat[ids, trunk] = quat_from_matrix(vectors)


@requires_model_fields("geom_friction")
def foot_friction(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, coefficient: float | None
) -> None:
    """Set the sliding friction of all four feet; ``None`` restores the model's.

    The Go1 feet have contact priority over the floor, so this is the
    foot-to-floor coefficient.
    """
    ids = _env_ids(env, env_ids)
    asset: Entity = env.scene[ROBOT]
    local, _ = asset.find_geoms(FOOT_GEOMS)
    feet = asset.indexing.geom_ids[local]
    default = env.sim.get_default_field("geom_friction")[feet, 0]
    value = default if coefficient is None else torch.full_like(default, coefficient)
    env.sim.model.geom_friction[ids[:, None], feet, 0] = value


def latency(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, steps: float) -> None:
    """Delay every motor command by ``steps`` control steps.

    ``steps`` may be a fraction that is a whole number of physics steps. Needs
    actuators built with ``delay_max_lag >= steps * decimation`` and
    ``delay_hold_prob=1.0`` (see ``faultladder.tasks``). Run in ``reset`` mode:
    mjlab zeroes the lag whenever an environment resets.
    """
    ids = _env_ids(env, env_ids)
    lag = steps * env.cfg.decimation
    if lag != round(lag):
        raise ValueError(f"latency of {steps} control steps is not whole physics steps")
    _set_lags(env, ids, torch.full((len(ids),), round(lag), device=env.device))


def _set_lags(env: ManagerBasedRlEnv, ids: torch.Tensor, lags: torch.Tensor) -> None:
    """Set the command delay of ``ids`` in physics steps."""
    asset: Entity = env.scene[ROBOT]
    for actuator in asset.actuators:
        if int(lags.max()) > actuator.cfg.delay_max_lag:
            raise ValueError(
                f"a delay of {int(lags.max())} physics steps needs delay_max_lag "
                f">= {int(lags.max())}, got {actuator.cfg.delay_max_lag}"
            )
        actuator.set_lags(lags, ids)


@requires_model_fields("actuator_gainprm", "actuator_biasprm", "actuator_forcerange")
def train_faults(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motor_prob: float,
    min_strength: float,
    max_latency_steps: float,
) -> None:
    """Training randomisation: a weak motor and a command delay per episode.

    With probability ``motor_prob`` one of the twelve motors gets a strength
    drawn from ``[min_strength, 1]``; the delay is 0 to ``max_latency_steps``
    control steps, in whole physics steps. Both are redrawn at every reset.
    """
    ids = _env_ids(env, env_ids)
    motor_strength(env, ids, None, 1.0)
    asset: Entity = env.scene[ROBOT]
    faulty = ids[torch.rand(len(ids), device=env.device) < motor_prob]
    every = asset.indexing.ctrl_ids
    ctrl = every[torch.randint(len(every), (len(faulty),), device=env.device)]
    strength = min_strength + torch.rand(len(faulty), device=env.device) * (
        1.0 - min_strength
    )
    _weaken(env, faulty, ctrl, strength)
    max_lag = round(max_latency_steps * env.cfg.decimation)
    _set_lags(env, ids, torch.randint(max_lag + 1, (len(ids),), device=env.device))


def fault_parameters(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Privileged observation: the fault parameters, one row per environment.

    Twelve motor strengths, added trunk mass over nominal robot mass, sideways
    shift of the trunk's centre of mass (m), foot friction and command delay
    (control steps). Pushes are instantaneous and have no parameter to report.
    """
    asset: Entity = env.scene[ROBOT]
    model = env.sim.model
    ctrl = asset.indexing.ctrl_ids
    trunk = int(asset.indexing.root_body_id)
    gain0 = env.sim.get_default_field("actuator_gainprm")[ctrl, 0]
    mass0 = env.sim.get_default_field("body_mass")
    ipos0 = env.sim.get_default_field("body_ipos")[trunk, 1]
    local, _ = asset.find_geoms(FOOT_GEOMS)
    foot = asset.indexing.geom_ids[local[0]]
    # mjlab exposes no getter for the current lag.
    buffer = asset.actuators[0]._delay_buffer
    lag = (
        torch.zeros(env.num_envs, device=env.device)
        if buffer is None
        else buffer.current_lags.float()
    )
    return torch.cat(
        [
            model.actuator_gainprm[:, ctrl, 0] / gain0,
            (
                (model.body_mass[:, trunk] - mass0[trunk])
                / mass0[asset.indexing.body_ids].sum()
            )[:, None],
            (model.body_ipos[:, trunk, 1] - ipos0)[:, None],
            model.geom_friction[:, foot, 0][:, None],
            (lag / env.cfg.decimation)[:, None],
        ],
        dim=1,
    )


def lateral_push(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, speed: float
) -> None:
    """Add ``speed`` m/s to the trunk's velocity, to its left or right at random."""
    if speed == 0.0:
        return
    ids = _env_ids(env, env_ids)
    asset: Entity = env.scene[ROBOT]
    sign = torch.randint(2, (len(ids),), device=env.device) * 2.0 - 1.0
    kick = torch.zeros(len(ids), 3, device=env.device)
    kick[:, 1] = sign * speed
    velocity = asset.data.root_link_vel_w[ids].clone()
    velocity[:, :3] += quat_apply_yaw(asset.data.root_link_quat_w[ids], kick)
    asset.write_root_link_velocity_to_sim(velocity, env_ids=ids)
