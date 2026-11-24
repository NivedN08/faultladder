"""FaultLadder tasks, registered with mjlab on import.

The evaluation task is mjlab's flat Go1 velocity task with its own
randomisation removed and the five FaultLadder event terms added, so a policy
trained on the base task can be scored on it unchanged.
"""

from dataclasses import dataclass, replace

from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from faultladder import events, ladder

EVAL_TASK = "FaultLadder-Eval-Unitree-Go1"
EVAL_PRIVILEGED_TASK = "FaultLadder-Eval-Privileged-Unitree-Go1"
MAX_LATENCY_STEPS = 5
# Wide randomisation covers the ladder up to this level; 4 and 5 are held out.
WIDE_LEVEL = 3


@dataclass(frozen=True)
class Baseline:
    """A training configuration and the task its policies are scored on."""

    train_task: str
    eval_task: str


BASELINES = {
    "nodr": Baseline("FaultLadder-NoDR-Unitree-Go1", EVAL_TASK),
    "default": Baseline("FaultLadder-Default-Unitree-Go1", EVAL_TASK),
    "wide": Baseline("FaultLadder-Wide-Unitree-Go1", EVAL_TASK),
    "privileged": Baseline("FaultLadder-Privileged-Unitree-Go1", EVAL_PRIVILEGED_TASK),
}
# Randomisation the base task applies itself; evaluation holds these fixed.
BASE_RANDOMISATION = (
    "push_robot",
    "foot_friction_slide",
    "foot_friction_spin",
    "foot_friction_roll",
    "encoder_bias",
    "base_com",
)


def _delay_buffers(cfg: ManagerBasedRlEnvCfg, max_steps: int) -> None:
    """Give the actuators a command-delay buffer whose lag only events change."""
    robot = cfg.scene.entities[events.ROBOT]
    assert robot.articulation is not None
    actuators = tuple(
        replace(
            actuator,
            delay_min_lag=0,
            delay_max_lag=max_steps * cfg.decimation,
            delay_hold_prob=1.0,
        )
        for actuator in robot.articulation.actuators
    )
    cfg.scene.entities[events.ROBOT] = replace(
        robot, articulation=EntityArticulationInfoCfg(actuators=actuators)
    )


def add_ladder_events(cfg: ManagerBasedRlEnvCfg) -> None:
    """Add the FaultLadder event terms, all at their nominal values."""
    spec = ladder.load()
    params = ladder.event_params(ladder.nominal_cell())
    # mjlab fires an interval event on the first step after its timer runs
    # out, so half a control step less makes it fire exactly on the interval.
    interval = (
        spec["push_interval_seconds"] - 0.5 * cfg.sim.mujoco.timestep * cfg.decimation
    )
    funcs = {
        ladder.MOTOR: events.motor_strength,
        ladder.PAYLOAD: events.payload,
        ladder.FRICTION: events.foot_friction,
        ladder.LATENCY: events.latency,
    }
    for name, func in funcs.items():
        cfg.events[name] = EventTermCfg(mode="reset", func=func, params=params[name])
    cfg.events[ladder.PUSH] = EventTermCfg(
        mode="interval",
        func=events.lateral_push,
        interval_range_s=(interval, interval),
        params=params[ladder.PUSH],
    )
    _delay_buffers(cfg, MAX_LATENCY_STEPS)


def faultladder_eval_env_cfg(privileged: bool = False) -> ManagerBasedRlEnvCfg:
    """The Go1 flat task with one fixed command per episode and no base DR."""
    cfg = unitree_go1_flat_env_cfg()
    for name in BASE_RANDOMISATION:
        del cfg.events[name]
    cfg.curriculum = {}
    # The evaluator ends episodes itself and writes one command per episode.
    cfg.episode_length_s = 1.0e6
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.resampling_time_range = (1.0e6, 1.0e6)
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.heading_command = False
    twist.ranges.heading = None
    add_ladder_events(cfg)
    if privileged:
        _add_fault_observation(cfg)
    return cfg


def _add_fault_observation(cfg: ManagerBasedRlEnvCfg) -> None:
    for group in ("actor", "critic"):
        cfg.observations[group].terms["fault_parameters"] = ObservationTermCfg(
            func=events.fault_parameters
        )


def no_randomisation_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """The base task with its randomisation events removed."""
    cfg = unitree_go1_flat_env_cfg(play=play)
    for name in BASE_RANDOMISATION:
        cfg.events.pop(name, None)
    return cfg


def wide_randomisation_env_cfg(
    play: bool = False, privileged: bool = False
) -> ManagerBasedRlEnvCfg:
    """The base task with randomisation widened to level 3 of every axis.

    Friction needs no change: the base task already draws 0.3 to 1.5, and
    level 3 is 0.3. ``privileged`` also shows the policy the fault parameters.
    """
    cfg = unitree_go1_flat_env_cfg(play=play)
    spec = ladder.load()
    top = {
        axis: entry["levels"][WIDE_LEVEL - 1] for axis, entry in spec["axes"].items()
    }
    _delay_buffers(cfg, MAX_LATENCY_STEPS)
    cfg.events["wide_faults"] = EventTermCfg(
        mode="reset",
        func=events.train_faults,
        params={
            "motor_prob": 0.5,
            "min_strength": min(top[f"motor_{kind}"] for kind in events.MOTOR_TYPES),
            "max_latency_steps": int(top["latency"]),
        },
    )
    cfg.events["wide_payload"] = EventTermCfg(
        mode="startup",
        func=events.payload,
        params={
            "mass_fraction": (0.0, top["payload"]),
            "max_offset": spec["payload_max_offset"],
        },
    )
    if "push_robot" in cfg.events:  # The play setup has no pushes.
        push = cfg.events["push_robot"].params["velocity_range"]
        push["y"] = (-top["push"], top["push"])
    if privileged:
        _add_fault_observation(cfg)
    return cfg


def _register(task_id: str, env_cfg, play_env_cfg) -> None:
    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_cfg,
        play_env_cfg=play_env_cfg,
        rl_cfg=unitree_go1_ppo_runner_cfg(),
        runner_cls=VelocityOnPolicyRunner,
    )


_register(EVAL_TASK, faultladder_eval_env_cfg(), faultladder_eval_env_cfg())
_register(
    EVAL_PRIVILEGED_TASK,
    faultladder_eval_env_cfg(privileged=True),
    faultladder_eval_env_cfg(privileged=True),
)
_register(
    BASELINES["nodr"].train_task,
    no_randomisation_env_cfg(),
    no_randomisation_env_cfg(play=True),
)
_register(
    BASELINES["default"].train_task,
    unitree_go1_flat_env_cfg(),
    unitree_go1_flat_env_cfg(play=True),
)
_register(
    BASELINES["wide"].train_task,
    wide_randomisation_env_cfg(),
    wide_randomisation_env_cfg(play=True),
)
_register(
    BASELINES["privileged"].train_task,
    wide_randomisation_env_cfg(privileged=True),
    wide_randomisation_env_cfg(play=True, privileged=True),
)
