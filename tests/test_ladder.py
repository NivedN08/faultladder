"""Each ladder axis must change the physical quantity it claims to change."""

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply_yaw

from faultladder import events, ladder
from faultladder.ladder import Cell
from faultladder.score import AXES
from faultladder.tasks import EVAL_TASK

NUM_ENVS = 4
PLAN_LEVELS = {
    "motor_hip": [0.8, 0.6, 0.4, 0.2, 0.0],
    "motor_thigh": [0.8, 0.6, 0.4, 0.2, 0.0],
    "motor_calf": [0.8, 0.6, 0.4, 0.2, 0.0],
    "payload": [0.1, 0.2, 0.3, 0.4, 0.5],
    "friction": [0.6, 0.45, 0.3, 0.2, 0.1],
    "latency": [1, 2, 3, 4, 5],
    "push": [0.5, 1.0, 1.5, 2.0, 2.5],
}


@pytest.fixture(scope="module")
def env():
    cfg = load_env_cfg(EVAL_TASK)
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    environment = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield environment
    environment.close()


def _apply(env: ManagerBasedRlEnv, cell: Cell) -> None:
    ladder.set_cell(env, cell)
    env.reset()


def _default(env: ManagerBasedRlEnv, field: str) -> torch.Tensor:
    return env.sim.get_default_field(field)


def _field(env: ManagerBasedRlEnv, field: str) -> torch.Tensor:
    return torch.as_tensor(getattr(env.sim.model, field)[:]).clone()


def test_ladder_file_matches_the_plan():
    spec = ladder.load()
    assert {a: spec["axes"][a]["levels"] for a in AXES} == PLAN_LEVELS
    all_cells = ladder.cells()
    assert len(all_cells) == 36
    assert all_cells[0].name == "nominal"
    assert len({cell.name for cell in all_cells}) == 36


@pytest.mark.parametrize("joint_type", events.MOTOR_TYPES)
def test_motor_fault_scales_exactly_one_joint_of_the_named_type(env, joint_type):
    strength = 0.4
    _apply(env, Cell(f"motor_{joint_type}", 3, strength))
    robot = env.scene["robot"]
    of_type = set(events.motor_ctrl_ids(robot, joint_type).tolist())
    assert len(of_type) == 4
    kp0 = _default(env, "actuator_gainprm")[:, 0]
    kp = _field(env, "actuator_gainprm")[:, :, 0]
    kd_ratio = (
        _field(env, "actuator_biasprm")[:, :, 2]
        / _default(env, "actuator_biasprm")[:, 2]
    )
    limit_ratio = _field(env, "actuator_forcerange") / _default(
        env, "actuator_forcerange"
    )
    for i in range(NUM_ENVS):
        changed = (kp[i] != kp0).nonzero().flatten().tolist()
        assert len(changed) == 1
        assert changed[0] in of_type
        ratios = torch.ones(len(kp0))
        ratios[changed[0]] = strength
        torch.testing.assert_close(kp[i] / kp0, ratios)
        torch.testing.assert_close(kd_ratio[i], ratios)
        torch.testing.assert_close(limit_ratio[i], ratios[:, None].expand(-1, 2))
    # The stiffness term of the position actuator stays consistent: bias = -kp.
    torch.testing.assert_close(_field(env, "actuator_biasprm")[:, :, 1], -kp)

    _apply(env, ladder.nominal_cell())
    torch.testing.assert_close(
        _field(env, "actuator_gainprm")[:, :, 0], kp0.expand_as(kp)
    )
    torch.testing.assert_close(
        _field(env, "actuator_forcerange"),
        _default(env, "actuator_forcerange").expand(NUM_ENVS, -1, -1),
    )


def test_dead_motor_produces_no_torque(env):
    _apply(env, Cell("motor_calf", 5, 0.0))
    kp = _field(env, "actuator_gainprm")[:, :, 0]
    dead = (kp == 0).nonzero()
    assert len(dead) == NUM_ENVS
    for _ in range(5):
        env.step(torch.ones(NUM_ENVS, 12))
    force = torch.as_tensor(env.sim.data.actuator_force[:])
    assert torch.all(force[dead[:, 0], dead[:, 1]] == 0)
    assert torch.any(force != 0)
    _apply(env, ladder.nominal_cell())


def test_payload_adds_the_specified_mass_with_a_sideways_offset(env):
    fraction = 0.3
    robot = env.scene["robot"]
    bodies = robot.indexing.body_ids
    trunk = int(robot.indexing.root_body_id)
    mass0 = _default(env, "body_mass")
    robot_mass = mass0[bodies].sum()
    _apply(env, Cell("payload", 3, fraction))
    mass = _field(env, "body_mass")
    torch.testing.assert_close(
        mass[:, bodies].sum(dim=1), (robot_mass * (1 + fraction)).expand(NUM_ENVS)
    )
    # Only the trunk gets heavier.
    others = [int(b) for b in bodies if int(b) != trunk]
    torch.testing.assert_close(mass[:, others], mass0[others].expand(NUM_ENVS, -1))
    added = fraction * robot_mass
    shift = _field(env, "body_ipos")[:, trunk] - _default(env, "body_ipos")[trunk]
    assert torch.all(shift[:, [0, 2]] == 0)
    # Payload position recovered from the centre-of-mass shift is within 0.1 m.
    offset = shift[:, 1] * (mass0[trunk] + added) / added
    assert torch.all(offset.abs() <= 0.1 + 1e-6)
    assert offset.abs().max() > 0.01
    assert len(offset.unique()) == NUM_ENVS
    # A point mass can only add rotational inertia.
    inertia = _field(env, "body_inertia")[:, trunk]
    assert torch.all(inertia.sum(dim=1) >= _default(env, "body_inertia")[trunk].sum())
    torch.testing.assert_close(
        _field(env, "body_iquat")[:, trunk].norm(dim=1), torch.ones(NUM_ENVS)
    )
    # The simulator's own total (recomputed constants) agrees.
    subtree = torch.as_tensor(env.sim.model.body_subtreemass[:])
    torch.testing.assert_close(
        subtree[:, trunk], (robot_mass * (1 + fraction)).expand(NUM_ENVS)
    )

    _apply(env, ladder.nominal_cell())
    torch.testing.assert_close(_field(env, "body_mass"), mass0.expand(NUM_ENVS, -1))
    torch.testing.assert_close(
        _field(env, "body_inertia")[:, trunk],
        _default(env, "body_inertia")[trunk].expand(NUM_ENVS, -1),
    )


def test_friction_sets_the_foot_to_floor_coefficient(env):
    robot = env.scene["robot"]
    local, names = robot.find_geoms(events.FOOT_GEOMS)
    assert len(names) == 4
    feet = robot.indexing.geom_ids[local]
    default = _default(env, "geom_friction")
    _apply(env, Cell("friction", 3, 0.3))
    friction = _field(env, "geom_friction")
    torch.testing.assert_close(friction[:, feet, 0], torch.full((NUM_ENVS, 4), 0.3))
    untouched = friction.clone()
    untouched[:, feet, 0] = default[feet, 0]
    torch.testing.assert_close(untouched, default.expand(NUM_ENVS, -1, -1))
    # MuJoCo uses the higher-priority geom's friction: the feet must win.
    priority = torch.as_tensor(env.sim.mj_model.geom_priority)
    others = torch.ones(len(priority), dtype=torch.bool)
    others[feet] = False
    assert priority[feet].min() > priority[others].max()

    _apply(env, ladder.nominal_cell())
    torch.testing.assert_close(
        _field(env, "geom_friction"), default.expand(NUM_ENVS, -1, -1)
    )
    assert torch.all(default[feet, 0] == ladder.load()["axes"]["friction"]["nominal"])


@pytest.mark.parametrize("steps", [0, 1, 3, 5])
def test_latency_applies_the_action_from_k_steps_earlier(env, steps):
    cell = Cell("latency", steps, steps) if steps else ladder.nominal_cell()
    _apply(env, cell)
    robot = env.scene["robot"]
    ctrl_ids = robot.indexing.ctrl_ids
    order = [
        robot.joint_names.index(name)
        for actuator in robot.actuators
        for name in actuator.target_names
    ]
    generator = torch.Generator().manual_seed(steps)
    targets, applied = [], []
    for _ in range(12):
        env.step(torch.rand(NUM_ENVS, 12, generator=generator) * 2 - 1)
        targets.append(robot.data.joint_pos_target[:, order].clone())
        applied.append(torch.as_tensor(env.sim.data.ctrl[:])[:, ctrl_ids].clone())
    for t in range(steps, 12):
        torch.testing.assert_close(applied[t], targets[t - steps])
    if steps:
        assert not torch.allclose(applied[-1], targets[-1])
    _apply(env, ladder.nominal_cell())


def test_push_changes_sideways_velocity_by_the_push_size(env):
    _apply(env, ladder.nominal_cell())
    robot = env.scene["robot"]
    before = robot.data.root_link_vel_w.clone()
    lateral = torch.zeros(NUM_ENVS, 3)
    lateral[:, 1] = 1.0
    lateral_w = quat_apply_yaw(robot.data.root_link_quat_w, lateral)
    events.lateral_push(env, None, speed=1.5)
    env.sim.forward()
    delta = robot.data.root_link_vel_w - before
    torch.testing.assert_close(
        delta[:, 3:], torch.zeros(NUM_ENVS, 3), atol=1e-6, rtol=0
    )
    along = (delta[:, :3] * lateral_w).sum(dim=1)
    torch.testing.assert_close(
        along.abs(), torch.full((NUM_ENVS,), 1.5), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(delta[:, :3].norm(dim=1), along.abs(), atol=1e-5, rtol=0)
    assert torch.all(delta[:, 2].abs() < 1e-6)


def test_push_fires_every_two_seconds_in_the_task(env):
    _apply(env, Cell("push", 1, 0.5))
    robot = env.scene["robot"]
    jumps = []
    previous = robot.data.root_link_vel_w[:, :2].clone()
    for step in range(1, 251):
        env.step(torch.zeros(NUM_ENVS, 12))
        now = robot.data.root_link_vel_w[:, :2].clone()
        change = (now - previous).norm(dim=1)
        if torch.any((change - 0.5).abs() < 0.01):
            jumps.append(step)
        previous = now
    assert jumps == [100, 200]
    _apply(env, ladder.nominal_cell())


def test_nominal_cell_matches_the_unmodified_model(env):
    _apply(env, Cell("payload", 5, 0.5))
    _apply(env, ladder.nominal_cell())
    for field in (
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "body_mass",
        "body_ipos",
        "body_inertia",
        "body_iquat",
        "geom_friction",
    ):
        default = _default(env, field)
        torch.testing.assert_close(
            _field(env, field), default.expand(NUM_ENVS, *default.shape)
        )
