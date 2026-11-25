"""The ladder's events on the Unitree G1 humanoid, four environments on CPU."""

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from faultladder import events, ladder
from faultladder.ladder import Cell
from faultladder.tasks import BASELINES, TASK_ROBOT

NUM_ENVS = 4
PARTS = {"hip": "_hip_", "thigh": "_knee_", "calf": "_ankle_"}
TASK = BASELINES["g1_default"].eval_task


@pytest.fixture(scope="module")
def env():
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    environment = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield environment
    environment.close()


def _field(env: ManagerBasedRlEnv, field: str) -> torch.Tensor:
    return torch.as_tensor(getattr(env.sim.model, field)[:]).clone()


def test_g1_tasks_use_the_g1_ladder():
    assert TASK_ROBOT[TASK] == "g1"
    assert set(ladder.load("g1")["axes"]) == set(ladder.load("go1")["axes"])
    assert len(ladder.cells("g1")) == 36


@pytest.mark.parametrize(
    ("joint_type", "count"), [("hip", 6), ("thigh", 2), ("calf", 4)]
)
def test_motor_axes_cover_only_leg_joints(env, joint_type, count):
    robot = env.scene["robot"]
    ids = events.motor_ctrl_ids(robot, joint_type)
    assert len(ids) == count
    model = env.sim.mj_model
    names = [model.joint(int(model.actuator_trnid[int(i), 0])).name for i in ids]
    assert all("left_" in n or "right_" in n for n in names)
    assert all(part in n for n in names for part in [PARTS[joint_type]])

    default = env.sim.get_default_field("actuator_gainprm")
    ladder.set_cell(env, Cell(f"motor_{joint_type}", 3, 0.4))
    env.reset()
    ratio = _field(env, "actuator_gainprm")[:, :, 0] / default[:, 0]
    weak = torch.isclose(ratio, torch.tensor(0.4))
    assert torch.all(weak.sum(dim=1) == 1)
    assert torch.all(weak[:, [int(i) for i in ids]].sum(dim=1) == 1)
    ladder.set_cell(env, ladder.nominal_cell())
    env.reset()
    torch.testing.assert_close(
        _field(env, "actuator_gainprm"), default.expand(NUM_ENVS, -1, -1)
    )


def test_payload_goes_on_the_torso(env):
    robot = env.scene["robot"]
    torso = events.payload_body(robot)
    assert torso != int(robot.indexing.root_body_id)
    mass0 = env.sim.get_default_field("body_mass")
    robot_mass = mass0[robot.indexing.body_ids].sum()
    ladder.set_cell(env, Cell("payload", 3, 0.36))
    env.reset()
    added = _field(env, "body_mass") - mass0
    torch.testing.assert_close(added[:, torso], (0.36 * robot_mass).expand(NUM_ENVS))
    assert torch.all(added.sum(dim=1) == added[:, torso])
    ladder.set_cell(env, ladder.nominal_cell())
    env.reset()
    torch.testing.assert_close(_field(env, "body_mass"), mass0.expand(NUM_ENVS, -1))


def test_friction_covers_both_feet(env):
    robot = env.scene["robot"]
    local, names = robot.find_geoms(events.FOOT_GEOMS)
    assert len(names) == 14
    feet = robot.indexing.geom_ids[local]
    ladder.set_cell(env, Cell("friction", 3, 0.3))
    env.reset()
    friction = _field(env, "geom_friction")[:, feet, 0]
    torch.testing.assert_close(friction, torch.full((NUM_ENVS, 14), 0.3))
    priority = torch.as_tensor(env.sim.mj_model.geom_priority)
    others = torch.ones(len(priority), dtype=torch.bool)
    others[feet] = False
    assert priority[feet].min() > priority[others].max()
    ladder.set_cell(env, ladder.nominal_cell())
    env.reset()
