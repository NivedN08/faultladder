"""The four training configurations of Phase 4."""

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from faultladder import events, ladder
from faultladder.baseline import BASE_TASK
from faultladder.ladder import Cell
from faultladder.tasks import BASE_RANDOMISATION, BASELINES, EVAL_PRIVILEGED_TASK

NUM_ENVS = 64
NUM_FAULT_PARAMETERS = 16


def _env(task: str, num_envs: int = NUM_ENVS) -> ManagerBasedRlEnv:
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = num_envs
    cfg.seed = 0
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def _obs_dims(env: ManagerBasedRlEnv) -> dict[str, int]:
    obs, _ = env.reset()
    return {group: torch.as_tensor(value).shape[1] for group, value in obs.items()}


@pytest.fixture(scope="module")
def base_dims():
    env = _env(BASE_TASK, 2)
    dims = _obs_dims(env)
    env.close()
    return dims


@pytest.fixture(scope="module")
def wide():
    env = _env(BASELINES["wide"].train_task)
    env.reset()
    yield env
    env.close()


def test_no_randomisation_removes_every_randomisation_event(base_dims):
    cfg = load_env_cfg(BASELINES["nodr"].train_task)
    assert not set(BASE_RANDOMISATION) & set(cfg.events)
    assert set(cfg.events) == {"reset_base", "reset_robot_joints"}
    env = _env(BASELINES["nodr"].train_task, 2)
    assert _obs_dims(env) == base_dims
    env.close()


def test_default_is_the_base_task():
    ours = load_env_cfg(BASELINES["default"].train_task)
    base = load_env_cfg(BASE_TASK)
    assert set(ours.events) == set(base.events)
    assert set(ours.rewards) == set(base.rewards)


def test_wide_keeps_the_policy_interface(wide, base_dims):
    obs = wide.observation_manager.compute()
    assert {g: torch.as_tensor(v).shape[1] for g, v in obs.items()} == base_dims


def test_wide_weakens_at_most_one_motor_down_to_level_three(wide):
    strengths = events.fault_parameters(wide)[:, :12]
    weak = strengths < 0.999
    assert weak.sum(dim=1).max() == 1
    assert 0.25 < weak.any(dim=1).float().mean() < 0.75
    assert strengths.min() >= 0.4 - 1e-6
    assert strengths.max() <= 1.0 + 1e-6


def test_wide_payload_friction_and_latency_cover_levels_one_to_three(wide):
    parameters = events.fault_parameters(wide)
    payload, offset, friction, latency = parameters[:, 12:].T
    assert payload.min() >= 0.0 and payload.max() <= 0.36 + 1e-6
    assert payload.max() > 0.24
    assert offset.abs().max() <= 0.1
    assert friction.min() >= 0.3 and friction.min() < 0.6
    assert set(latency.tolist()) == {i / 4 for i in range(9)}


def test_wide_pushes_sideways_up_to_level_three():
    cfg = load_env_cfg(BASELINES["wide"].train_task)
    assert cfg.events["push_robot"].params["velocity_range"]["y"] == (-1.0, 1.0)
    base = load_env_cfg(BASE_TASK)
    assert base.events["push_robot"].params["velocity_range"]["y"] == (-0.5, 0.5)


def test_wide_survives_zero_and_random_actions(wide):
    for scale in (0.0, 1.0):
        for _ in range(60):
            action = scale * (torch.rand(NUM_ENVS, 12) * 2 - 1)
            obs, reward, *_ = wide.step(action)
            assert torch.isfinite(torch.as_tensor(obs["actor"])).all()
            assert torch.isfinite(reward).all()
    # Faults are redrawn on reset, still within range.
    strengths = events.fault_parameters(wide)[:, :12]
    assert strengths.min() >= 0.4 - 1e-6
    assert (strengths < 0.999).sum(dim=1).max() == 1


def test_privileged_adds_the_fault_parameters_to_actor_and_critic(base_dims):
    env = _env(BASELINES["privileged"].train_task, 2)
    dims = _obs_dims(env)
    env.close()
    assert dims == {g: n + NUM_FAULT_PARAMETERS for g, n in base_dims.items()}


def test_privileged_eval_reports_the_cell_being_run():
    env = _env(EVAL_PRIVILEGED_TASK, 4)
    expected = {
        Cell("motor_calf", 3, 0.4): lambda p: torch.testing.assert_close(
            p[:, :12].sort(dim=1).values[:, :2],
            torch.tensor([[0.4, 1.0]]).expand(4, 2),
        ),
        Cell("payload", 2, 0.24): lambda p: torch.testing.assert_close(
            p[:, 12], torch.full((4,), 0.24), atol=1e-4, rtol=0
        ),
        Cell("friction", 3, 0.3): lambda p: torch.testing.assert_close(
            p[:, 14], torch.full((4,), 0.3)
        ),
        Cell("latency", 2, 2.0): lambda p: torch.testing.assert_close(
            p[:, 15], torch.full((4,), 2.0)
        ),
    }
    for cell, check in expected.items():
        ladder.set_cell(env, cell)
        obs, _ = env.reset()
        parameters = events.fault_parameters(env)
        check(parameters)
        actor = torch.as_tensor(obs["actor"])
        torch.testing.assert_close(actor[:, -NUM_FAULT_PARAMETERS:], parameters)
    ladder.set_cell(env, ladder.nominal_cell())
    env.reset()
    nominal = events.fault_parameters(env)
    torch.testing.assert_close(nominal[:, :12], torch.ones(4, 12))
    torch.testing.assert_close(
        nominal[:, [12, 13, 15]], torch.zeros(4, 3), atol=1e-6, rtol=0
    )
    env.close()
