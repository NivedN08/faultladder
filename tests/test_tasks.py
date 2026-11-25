import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from faultladder import ladder
from faultladder.baseline import BASE_TASK
from faultladder.tasks import BASE_RANDOMISATION, EVAL_TASK, MAX_LATENCY_STEPS


def _actuators(cfg):
    articulation = cfg.scene.entities["robot"].articulation
    assert articulation is not None
    return articulation.actuators


def _spaces(task: str) -> tuple[dict[str, tuple[int, ...]], int]:
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    obs, _ = env.reset()
    shapes = {
        group: tuple(torch.as_tensor(value).shape[1:]) for group, value in obs.items()
    }
    actions = env.action_manager.total_action_dim
    env.close()
    return shapes, actions


def test_eval_task_keeps_the_base_task_interface():
    assert _spaces(EVAL_TASK) == _spaces(BASE_TASK)


def test_eval_task_has_only_ladder_perturbations():
    cfg = load_env_cfg(EVAL_TASK)
    assert not set(BASE_RANDOMISATION) & set(cfg.events)
    assert set(cfg.events) == {
        "reset_base",
        "reset_robot_joints",
        ladder.MOTOR,
        ladder.PAYLOAD,
        ladder.FRICTION,
        ladder.LATENCY,
        ladder.PUSH,
    }
    assert cfg.events[ladder.PUSH].interval_range_s == (1.99, 1.99)
    assert cfg.observations["actor"].enable_corruption
    for actuator in _actuators(cfg):
        assert actuator.delay_max_lag == MAX_LATENCY_STEPS * cfg.decimation
        assert actuator.delay_hold_prob == 1.0


def test_base_task_is_left_untouched():
    cfg = load_env_cfg(BASE_TASK)
    assert set(BASE_RANDOMISATION) - {"foot_friction"} <= set(cfg.events)
    for actuator in _actuators(cfg):
        assert actuator.delay_max_lag == 0


def test_set_cell_rejects_an_unknown_axis():
    with pytest.raises(ValueError):
        ladder.event_params(ladder.Cell("wind", 1, 1.0))
