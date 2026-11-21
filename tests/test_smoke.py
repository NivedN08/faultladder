import pytest

from faultladder.smoke import BASE_TASK, rollout


def test_base_task_is_registered():
    from mjlab.tasks.registry import list_tasks

    assert BASE_TASK in list_tasks()


@pytest.mark.slow
@pytest.mark.parametrize("agent", ["zero", "random"])
def test_dummy_agent_rollout_is_finite(agent):
    out = rollout(agent=agent, num_envs=4, steps=20)
    assert out["observations_finite"]
    assert out["env_steps_per_second"] > 0
