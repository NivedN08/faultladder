import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from faultladder.checks import evaluate, oracle_policy, random_policy, zero_policy
from faultladder.reacher.env import (
    LINK_LENGTHS,
    MAX_EPISODE_STEPS,
    REWARD_BOUNDS,
    TARGET_RADIUS_RANGE,
    ReacherEnv,
)
from faultladder.reacher.oracle import forward_kinematics, inverse_kinematics


@pytest.fixture
def env():
    e = ReacherEnv()
    yield e
    e.close()


def rollout(seed: int, actions: np.ndarray) -> list[np.ndarray]:
    e = ReacherEnv()
    obs, _ = e.reset(seed=seed)
    out = [obs]
    for a in actions:
        obs, _, terminated, truncated, _ = e.step(a)
        out.append(obs)
        if terminated or truncated:
            break
    e.close()
    return out


def test_spaces(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.shape == (10,)
    assert env.action_space.shape == (2,)
    assert obs.shape == (10,)
    assert obs.dtype == env.observation_space.dtype
    assert env.action_space.low.tolist() == [-1.0, -1.0]
    assert env.action_space.high.tolist() == [1.0, 1.0]
    assert info["is_success"] is False


def test_gymnasium_env_checker(env):
    check_env(env, skip_render_check=True)


def test_observation_layout(env):
    obs, _ = env.reset(seed=3)
    q = env.data.qpos.copy()
    tip = np.array(forward_kinematics(q[0], q[1]))
    np.testing.assert_allclose(obs[0:2], np.cos(q), atol=1e-6)
    np.testing.assert_allclose(obs[2:4], np.sin(q), atol=1e-6)
    np.testing.assert_allclose(obs[4:6], 0.0, atol=1e-6)
    np.testing.assert_allclose(obs[6:8], env.target, atol=1e-6)
    np.testing.assert_allclose(obs[8:10], env.target - tip, atol=1e-6)


def test_same_seed_and_actions_give_identical_observations():
    actions = np.random.default_rng(0).uniform(-1, 1, size=(60, 2)).astype(np.float32)
    a = rollout(7, actions)
    b = rollout(7, actions)
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_different_seeds_give_different_starts():
    a = rollout(1, np.zeros((0, 2), dtype=np.float32))[0]
    b = rollout(2, np.zeros((0, 2), dtype=np.float32))[0]
    assert not np.array_equal(a, b)


def test_time_limit_truncates_and_does_not_terminate(env):
    # Seed 0 does not start on the target, and a zero action holds the arm still.
    env.reset(seed=0)
    info = {}
    for step in range(1, MAX_EPISODE_STEPS + 1):
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        assert not terminated
        assert truncated == (step == MAX_EPISODE_STEPS)
    assert info["is_success"] is False


def test_reward_stays_within_bounds(env):
    rng = np.random.default_rng(0)
    low, high = REWARD_BOUNDS
    for seed in range(5):
        env.reset(seed=seed)
        done = False
        while not done:
            # Bang-bang actions give the largest action-change penalty.
            action = rng.choice([-1.0, 1.0], size=2).astype(np.float32)
            _, reward, terminated, truncated, _ = env.step(action)
            assert low <= reward <= high
            done = terminated or truncated


def test_reset_covers_joint_range_and_target_ring(env):
    q, r = [], []
    for seed in range(300):
        env.reset(seed=seed)
        q.append(env.data.qpos.copy())
        r.append(np.linalg.norm(env.target))
    q, r = np.array(q), np.array(r)
    lo, hi = TARGET_RADIUS_RANGE
    assert lo <= r.min() and r.max() <= hi
    assert r.min() < lo + 0.02 and r.max() > hi - 0.01
    for j in range(2):
        jlo, jhi = env.joint_range[j]
        assert jlo <= q[:, j].min() and q[:, j].max() <= jhi
        assert q[:, j].min() < jlo + 0.2 * (jhi - jlo)
        assert q[:, j].max() > jhi - 0.2 * (jhi - jlo)


def test_hidden_target_removes_target_information():
    e = ReacherEnv(hide_target=True)
    for seed in range(5):
        obs, _ = e.reset(seed=seed)
        assert np.all(obs[6:] == 0.0)
        obs, *_ = e.step(e.action_space.sample())
        assert np.all(obs[6:] == 0.0)
    e.close()


def test_inverse_kinematics_round_trips_through_forward_kinematics():
    rng = np.random.default_rng(0)
    lo, hi = TARGET_RADIUS_RANGE
    for _ in range(200):
        radius = rng.uniform(lo, hi)
        angle = rng.uniform(-np.pi, np.pi)
        target = radius * np.array([np.cos(angle), np.sin(angle)])
        solutions = inverse_kinematics(target[0], target[1])
        assert len(solutions) == 2
        for q1, q2 in solutions:
            np.testing.assert_allclose(forward_kinematics(q1, q2), target, atol=1e-9)


def test_elbow_range_is_in_radians(env):
    np.testing.assert_allclose(env.joint_range[1], [-2.7, 2.7])


def test_arm_at_rest_stays_at_rest_under_zero_action(env):
    env.reset(seed=0)
    start = env.data.qpos.copy()
    for _ in range(20):
        env.step(np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(env.data.qpos, start, atol=1e-9)


def test_forward_kinematics_matches_the_mujoco_model(env):
    assert sum(LINK_LENGTHS) > TARGET_RADIUS_RANGE[1]
    for seed in range(10):
        env.reset(seed=seed)
        q = env.data.qpos
        np.testing.assert_allclose(
            forward_kinematics(q[0], q[1]), env.fingertip, atol=1e-9
        )


def test_success_needs_ten_still_steps_on_target(env):
    env.reset(seed=0)
    q1, q2 = inverse_kinematics(env.target[0], env.target[1])[0]
    env.set_state(np.array([q1, q2]), np.zeros(2))
    reward = 0.0
    for step in range(1, 11):
        _, reward, terminated, truncated, info = env.step(np.zeros(2, np.float32))
        assert terminated == (step == 10)
        assert info["is_success"] == (step == 10)
        assert not truncated
    assert reward > 0.9


def test_registered_with_gymnasium():
    e = gym.make("FaultLadder/Reacher-v0")
    assert isinstance(e.unwrapped, ReacherEnv)
    e.close()


@pytest.mark.slow
def test_oracle_solves_and_dummy_agents_do_not():
    assert evaluate(oracle_policy, episodes=100, seed=0)["success_rate"] >= 0.95
    assert evaluate(zero_policy, episodes=100, seed=0)["success_rate"] <= 0.05
    assert evaluate(random_policy, episodes=100, seed=0)["success_rate"] <= 0.05
