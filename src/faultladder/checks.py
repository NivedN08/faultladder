"""Environment checks: zero agent, random agent, oracle, determinism, speed.

Run these on every new or changed environment before training anything.

    uv run python -m faultladder.checks --out results/reacher.json
"""

import argparse
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from faultladder.reacher.env import ACTION_SCALE, ReacherEnv
from faultladder.reacher.oracle import joint_targets
from faultladder.results import update_results

Policy = Callable[[ReacherEnv, np.ndarray], np.ndarray]

# Evaluation seeds start here; training uses seeds below it.
EVAL_SEED_OFFSET = 1_000_000


def zero_policy(env: ReacherEnv, obs: np.ndarray) -> np.ndarray:
    return np.zeros(2, dtype=np.float32)


def random_policy(env: ReacherEnv, obs: np.ndarray) -> np.ndarray:
    return env.action_space.sample()


def oracle_policy(env: ReacherEnv, obs: np.ndarray) -> np.ndarray:
    """Drive both joints to the inverse-kinematics solution. Reads env state."""
    q = env.data.qpos
    offset = (joint_targets(q, env.target) - q) / ACTION_SCALE
    return np.clip(offset, -1.0, 1.0).astype(np.float32)


def evaluate(
    policy: Policy,
    episodes: int = 1000,
    seed: int = 0,
    hide_target: bool = False,
) -> dict[str, Any]:
    """Success rate over ``episodes`` episodes, each reset with its own seed."""
    env = ReacherEnv(hide_target=hide_target)
    env.action_space.seed(seed)
    successes, returns, lengths = 0, [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED_OFFSET + seed + episode)
        total, steps, done, info = 0.0, 0, False, {}
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(env, obs))
            total += reward
            steps += 1
            done = terminated or truncated
        successes += bool(info["is_success"])
        returns.append(total)
        lengths.append(steps)
    env.close()
    return {
        "episodes": episodes,
        "seed": seed,
        "success_rate": successes / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
    }


def determinism(steps: int = 200, seed: int = 0) -> dict[str, Any]:
    """Largest observation difference between two runs of one seed and actions."""
    actions = np.random.default_rng(seed).uniform(-1, 1, (steps, 2)).astype(np.float32)
    runs = []
    for _ in range(2):
        env = ReacherEnv()
        obs = [env.reset(seed=seed)[0]]
        for action in actions:
            o, _, terminated, truncated, _ = env.step(action)
            obs.append(o)
            if terminated or truncated:
                break
        runs.append(np.array(obs))
        env.close()
    same_length = len(runs[0]) == len(runs[1])
    diff = float(np.abs(runs[0] - runs[1]).max()) if same_length else float("inf")
    return {"steps": len(runs[0]) - 1, "max_abs_difference": diff}


def speed(steps: int = 20_000) -> dict[str, Any]:
    """Single-environment steps per second with random actions, resets included."""
    env = ReacherEnv()
    env.action_space.seed(0)
    env.reset(seed=0)
    start = time.perf_counter()
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - start
    env.close()
    return {"steps": steps, "steps_per_second": steps / elapsed}


def run_all(episodes: int = 1000, seed: int = 0) -> dict[str, Any]:
    return {
        "zero": evaluate(zero_policy, episodes, seed),
        "random": evaluate(random_policy, episodes, seed),
        "oracle": evaluate(oracle_policy, episodes, seed),
        "determinism": determinism(),
        "speed": speed(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/reacher.json")
    args = parser.parse_args()
    command = (
        f"uv run python -m faultladder.checks --episodes {args.episodes} "
        f"--seed {args.seed} --out {args.out}"
    )
    for name, value in run_all(args.episodes, args.seed).items():
        update_results(args.out, name, {**value, "command": command})
        print(name, value)


if __name__ == "__main__":
    main()
