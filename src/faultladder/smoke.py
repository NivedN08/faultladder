"""Headless smoke rollout of an mjlab task with a zero or random agent.

Usage: uv run python -m faultladder.smoke --agent zero --num-envs 16 --steps 200
"""

import argparse
import json
import time
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

BASE_TASK = "Mjlab-Velocity-Flat-Unitree-Go1"


def rollout(
    task_id: str = BASE_TASK,
    agent: str = "zero",
    num_envs: int = 16,
    steps: int = 200,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    """Step ``task_id`` with a dummy agent and return summary numbers."""
    if agent not in ("zero", "random"):
        raise ValueError(f"unknown agent: {agent}")
    torch.manual_seed(seed)
    env_cfg = load_env_cfg(task_id)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    agent_cfg = load_rl_cfg(task_id)
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=env_cfg, device=device),
        clip_actions=agent_cfg.clip_actions,
    )
    shape = (num_envs, env.num_actions)
    env.reset()
    reward_sum = 0.0
    dones = 0
    finite = True
    start = time.perf_counter()
    for _ in range(steps):
        if agent == "zero":
            action = torch.zeros(shape, device=device)
        else:
            action = 2 * torch.rand(shape, device=device) - 1
        obs, reward, done, _ = env.step(action)
        reward_sum += float(reward.sum())
        dones += int(done.sum())
        finite = finite and all(
            bool(torch.isfinite(torch.as_tensor(v)).all()) for v in obs.values()
        )
    elapsed = time.perf_counter() - start
    env.close()
    return {
        "task": task_id,
        "agent": agent,
        "num_envs": num_envs,
        "steps": steps,
        "device": device,
        "mean_reward_per_step": reward_sum / (steps * num_envs),
        "episodes_ended": dones,
        "observations_finite": finite,
        "env_steps_per_second": steps * num_envs / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=BASE_TASK)
    parser.add_argument("--agent", choices=("zero", "random"), default="zero")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    out = rollout(args.task, args.agent, args.num_envs, args.steps, args.device)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
