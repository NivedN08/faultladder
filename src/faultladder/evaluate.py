"""Score an agent on the FaultLadder: 36 cells, one ``ladder.json`` per run.

    uv run python -m faultladder.evaluate --agent ppo \\
        --checkpoint runs/go1_seed0/model_2999.pt --out results/ppo_default_seed0
"""

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.torch import configure_torch_backends

from faultladder import ladder, score
from faultladder.ladder import Cell
from faultladder.tasks import EVAL_TASK

AGENTS = ("ppo", "zero", "random")
Policy = Callable[[Any], torch.Tensor]


def sample_commands(
    num: int, seed: int, ranges: UniformVelocityCommandCfg.Ranges, min_speed: float
) -> torch.Tensor:
    """One (vx, vy, yaw rate) command per episode from the task's ranges.

    Commands slower than ``min_speed`` are redrawn: standing still must never
    be within the tracking tolerance.
    """
    generator = torch.Generator().manual_seed(seed)
    low = torch.tensor([ranges.lin_vel_x[0], ranges.lin_vel_y[0], ranges.ang_vel_z[0]])
    high = torch.tensor([ranges.lin_vel_x[1], ranges.lin_vel_y[1], ranges.ang_vel_z[1]])
    commands = torch.empty(num, 3)
    todo = torch.ones(num, dtype=torch.bool)
    while todo.any():
        draw = torch.rand(int(todo.sum()), 3, generator=generator)
        commands[todo] = low + draw * (high - low)
        todo = commands[:, :2].norm(dim=1) < min_speed
    return commands


def make_env(
    num_envs: int, device: str, seed: int
) -> tuple[ManagerBasedRlEnv, RslRlVecEnvWrapper]:
    configure_torch_backends()
    cfg = load_env_cfg(EVAL_TASK)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    clip = load_rl_cfg(EVAL_TASK).clip_actions
    return env, RslRlVecEnvWrapper(env, clip_actions=clip)


def make_policy(
    agent: str,
    wrapped: RslRlVecEnvWrapper,
    checkpoint: str | Path | None,
    device: str,
    seed: int,
) -> Policy:
    num_envs, num_actions = wrapped.num_envs, wrapped.num_actions
    if agent == "zero":
        return lambda obs: torch.zeros(num_envs, num_actions, device=device)
    if agent == "random":
        generator = torch.Generator(device=device).manual_seed(seed)
        return lambda obs: (
            torch.rand(num_envs, num_actions, device=device, generator=generator) * 2
            - 1
        )
    if agent != "ppo":
        raise ValueError(f"agent must be one of {AGENTS}, got {agent!r}")
    if checkpoint is None or not Path(checkpoint).is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    runner_cls = load_runner_cls(EVAL_TASK) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(load_rl_cfg(EVAL_TASK)), None, device)
    runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    return runner.get_inference_policy(device=device)


def run_cell(
    env: ManagerBasedRlEnv,
    wrapped: RslRlVecEnvWrapper,
    policy: Policy,
    cell: Cell,
    commands: torch.Tensor,
    seed: int,
    episode_seconds: float,
    scored_seconds: float,
) -> dict[str, Any]:
    """Run one episode per environment under ``cell`` and score it.

    Call inside ``torch.inference_mode()``: mjlab's buffers become inference
    tensors on the first step and cannot be reset outside it afterwards.
    """
    torch.manual_seed(seed)
    ladder.set_cell(env, cell)
    wrapped.reset()
    twist = env.command_manager.get_term("twist")
    assert isinstance(twist, UniformVelocityCommand)
    twist.vel_command_b[:] = commands
    robot = env.scene["robot"]
    steps = round(episode_seconds / env.step_dt)
    first_scored = steps - round(scored_seconds / env.step_dt)
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    lin_err = torch.zeros(env.num_envs, device=env.device)
    ang_err = torch.zeros(env.num_envs, device=env.device)
    obs = wrapped.get_observations()
    for step in range(steps):
        obs, _, dones, _ = wrapped.step(policy(obs))
        # Episodes never time out here, so a done is a fall. The fallen robot
        # respawns with a new command; it is already a failure.
        fell |= dones.bool()
        twist.vel_command_b[:] = commands
        if step >= first_scored:
            lin_vel = robot.data.root_link_lin_vel_b[:, :2]
            yaw_rate = robot.data.root_link_ang_vel_b[:, 2]
            lin_err += (commands[:, :2] - lin_vel).norm(dim=1)
            ang_err += (commands[:, 2] - yaw_rate).abs()
    lin_err /= steps - first_scored
    ang_err /= steps - first_scored
    success = score.episode_success(fell, lin_err, ang_err)
    upright = ~fell
    return {
        "axis": cell.axis,
        "level": cell.level,
        "value": cell.value,
        "score": score.cell_score(success),
        "fall_rate": float(fell.float().mean()),
        "episodes": env.num_envs,
        # Diagnostics over the episodes that stayed upright.
        "mean_lin_err": float(lin_err[upright].mean()) if upright.any() else None,
        "mean_ang_err": float(ang_err[upright].mean()) if upright.any() else None,
    }


def summarize(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    """FaultLadder scores from cell results, or None if any cell is missing."""
    by_axis: dict[str, dict[int, float]] = {}
    nominal = None
    for cell in cells:
        if cell["level"] == 0:
            nominal = cell["score"]
        else:
            by_axis.setdefault(cell["axis"], {})[cell["level"]] = cell["score"]
    levels = range(1, score.NUM_LEVELS + 1)
    complete = nominal is not None and all(
        set(by_axis.get(axis, {})) == set(levels) for axis in score.AXES
    )
    if not complete:
        return None
    assert nominal is not None
    scores = score.ladder_scores(
        {axis: [by_axis[axis][i] for i in levels] for axis in score.AXES}, nominal
    )
    perturbed = [cell["fall_rate"] for cell in cells if cell["level"] > 0]
    scores["fall_rate"] = sum(perturbed) / len(perturbed)
    return scores


def evaluate(
    agent: str,
    out_dir: str | Path,
    checkpoint: str | Path | None = None,
    num_envs: int | None = None,
    device: str = "cuda:0",
    seed: int = 0,
    only: list[str] | None = None,
    episode_seconds: float | None = None,
) -> dict[str, Any]:
    """Run every ladder cell (or those named in ``only``) and write ladder.json."""
    spec = ladder.load()
    num_envs = int(num_envs or spec["episodes"])
    episode_seconds = float(episode_seconds or spec["episode_seconds"])
    scored_seconds = episode_seconds * spec["scored_seconds"] / spec["episode_seconds"]
    cells = [c for c in ladder.cells() if only is None or c.name in only]
    if not cells:
        raise ValueError(f"no ladder cell matches {only}")
    start = time.perf_counter()
    env, wrapped = make_env(num_envs, device, seed)
    policy = make_policy(agent, wrapped, checkpoint, device, seed)
    twist_cfg = env.cfg.commands["twist"]
    assert isinstance(twist_cfg, UniformVelocityCommandCfg)
    commands = sample_commands(
        num_envs, seed, twist_cfg.ranges, spec["min_command_speed"]
    ).to(device)
    results = []
    for cell in cells:
        cell_start = time.perf_counter()
        with torch.inference_mode():
            result = run_cell(
                env,
                wrapped,
                policy,
                cell,
                commands,
                seed,
                episode_seconds,
                scored_seconds,
            )
        results.append(result)
        print(
            f"{cell.name:>14}  score {result['score']:.3f}  "
            f"fell {result['fall_rate']:.3f}  "
            f"{time.perf_counter() - cell_start:.0f}s",
            flush=True,
        )
    env.close()
    report = {
        "ladder": spec["version"],
        "task": EVAL_TASK,
        "agent": agent,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "seed": seed,
        "episodes_per_cell": num_envs,
        "episode_seconds": episode_seconds,
        "scored_seconds": scored_seconds,
        "tolerances": {"lin": score.LIN_TOL, "ang": score.ANG_TOL},
        "device": device,
        "versions": {p: version(p) for p in ("mjlab", "mujoco", "mujoco-warp")},
        "wall_seconds": time.perf_counter() - start,
        "scores": summarize(results),
        "cells": results,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ladder.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=AGENTS, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", required=True, help="directory for ladder.json")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="*", help="cell names, e.g. nominal push/3")
    args = parser.parse_args()
    report = evaluate(
        args.agent,
        args.out,
        checkpoint=args.checkpoint,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        only=args.only,
    )
    print(json.dumps(report["scores"], indent=2))


if __name__ == "__main__":
    main()
