"""Score an agent on the two extra tracks; one ``tracks.json`` per run.

Onset: a motor fault appears mid-episode instead of from the start.
Compound: two ladder axes are perturbed together.

    uv run python -m faultladder.tracks --agent ppo \\
        --checkpoint runs/go1_seed0/model_1319.pt --out results/ppo_default_seed0
"""

import argparse
import json
import time
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from faultladder import ladder, score
from faultladder.evaluate import (
    AGENTS,
    make_env,
    make_policy,
    run_episodes,
    sample_commands,
)
from faultladder.ladder import Cell
from faultladder.tasks import EVAL_TASK, TASK_ROBOT

TRACKS_VERSION = "v1"
ONSET_SECONDS = 5.0  # the fault appears here, after the robot is walking
ONSET_SCORED_SECONDS = 4.0  # the final seconds scored; one second to recover
COMPOUND_LEVELS = (2, 3)


def compound_pairs() -> list[tuple[str, str]]:
    """Every pair of axes except two motor axes: 18 pairs."""
    return [
        (a, b)
        for a, b in combinations(score.AXES, 2)
        if not (a in score.MOTOR_AXES and b in score.MOTOR_AXES)
    ]


def evaluate(
    agent: str,
    out_dir: str | Path,
    checkpoint: str | Path | None = None,
    num_envs: int | None = None,
    device: str = "cuda:0",
    seed: int = 0,
    task: str = EVAL_TASK,
    baseline: str | None = None,
    episode_seconds: float | None = None,
) -> dict[str, Any]:
    """Run both tracks and write ``tracks.json``."""
    robot = TASK_ROBOT[task]
    spec = ladder.load(robot)
    num_envs = int(num_envs or spec["episodes"])
    full = float(spec["episode_seconds"])
    episode_seconds = float(episode_seconds or full)
    scale = episode_seconds / full
    start = time.perf_counter()
    env, wrapped = make_env(num_envs, device, seed, task)
    policy = make_policy(agent, wrapped, checkpoint, device, seed, task)
    twist_cfg = env.cfg.commands["twist"]
    assert isinstance(twist_cfg, UniformVelocityCommandCfg)
    commands = sample_commands(
        num_envs, seed, twist_cfg.ranges, spec["min_command_speed"]
    ).to(device)

    def run(cells: tuple[Cell, ...], scored: float, onset: float = 0.0) -> dict:
        with torch.inference_mode():
            result = run_episodes(
                env,
                wrapped,
                policy,
                cells,
                commands,
                seed,
                episode_seconds,
                scored * scale,
                onset * scale,
            )
        name = "+".join(cell.name for cell in cells)
        print(
            f"{name:>24}  onset {onset:.0f}s  score {result['score']:.3f}", flush=True
        )
        return result

    levels = range(1, score.NUM_LEVELS + 1)
    onset_cells = []
    for axis in score.MOTOR_AXES:
        for level in levels:
            cell = ladder.cell(axis, level, robot)
            onset_cells.append(
                {
                    "axis": axis,
                    "level": level,
                    "static": run((cell,), ONSET_SCORED_SECONDS),
                    "onset": run((cell,), ONSET_SCORED_SECONDS, ONSET_SECONDS),
                }
            )
    onset = score.onset_scores(
        *(
            {
                axis: [c[kind]["score"] for c in onset_cells if c["axis"] == axis]
                for axis in score.MOTOR_AXES
            }
            for kind in ("static", "onset")
        )
    )

    scored = float(spec["scored_seconds"])
    nominal = run((ladder.nominal_cell(),), scored)
    singles = {
        (axis, level): run((ladder.cell(axis, level, robot),), scored)
        for axis in score.AXES
        for level in COMPOUND_LEVELS
    }
    compound_cells = []
    for level in COMPOUND_LEVELS:
        for a, b in compound_pairs():
            result = run(
                (ladder.cell(a, level, robot), ladder.cell(b, level, robot)), scored
            )
            single_a = singles[a, level]["score"]
            single_b = singles[b, level]["score"]
            compound_cells.append(
                {
                    "a": a,
                    "b": b,
                    "level": level,
                    "observed": result["score"],
                    "single_a": single_a,
                    "single_b": single_b,
                    "predicted": score.compound_prediction(
                        single_a, single_b, nominal["score"]
                    ),
                    "fall_rate": result["fall_rate"],
                }
            )
    compound = score.compound_scores(compound_cells, nominal["score"])
    env.close()
    report = {
        "tracks": TRACKS_VERSION,
        "ladder": spec["version"],
        "robot": robot,
        "task": task,
        "agent": agent,
        "baseline": baseline or agent,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "seed": seed,
        "episodes_per_cell": num_envs,
        "episode_seconds": episode_seconds,
        "onset_seconds": ONSET_SECONDS * scale,
        "onset_scored_seconds": ONSET_SCORED_SECONDS * scale,
        "compound_levels": list(COMPOUND_LEVELS),
        "tolerances": {"lin": score.LIN_TOL, "ang": score.ANG_TOL},
        "device": device,
        "versions": {p: version(p) for p in ("mjlab", "mujoco", "mujoco-warp")},
        "wall_seconds": time.perf_counter() - start,
        "scores": {"onset": onset, "compound": compound},
        "nominal": nominal["score"],
        "onset_cells": onset_cells,
        "compound_cells": compound_cells,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tracks.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=AGENTS, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", required=True, help="directory for tracks.json")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default=EVAL_TASK, help="evaluation task")
    parser.add_argument("--baseline", help="leaderboard name, e.g. wide")
    args = parser.parse_args()
    report = evaluate(
        args.agent,
        args.out,
        checkpoint=args.checkpoint,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        task=args.task,
        baseline=args.baseline,
    )
    print(json.dumps(report["scores"], indent=2))


if __name__ == "__main__":
    main()
