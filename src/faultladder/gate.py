"""Check the six validity gates from saved ladder runs and write ``gate.md``.

uv run python -m faultladder.gate
"""

import argparse
import json
from pathlib import Path
from typing import Any

from faultladder import score

BASELINES = {
    "nodr": "PPO no randomisation",
    "default": "PPO default",
    "wide": "PPO wide",
    "privileged": "PPO privileged",
    "recurrent": "PPO wide, recurrent",
}
ROBOTS = {"go1": "Unitree Go1", "g1": "Unitree G1"}


def robot_label(report: dict[str, Any]) -> str:
    """The robot a run was scored on; runs from before the G1 are Go1 runs."""
    return ROBOTS[report.get("robot", "go1")]


# The baselines the gates are defined on; the others only join the leaderboard.
GATE_BASELINES = ("nodr", "default", "wide", "privileged")
COMMANDS = {
    "G1": "uv run modal run modal_app.py::baselines",
    "G2": "uv run modal run modal_app.py::ladder --agent zero (and --agent random)",
    "G3": "uv run modal run modal_app.py::baselines",
    "G4": "uv run modal run modal_app.py::baselines",
    "G5": "uv run modal run modal_app.py::baselines",
    "G6": "uv run modal run modal_app.py::ladder --name ppo_default_seed0_rescore "
    "--baseline default --seed 0 --eval-seed 1",
}


def load(run: str | Path) -> dict[str, Any]:
    report = json.loads((Path(run) / "ladder.json").read_text())
    if report["scores"] is None:
        raise ValueError(f"{run} does not cover the whole ladder")
    return report


def cells(report: dict[str, Any]) -> dict[str, list[float]]:
    """Success rate at levels 1 to 5 of every axis."""
    out: dict[str, list[float]] = {axis: [] for axis in score.AXES}
    for cell in sorted(report["cells"], key=lambda c: c["level"]):
        if cell["level"] > 0:
            out[cell["axis"]].append(cell["score"])
    return out


def load_baselines(results: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Every ``ppo_<baseline>_seed<n>`` run under ``results``, in seed order."""
    runs: dict[str, list[dict[str, Any]]] = {}
    for name in BASELINES:
        paths = [
            p
            for p in Path(results).glob(f"ppo_{name}_seed*")
            if p.name.removeprefix(f"ppo_{name}_seed").isdigit()
        ]
        paths.sort(key=lambda p: int(p.name.removeprefix(f"ppo_{name}_seed")))
        runs[name] = [
            {**load(p), "train_seed": int(p.name.removeprefix(f"ppo_{name}_seed"))}
            for p in paths
        ]
    return runs


def _measured(gate: dict[str, Any]) -> str:
    parts = []
    value = gate.get("value")
    if isinstance(value, dict):
        parts += [f"{k} {v:.4f}" for k, v in value.items()]
    elif value is not None:
        parts.append(f"{value:.4f}")
    if gate.get("failing_axes"):
        parts.append(f"failing axes: {', '.join(gate['failing_axes'])}")
    return "; ".join(parts) or "all axes pass"


def check(results: str | Path) -> tuple[dict[str, Any], str]:
    """All six gates from the runs under ``results``, and a Markdown report.

    G1, G4 and G6 use the default policy of the lowest seed.
    """
    results = Path(results)
    runs = load_baselines(results)
    for name in GATE_BASELINES:
        if not runs[name]:
            raise ValueError(f"no ppo_{name}_seed* runs under {results}")
    default = runs["default"][0]
    again = load(results / f"ppo_default_seed{default['train_seed']}_rescore")
    if again["checkpoint"] != default["checkpoint"]:
        raise ValueError(
            "the rescore is of a different checkpoint than the default run"
        )
    floor = {name: load(results / name) for name in ("zero", "random")}
    everything = [default, again, *floor.values()]
    everything += [r for reports in runs.values() for r in reports]
    if len({r["ladder"] for r in everything}) != 1:
        raise ValueError("the runs were scored on different ladder versions")
    default_cells = cells(default)
    first = score.gates(
        ppo_default=default["scores"],
        ppo_default_cells=default_cells,
        zero=floor["zero"]["scores"]["faultladder"],
        random=floor["random"]["scores"]["faultladder"],
        rescored=again["scores"]["faultladder"],
    )
    level3 = {
        axis: [cells(r)[axis][2] for r in runs["privileged"]] for axis in score.AXES
    }
    second = score.baseline_gates(
        nodr=[r["scores"]["faultladder"] for r in runs["nodr"]],
        wide=[r["scores"]["faultladder"] for r in runs["wide"]],
        privileged_level3=level3,
    )
    checks = dict(sorted({**first["checks"], **second["checks"]}.items()))
    result = {"checks": checks, "passed": all(c["passed"] for c in checks.values())}
    seeds = ", ".join(str(r["train_seed"]) for r in runs["wide"])
    lines = [
        "# FaultLadder validity gates",
        "",
        f"{robot_label(default)}, ladder {default['ladder']}, "
        f"{default['episodes_per_cell']} episodes per "
        f"cell. G1, G4 and G6 use the default policy of training seed "
        f"{default['train_seed']}; G3 and G5 use training seeds {seeds}. "
        "The seed spread in G3 is the larger of the two baselines' standard "
        "deviations across seeds; G5 uses the mean across seeds.",
        "",
        "Every row is recomputed from the saved runs by "
        "`uv run python -m faultladder.gate`. The last column is the command that "
        "produced those runs.",
        "",
        "| Gate | Condition | Measured | Result | Runs produced by |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, gate in checks.items():
        verdict = "pass" if gate["passed"] else "FAIL"
        lines.append(
            f"| {name} | {gate['condition']} | {_measured(gate)} | {verdict} | "
            f"`{COMMANDS[name]}` |"
        )
    lines += [
        "",
        "All six gates pass." if result["passed"] else "At least one gate FAILS.",
        "",
        "## Default policy, success rate per cell",
        "",
        "| Axis | L1 | L2 | L3 | L4 | L5 | Axis score |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for axis in score.AXES:
        row = " | ".join(f"{value:.3f}" for value in default_cells[axis])
        lines.append(f"| {axis} | {row} | {default['scores']['axes'][axis]:.3f} |")
    lines += [
        "",
        "## Privileged policy at level 3, per seed",
        "",
        "| Axis | "
        + " | ".join(f"Seed {r['train_seed']}" for r in runs["privileged"])
        + " |",
        "| --- |" + " --- |" * len(runs["privileged"]),
    ]
    for axis in score.AXES:
        lines.append(
            f"| {axis} | " + " | ".join(f"{v:.3f}" for v in level3[axis]) + " |"
        )
    return result, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/gate.md")
    args = parser.parse_args()
    result, report = check(args.results)
    Path(args.out).write_text(report)
    print(report)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
