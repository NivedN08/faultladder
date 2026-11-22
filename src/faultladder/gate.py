"""Check the validity gates from saved ladder runs and write ``gate.md``.

uv run python -m faultladder.gate
"""

import argparse
import json
from pathlib import Path
from typing import Any

from faultladder import score


def _load(run: str | Path) -> dict[str, Any]:
    report = json.loads((Path(run) / "ladder.json").read_text())
    if report["scores"] is None:
        raise ValueError(f"{run} does not cover the whole ladder")
    return report


def _cells(report: dict[str, Any]) -> dict[str, list[float]]:
    cells: dict[str, list[float]] = {axis: [] for axis in score.AXES}
    for cell in sorted(report["cells"], key=lambda c: c["level"]):
        if cell["level"] > 0:
            cells[cell["axis"]].append(cell["score"])
    return cells


def check(
    ppo: str | Path, rescore: str | Path, zero: str | Path, random: str | Path
) -> tuple[dict[str, Any], str]:
    """Gates G1, G2, G4 and G6 for the default policy, and a Markdown report."""
    default, again = _load(ppo), _load(rescore)
    floor = {name: _load(run) for name, run in (("zero", zero), ("random", random))}
    cells = _cells(default)
    result = score.gates(
        ppo_default=default["scores"],
        ppo_default_cells=cells,
        zero=floor["zero"]["scores"]["faultladder"],
        random=floor["random"]["scores"]["faultladder"],
        rescored=again["scores"]["faultladder"],
    )
    lines = [
        "# FaultLadder validity gates",
        "",
        f"Ladder {default['ladder']}, {default['episodes_per_cell']} episodes per "
        f"cell. Default policy: `{default['checkpoint']}`, seed {default['seed']}; "
        f"rescored with seed {again['seed']}.",
        "",
        "| Gate | Condition | Measured | Result |",
        "| --- | --- | --- | --- |",
    ]
    for name, gate in result["checks"].items():
        measured = gate.get("value", f"failing axes: {gate.get('failing_axes')}")
        if isinstance(measured, float):
            measured = f"{measured:.4f}"
        verdict = "pass" if gate["passed"] else "FAIL"
        lines.append(f"| {name} | {gate['condition']} | {measured} | {verdict} |")
    lines += [
        "",
        "G3 and G5 need the Phase 4 baselines and are not checked here.",
        "",
        "## Default policy, success rate per cell",
        "",
        "| Axis | L1 | L2 | L3 | L4 | L5 | Axis score |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for axis in score.AXES:
        row = " | ".join(f"{value:.3f}" for value in cells[axis])
        lines.append(f"| {axis} | {row} | {default['scores']['axes'][axis]:.3f} |")
    lines += [
        "",
        "| Agent | FaultLadder | Fault | Shift | Nominal | Fall rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, report in (("ppo_default", default), *floor.items()):
        s = report["scores"]
        lines.append(
            f"| {name} | {s['faultladder']:.3f} | {s['fault']:.3f} | "
            f"{s['shift']:.3f} | {s['nominal']:.3f} | {s['fall_rate']:.3f} |"
        )
    return result, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo", default="results/ppo_default_seed0")
    parser.add_argument("--rescore", default="results/ppo_default_seed0_rescore")
    parser.add_argument("--zero", default="results/zero")
    parser.add_argument("--random", default="results/random")
    parser.add_argument("--out", default="results/gate.md")
    args = parser.parse_args()
    result, report = check(args.ppo, args.rescore, args.zero, args.random)
    Path(args.out).write_text(report)
    print(report)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
