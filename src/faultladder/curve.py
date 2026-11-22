"""Read a training reward curve and find where it stops improving.

Usage: uv run python -m faultladder.curve runs/<run directory> --out results/x.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REWARD_TAG = "Train/mean_reward"
FINAL_WINDOW = 100


def read_scalar(log_dir: str | Path, tag: str = REWARD_TAG) -> list[float]:
    """Values of ``tag`` by iteration, merged over every event file in ``log_dir``.

    A resumed run logs some iterations twice; the later file wins.
    """
    by_step: dict[int, float] = {}
    for path in sorted(Path(log_dir).glob("events.out.tfevents.*")):
        events = EventAccumulator(str(path), size_guidance={"scalars": 0})
        events.Reload()
        if tag in events.Tags()["scalars"]:
            for event in events.Scalars(tag):
                by_step[event.step] = event.value
    if not by_step:
        raise ValueError(f"no {tag} values in {log_dir}")
    return [by_step[step] for step in sorted(by_step)]


def plateau(
    values: list[float], window: int = 500, tolerance: float = 0.02
) -> int | None:
    """First index from which ``values`` stay within ``tolerance`` of the final
    value for ``window`` entries, or None if that never happens.

    The final value is the mean of the last ``FINAL_WINDOW`` entries.
    """
    tail = values[-FINAL_WINDOW:]
    final = sum(tail) / len(tail)
    band = tolerance * abs(final)
    run = 0
    for index, value in enumerate(values):
        run = run + 1 if abs(value - final) <= band else 0
        if run == window:
            return index - window + 1
    return None


def smooth(values: list[float], window: int = FINAL_WINDOW) -> list[float]:
    """Trailing mean of ``values`` over up to ``window`` entries."""
    out: list[float] = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        out.append(total / min(index + 1, window))
    return out


def summarize(log_dir: str | Path) -> dict[str, Any]:
    values = read_scalar(log_dir)
    tail = values[-FINAL_WINDOW:]
    start = plateau(smooth(values))
    return {
        "tag": REWARD_TAG,
        "iterations_logged": len(values),
        "final_mean_reward": sum(tail) / len(tail),
        "plateau_iteration": start,
        "iteration_budget": None if start is None else math.ceil(1.2 * start),
        "plateau_iteration_unsmoothed": plateau(values),
        "rule": "first iteration from which the trailing "
        f"{FINAL_WINDOW}-iteration mean of mean reward stays within 2% of the "
        f"mean of the last {FINAL_WINDOW} iterations for 500 iterations; the "
        "budget is 1.2 times that. Unsmoothed, single noisy iterations leave "
        "the 2% band and reset the count.",
        "reward_every_250": [round(value, 3) for value in values[::250]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir")
    parser.add_argument("--out")
    args = parser.parse_args()
    summary = summarize(args.log_dir)
    text = json.dumps(summary, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
