import json

import pytest

from faultladder.gate import check
from faultladder.score import AXES, ladder_scores


def _run(tmp_path, name: str, levels: list[float], nominal: float):
    cells = [{"axis": "nominal", "level": 0, "score": nominal, "fall_rate": 0.0}]
    cells += [
        {"axis": axis, "level": i + 1, "score": value, "fall_rate": 0.0}
        for axis in AXES
        for i, value in enumerate(levels)
    ]
    scores = ladder_scores({axis: levels for axis in AXES}, nominal)
    scores["fall_rate"] = 0.0
    report = {
        "ladder": "v0",
        "episodes_per_cell": 4096,
        "checkpoint": f"{name}.pt",
        "seed": 0,
        "scores": scores,
        "cells": cells,
    }
    (tmp_path / name).mkdir()
    (tmp_path / name / "ladder.json").write_text(json.dumps(report))
    return tmp_path / name


def test_gate_report_passes_and_fails(tmp_path):
    good = _run(tmp_path, "ppo", [0.9, 0.8, 0.6, 0.4, 0.2], 0.99)
    floor = _run(tmp_path, "zero", [0.0] * 5, 0.0)
    result, report = check(good, good, floor, floor)
    assert result["passed"]
    assert "| G4 |" in report and "FAIL" not in report
    flat = _run(tmp_path, "flat", [0.9] * 5, 0.99)
    result, report = check(flat, flat, floor, floor)
    assert not result["passed"]
    assert "FAIL" in report


def test_gate_needs_the_whole_ladder(tmp_path):
    run = _run(tmp_path, "ppo", [0.9, 0.8, 0.6, 0.4, 0.2], 0.99)
    report = json.loads((run / "ladder.json").read_text())
    report["scores"] = None
    (run / "ladder.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        check(run, run, run, run)
