import json

import pytest

from faultladder.gate import check
from faultladder.leaderboard import build
from faultladder.score import AXES, ladder_scores

GRADED = [0.9, 0.8, 0.6, 0.4, 0.2]


def _run(root, name: str, levels: list[float], nominal: float = 0.99, seed: int = 0):
    cells = [{"axis": "nominal", "level": 0, "score": nominal, "fall_rate": 0.0}]
    cells += [
        {"axis": axis, "level": i + 1, "score": value, "fall_rate": 0.0}
        for axis in AXES
        for i, value in enumerate(levels)
    ]
    scores = ladder_scores({axis: levels for axis in AXES}, nominal)
    scores["fall_rate"] = 0.0
    report = {
        "ladder": "v1",
        "episodes_per_cell": 4096,
        "checkpoint": f"{name.removesuffix('_rescore')}.pt",
        "seed": seed,
        "scores": scores,
        "cells": cells,
    }
    (root / name).mkdir(parents=True)
    (root / name / "ladder.json").write_text(json.dumps(report))
    return root / name


def _results(root, wide: list[float], nodr: list[float] | None = None):
    for seed in range(3):
        _run(root, f"ppo_nodr_seed{seed}", nodr or [0.8, 0.5, 0.2, 0.0, 0.0], seed=seed)
        _run(root, f"ppo_default_seed{seed}", GRADED, seed=seed)
        _run(root, f"ppo_wide_seed{seed}", wide, seed=seed)
        _run(root, f"ppo_privileged_seed{seed}", wide, seed=seed)
    _run(root, "ppo_default_seed0_rescore", GRADED, seed=1)
    _run(root, "zero", [0.0] * 5, 0.0)
    _run(root, "random", [0.0] * 5, 0.0)
    return root


def test_all_six_gates_pass(tmp_path):
    result, report = check(_results(tmp_path, [1.0, 0.9, 0.8, 0.5, 0.3]))
    assert result["passed"]
    assert list(result["checks"]) == ["G1", "G2", "G3", "G4", "G5", "G6"]
    assert result["checks"]["G3"]["value"]["gap"] == pytest.approx(0.4)
    assert "FAIL" not in report
    assert "| Seed 0 | Seed 1 | Seed 2 |" in report


def test_g3_fails_when_wide_is_no_better(tmp_path):
    result, report = check(_results(tmp_path, [0.8, 0.5, 0.5, 0.0, 0.0]))
    assert not result["checks"]["G3"]["passed"]
    assert result["checks"]["G5"]["passed"]
    assert "FAIL" in report


def test_g3_fails_when_the_gap_is_within_seed_spread(tmp_path):
    root = _results(tmp_path, [1.0, 0.9, 0.8, 0.5, 0.3])
    _run(tmp_path / "x", "ppo_nodr_seed2", [1.0] * 5, seed=2).replace(
        root / "ppo_nodr_seed2_new"
    )
    (root / "ppo_nodr_seed2" / "ladder.json").unlink()
    (root / "ppo_nodr_seed2").rmdir()
    (root / "ppo_nodr_seed2_new").rename(root / "ppo_nodr_seed2")
    result, _ = check(root)
    g3 = result["checks"]["G3"]
    assert g3["value"]["gap"] >= 0.15
    assert not g3["passed"]


def test_g3_fails_with_a_single_seed(tmp_path):
    root = _results(tmp_path, [1.0, 0.9, 0.8, 0.5, 0.3])
    for seed in (1, 2):
        (root / f"ppo_wide_seed{seed}" / "ladder.json").unlink()
        (root / f"ppo_wide_seed{seed}").rmdir()
    assert not check(root)[0]["checks"]["G3"]["passed"]


def test_g5_fails_when_level_3_is_unsolved(tmp_path):
    result, _ = check(_results(tmp_path, [1.0, 1.0, 0.4, 0.4, 0.4]))
    assert result["checks"]["G5"]["failing_axes"] == list(AXES)


def test_gate_needs_the_whole_ladder(tmp_path):
    root = _results(tmp_path, [1.0, 0.9, 0.8, 0.5, 0.3])
    path = root / "zero" / "ladder.json"
    report = json.loads(path.read_text())
    report["scores"] = None
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        check(root)


def test_gate_refuses_a_rescore_of_another_checkpoint(tmp_path):
    root = _results(tmp_path, [1.0, 0.9, 0.8, 0.5, 0.3])
    path = root / "ppo_default_seed0_rescore" / "ladder.json"
    report = json.loads(path.read_text())
    report["checkpoint"] = "older.pt"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        check(root)


def test_gate_refuses_mixed_ladder_versions(tmp_path):
    root = _results(tmp_path, [1.0, 0.9, 0.8, 0.5, 0.3])
    path = root / "ppo_wide_seed1" / "ladder.json"
    report = json.loads(path.read_text())
    report["ladder"] = "v0"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        check(root)


def test_leaderboard_ranks_by_score_and_reports_seed_statistics(tmp_path):
    report = build(_results(tmp_path, [1.0, 0.9, 0.8, 0.5, 0.3]))
    rows = [line for line in report.splitlines() if line.startswith("| PPO")]
    assert rows[0].startswith("| PPO wide | 3 | 0.700 ± 0.000 |")
    assert report.index("PPO default") < report.index("PPO no randomisation")
    assert "| Zero agent | 1 | 0.000 |" in report
