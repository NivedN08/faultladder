"""Seed statistics and the tolerance-sensitivity report."""

import json

import pytest
import torch

from faultladder import score, sensitivity
from faultladder.evaluate import summarize
from faultladder.score import AXES


def test_success_scales_both_tolerances():
    fell = torch.tensor([False, False, False, True])
    lin = torch.tensor([0.2, 0.3, 0.2, 0.0])
    ang = torch.tensor([0.4, 0.4, 0.9, 0.0])
    assert score.episode_success(fell, lin, ang).tolist() == [True, False, False, False]
    assert score.episode_success(fell, lin, ang, 2.0).tolist() == [
        True,
        True,
        True,
        False,
    ]
    assert not score.episode_success(fell, lin, ang, 0.5).any()


def test_interval_is_the_student_t_half_width():
    assert score.interval([0.5]) == 0.0
    # sd 0.1 over 5 seeds: 2.776 * 0.1 / sqrt(5).
    values = [0.4, 0.45, 0.5, 0.55, 0.6]
    assert score.interval(values) == pytest.approx(
        2.776 * score.spread(values) / 5**0.5
    )


def test_rank_agreement_is_kendalls_tau():
    assert score.rank_agreement([3, 2, 1], [0.9, 0.5, 0.1]) == 1.0
    assert score.rank_agreement([3, 2, 1], [0.1, 0.5, 0.9]) == -1.0
    assert score.rank_agreement([3, 2, 1], [0.9, 0.1, 0.5]) == pytest.approx(1 / 3)


def _run(root, name: str, strict: float, loose: float) -> None:
    def cell(axis: str, level: int) -> dict:
        by_tolerance = {
            f"{s:g}": strict if s <= 1.0 else loose for s in (0.5, 0.75, 1.0)
        }
        by_tolerance |= {"1.5": loose, "2": loose}
        return {
            "axis": axis,
            "level": level,
            "score": strict,
            "fall_rate": 0.0,
            "by_tolerance": by_tolerance,
        }

    cells = [cell("nominal", 0)] + [cell(a, i) for a in AXES for i in range(1, 6)]
    report = {
        "ladder": "v1",
        "episodes_per_cell": 4096,
        "seed": 0,
        "scores": summarize(cells),
        "cells": cells,
    }
    (root / name).mkdir(parents=True)
    (root / name / "ladder.json").write_text(json.dumps(report))


def test_summarize_reads_another_tolerance(tmp_path):
    _run(tmp_path, "ppo_wide_seed0", 0.6, 0.9)
    cells = json.loads((tmp_path / "ppo_wide_seed0" / "ladder.json").read_text())[
        "cells"
    ]
    strict, loose = summarize(cells), summarize(cells, 2.0)
    assert strict is not None and loose is not None
    assert strict["faultladder"] == pytest.approx(0.6)
    assert loose["faultladder"] == pytest.approx(0.9)


def test_report_shows_a_ranking_that_flips_with_the_tolerance(tmp_path):
    _run(tmp_path, "ppo_wide_seed0", 0.6, 0.7)
    _run(tmp_path, "ppo_nodr_seed0", 0.5, 0.8)
    report = sensitivity.build(tmp_path)
    assert "| PPO wide | 0.600 | 0.600 | 0.600 | 0.700 | 0.700 |" in report
    assert "| Rank agreement | 1.000 | 1.000 | 1.000 | -1.000 | -1.000 |" in report
    with pytest.raises(ValueError):
        sensitivity.build(tmp_path / "missing")
