import json

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from faultladder import ladder
from faultladder.evaluate import evaluate, sample_commands, summarize
from faultladder.score import AXES
from faultladder.tasks import EVAL_TASK


def _ranges() -> UniformVelocityCommandCfg.Ranges:
    twist = load_env_cfg(EVAL_TASK).commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    return twist.ranges


def test_commands_are_seeded_in_range_and_never_slow():
    commands = sample_commands(2000, seed=3, ranges=_ranges(), min_speed=0.4)
    assert torch.equal(commands, sample_commands(2000, 3, _ranges(), 0.4))
    assert not torch.equal(commands, sample_commands(2000, 4, _ranges(), 0.4))
    assert commands[:, :2].norm(dim=1).min() >= 0.4
    assert commands[:, :2].abs().max() <= 1.0
    assert commands[:, 2].abs().max() <= 0.5
    # A standing robot's error is the commanded speed: always out of tolerance.
    assert commands[:, :2].norm(dim=1).min() > 0.25


def _cell(axis: str, level: int, value: float) -> dict:
    return {"axis": axis, "level": level, "score": value, "fall_rate": 1 - value}


def test_summarize_needs_every_cell():
    cells = [_cell("nominal", 0, 1.0)]
    cells += [_cell(axis, level, 0.5) for axis in AXES for level in range(1, 6)]
    scores = summarize(cells)
    assert scores is not None
    assert scores["faultladder"] == pytest.approx(0.5)
    assert scores["nominal"] == 1.0
    assert scores["fall_rate"] == pytest.approx(0.5)
    assert summarize(cells[:-1]) is None
    assert summarize(cells[1:]) is None


def test_evaluate_writes_ladder_json(tmp_path):
    report = evaluate(
        "zero",
        tmp_path,
        num_envs=8,
        device="cpu",
        only=["nominal", "push/1"],
        episode_seconds=1.0,
    )
    assert json.loads((tmp_path / "ladder.json").read_text()) == report
    assert [c["axis"] for c in report["cells"]] == [ladder.NOMINAL, "push"]
    assert report["scores"] is None
    assert report["scored_seconds"] == pytest.approx(0.8)
    for cell in report["cells"]:
        assert cell["episodes"] == 8
        assert 0.0 <= cell["score"] <= 1.0
    # A robot that stands still never tracks a command of at least 0.4 m/s.
    assert report["cells"][0]["score"] == 0.0


def test_ppo_needs_a_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate("ppo", tmp_path, num_envs=2, device="cpu", only=["nominal"])
