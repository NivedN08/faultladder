import pytest
import torch

from faultladder.score import (
    AXES,
    cell_score,
    episode_success,
    gates,
    ladder_scores,
)


def test_episode_success_needs_no_fall_and_both_errors_within_tolerance():
    fell = torch.tensor([False, True, False, False, False])
    lin = torch.tensor([0.25, 0.10, 0.26, 0.10, 0.00])
    ang = torch.tensor([0.50, 0.10, 0.10, 0.51, 0.00])
    assert episode_success(fell, lin, ang).tolist() == [True, False, False, False, True]


def test_cell_score_is_the_fraction_of_successes():
    assert cell_score(torch.tensor([True, False, False, True])) == 0.5
    assert cell_score(torch.tensor([False, False])) == 0.0


def _cells(**overrides: list[float]) -> dict[str, list[float]]:
    cells = {axis: [1.0, 1.0, 1.0, 1.0, 1.0] for axis in AXES}
    cells.update(overrides)
    return cells


def test_ladder_scores_hand_computed():
    cells = {
        "motor_hip": [1.0, 0.8, 0.6, 0.4, 0.2],  # mean 0.6
        "motor_thigh": [0.9, 0.7, 0.5, 0.3, 0.1],  # mean 0.5
        "motor_calf": [1.0, 1.0, 0.5, 0.0, 0.0],  # mean 0.5
        "payload": [1.0, 1.0, 1.0, 1.0, 1.0],  # mean 1.0
        "friction": [0.5, 0.5, 0.5, 0.5, 0.5],  # mean 0.5
        "latency": [1.0, 0.5, 0.0, 0.0, 0.0],  # mean 0.3
        "push": [0.8, 0.8, 0.8, 0.8, 0.8],  # mean 0.8
    }
    scores = ladder_scores(cells, nominal=0.97)
    assert scores["axes"] == pytest.approx(
        {
            "motor_hip": 0.6,
            "motor_thigh": 0.5,
            "motor_calf": 0.5,
            "payload": 1.0,
            "friction": 0.5,
            "latency": 0.3,
            "push": 0.8,
        }
    )
    assert scores["faultladder"] == pytest.approx(4.2 / 7)
    assert scores["fault"] == pytest.approx((0.6 + 0.5 + 0.5 + 0.8) / 4)
    assert scores["shift"] == pytest.approx((1.0 + 0.5 + 0.3) / 3)
    assert scores["nominal"] == 0.97


def test_ladder_scores_ignores_the_nominal_cell():
    assert ladder_scores(_cells(), nominal=0.0)["faultladder"] == 1.0


def test_ladder_scores_rejects_missing_axes_and_wrong_level_counts():
    with pytest.raises(ValueError):
        ladder_scores({"push": [1.0] * 5}, nominal=1.0)
    with pytest.raises(ValueError):
        ladder_scores(_cells(push=[1.0] * 4), nominal=1.0)


def test_gates_at_the_thresholds_pass():
    informative = {axis: [0.8, 0.7, 0.6, 0.55, 0.5] for axis in AXES}
    result = gates(
        ppo_default=ladder_scores(informative, nominal=0.95),
        ppo_default_cells=informative,
        zero=0.05,
        random=0.0,
        rescored=ladder_scores(informative, nominal=0.95)["faultladder"] + 0.02,
    )
    assert result["passed"]
    assert all(check["passed"] for check in result["checks"].values())


def test_each_gate_fails_on_its_own_condition():
    informative = {axis: [0.8, 0.7, 0.6, 0.55, 0.5] for axis in AXES}
    base = dict(
        ppo_default=ladder_scores(informative, nominal=0.95),
        ppo_default_cells=informative,
        zero=0.0,
        random=0.0,
        rescored=ladder_scores(informative, nominal=0.95)["faultladder"],
    )

    def failed(**changes: object) -> list[str]:
        result = gates(**{**base, **changes})  # type: ignore[arg-type]
        assert not result["passed"]
        return [name for name, check in result["checks"].items() if not check["passed"]]

    assert failed(ppo_default=ladder_scores(informative, nominal=0.94)) == ["G1"]
    assert failed(random=0.06) == ["G2"]
    easy = {**informative, "push": [0.79, 0.7, 0.6, 0.55, 0.5]}
    assert failed(ppo_default_cells=easy) == ["G4"]
    hard = {**informative, "latency": [0.8, 0.7, 0.6, 0.55, 0.51]}
    assert failed(ppo_default_cells=hard) == ["G4"]
    assert failed(rescored=base["rescored"] + 0.021) == ["G6"]  # type: ignore[operator]


def test_g4_reports_the_axes_that_fail():
    informative = {axis: [0.8, 0.7, 0.6, 0.55, 0.5] for axis in AXES}
    cells = {**informative, "payload": [1.0, 1.0, 1.0, 1.0, 0.9]}
    result = gates(
        ppo_default=ladder_scores(cells, nominal=1.0),
        ppo_default_cells=cells,
        zero=0.0,
        random=0.0,
        rescored=ladder_scores(cells, nominal=1.0)["faultladder"],
    )
    assert result["checks"]["G4"]["failing_axes"] == ["payload"]
