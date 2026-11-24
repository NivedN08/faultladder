"""The onset and compound tracks."""

import json

import pytest
import torch

from faultladder import ladder, score, tracks
from faultladder.evaluate import make_env, run_episodes


def test_onset_scores_average_axes_and_report_the_gap():
    static = {axis: [1.0, 1.0, 1.0, 0.5, 0.5] for axis in score.MOTOR_AXES}
    onset = {axis: [1.0, 1.0, 0.5, 0.5, 0.0] for axis in score.MOTOR_AXES}
    onset["motor_calf"] = [0.0] * 5
    result = score.onset_scores(static, onset)
    assert result["static"] == pytest.approx(0.8)
    assert result["onset"] == pytest.approx((0.6 + 0.6 + 0.0) / 3)
    assert result["gap"] == pytest.approx(0.8 - 0.4)
    assert result["axes"]["motor_calf"] == {"static": pytest.approx(0.8), "onset": 0.0}


def test_onset_scores_need_every_motor_axis_and_level():
    good = {axis: [1.0] * 5 for axis in score.MOTOR_AXES}
    with pytest.raises(ValueError):
        score.onset_scores({"motor_hip": [1.0] * 5}, good)
    with pytest.raises(ValueError):
        score.onset_scores(good, {axis: [1.0] * 4 for axis in score.MOTOR_AXES})


def test_compound_prediction_multiplies_the_kept_fractions():
    assert score.compound_prediction(0.8, 0.5, 1.0) == pytest.approx(0.4)
    # Nominal 0.9: the faults keep 8/9 and 5/9 of the nominal successes.
    assert score.compound_prediction(0.8, 0.5, 0.9) == pytest.approx(0.4 / 0.9)
    assert score.compound_prediction(0.8, 0.5, 0.0) == 0.0


def test_compound_scores_report_the_interaction():
    pairs = [
        {"observed": 0.2, "single_a": 0.8, "single_b": 0.5},
        {"observed": 1.0, "single_a": 1.0, "single_b": 1.0},
    ]
    result = score.compound_scores(pairs, 1.0)
    assert result["compound"] == pytest.approx(0.6)
    assert result["predicted"] == pytest.approx(0.7)
    assert result["interaction"] == pytest.approx(-0.1)
    with pytest.raises(ValueError):
        score.compound_scores([], 1.0)


def test_compound_pairs_skip_two_motor_axes():
    pairs = tracks.compound_pairs()
    assert len(pairs) == 18
    assert ("motor_hip", "motor_thigh") not in pairs
    assert ("motor_calf", "push") in pairs and ("payload", "latency") in pairs


def test_two_cells_combine_their_event_parameters():
    params = ladder.event_params(ladder.cell("motor_calf", 3), ladder.cell("push", 2))
    assert params[ladder.MOTOR] == {"joint_type": "calf", "strength": 0.4}
    assert params[ladder.PUSH] == {"speed": 0.75}
    assert params[ladder.PAYLOAD]["mass_fraction"] == 0.0
    with pytest.raises(ValueError):
        ladder.event_params(ladder.cell("motor_calf", 3), ladder.cell("motor_hip", 1))


@pytest.fixture(scope="module")
def setup():
    env, wrapped = make_env(4, "cpu", 0)
    yield env, wrapped
    env.close()


def _weak_motors(env) -> torch.Tensor:
    ctrl = env.scene["robot"].indexing.ctrl_ids
    gain = torch.as_tensor(env.sim.model.actuator_gainprm[:])[:, ctrl, 0]
    return (gain < 0.99 * env.sim.get_default_field("actuator_gainprm")[ctrl, 0]).sum(
        dim=1
    )


def test_onset_weakens_the_motor_only_from_the_onset_step(setup):
    env, wrapped = setup
    seen = []

    def policy(obs):
        seen.append(_weak_motors(env).clone())
        return torch.zeros(4, wrapped.num_actions)

    commands = torch.tensor([[0.5, 0.0, 0.0]]).repeat(4, 1)
    with torch.inference_mode():
        run_episodes(
            env,
            wrapped,
            policy,
            (ladder.cell("motor_hip", 5),),
            commands,
            0,
            episode_seconds=0.4,
            scored_seconds=0.1,
            onset_seconds=0.2,
        )
        assert len(seen) == 20
        assert all((s == 0).all() for s in seen[:10])
        assert all((s == 1).all() for s in seen[10:])
        with pytest.raises(ValueError):
            run_episodes(
                env,
                wrapped,
                policy,
                (ladder.cell("payload", 1),),
                commands,
                0,
                0.4,
                0.1,
                0.2,
            )


def test_compound_applies_both_perturbations(setup):
    env, wrapped = setup
    ladder.set_cell(env, ladder.cell("motor_calf", 3), ladder.cell("payload", 2))
    with torch.inference_mode():
        env.reset()
    assert (_weak_motors(env) == 1).all()
    robot = env.scene["robot"]
    mass = torch.as_tensor(env.sim.model.body_mass[:])[:, robot.indexing.body_ids]
    nominal = env.sim.get_default_field("body_mass")[robot.indexing.body_ids].sum()
    torch.testing.assert_close(
        mass.sum(dim=1) / nominal, torch.full((4,), 1.24), rtol=1e-4, atol=0
    )


def test_evaluate_writes_tracks_json(tmp_path):
    report = tracks.evaluate(
        "zero", tmp_path, num_envs=2, device="cpu", episode_seconds=0.2
    )
    assert json.loads((tmp_path / "tracks.json").read_text())["tracks"] == "v1"
    assert len(report["onset_cells"]) == 15
    assert len(report["compound_cells"]) == 36
    assert report["scores"]["onset"]["onset"] == 0.0
    assert report["scores"]["compound"]["compound"] == 0.0
