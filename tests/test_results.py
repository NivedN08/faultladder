import json

from faultladder.results import reacher_gate, update_results

PASSING = {
    "oracle": {"success_rate": 0.95},
    "ppo": {"success_rate": 0.90},
    "random": {"success_rate": 0.05},
    "zero": {"success_rate": 0.0},
    "ppo_target_hidden": {"success_rate": 0.10},
}


def test_gate_passes_at_the_thresholds():
    assert reacher_gate(PASSING)["passed"]


def test_gate_fails_when_any_condition_fails():
    for name, rate in [("oracle", 0.94), ("ppo", 0.89), ("random", 0.06)]:
        gate = reacher_gate({**PASSING, name: {"success_rate": rate}})
        assert not gate["passed"]
        assert not gate["checks"][name]["passed"]


def test_gate_fails_when_a_number_is_missing():
    partial = {k: v for k, v in PASSING.items() if k != "ppo"}
    assert not reacher_gate(partial)["passed"]


def test_update_results_merges_and_recomputes_the_gate(tmp_path):
    path = tmp_path / "out" / "reacher.json"
    for name, value in PASSING.items():
        update_results(path, name, value)
    results = json.loads(path.read_text())
    assert results["gate"]["passed"]
    assert {k: results[k] for k in PASSING} == PASSING
