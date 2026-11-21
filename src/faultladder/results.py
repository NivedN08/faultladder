"""Reading and writing ``results/reacher.json`` and its Phase 1 gate."""

import json
from pathlib import Path
from typing import Any

# name -> (key in the results file, comparison, threshold on success rate)
REACHER_GATE = {
    "oracle": (">=", 0.95),
    "ppo": (">=", 0.90),
    "random": ("<=", 0.05),
    "zero": ("<=", 0.05),
    "ppo_target_hidden": ("<=", 0.10),
}


def reacher_gate(results: dict[str, Any]) -> dict[str, Any]:
    """Check every condition of the Phase 1 gate; a missing number fails it."""
    checks = {}
    for name, (op, threshold) in REACHER_GATE.items():
        rate = results.get(name, {}).get("success_rate")
        if rate is None:
            passed = False
        else:
            passed = rate >= threshold if op == ">=" else rate <= threshold
        checks[name] = {
            "success_rate": rate,
            "condition": f"{op} {threshold}",
            "passed": passed,
        }
    return {"checks": checks, "passed": all(c["passed"] for c in checks.values())}


def update_results(path: str | Path, key: str, value: dict[str, Any]) -> dict[str, Any]:
    """Set one entry in the results file and recompute the gate."""
    path = Path(path)
    results = json.loads(path.read_text()) if path.exists() else {}
    results[key] = value
    results["gate"] = reacher_gate(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    return results
