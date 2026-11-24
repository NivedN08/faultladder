"""The single definition of the FaultLadder score and its validity gates.

Every test, script and table calls this module; nothing else may recompute a
success, a cell score or the headline number.
"""

from typing import Any

import torch

AXES = (
    "motor_hip",
    "motor_thigh",
    "motor_calf",
    "payload",
    "friction",
    "latency",
    "push",
)
FAULT_AXES = ("motor_hip", "motor_thigh", "motor_calf", "push")
SHIFT_AXES = ("payload", "friction", "latency")
NUM_LEVELS = 5

LIN_TOL = 0.25  # m/s, mean planar velocity error over the scored window
ANG_TOL = 0.5  # rad/s, mean yaw-rate error over the scored window

# Multiples of both tolerances at which every cell is also scored, to check
# that the ranking of policies does not hinge on the tolerances chosen.
TOLERANCE_SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)

G1_NOMINAL = 0.95
G2_FLOOR = 0.05
G4_LEVEL1 = 0.8
G4_LEVEL5 = 0.5
G6_REPEAT = 0.02
G3_GAP = 0.15  # wide minus no randomisation
G5_LEVEL3 = 0.5  # privileged success at level 3
_EPS = 1e-9  # tolerance for comparisons that sit exactly on a threshold


def episode_success(
    fell: torch.Tensor,
    lin_err: torch.Tensor,
    ang_err: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """An episode succeeds if the robot never fell and tracked its command.

    ``lin_err`` and ``ang_err`` are each episode's mean error over the scored
    window (the last 8 s of a 10 s episode). ``scale`` multiplies both
    tolerances; the FaultLadder score uses 1.
    """
    return ~fell & (lin_err <= scale * LIN_TOL) & (ang_err <= scale * ANG_TOL)


def cell_score(success: torch.Tensor) -> float:
    """Fraction of successful episodes in one (axis, level) cell."""
    return float(success.float().mean())


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def ladder_scores(cells: dict[str, list[float]], nominal: float) -> dict[str, Any]:
    """Aggregate cell scores (levels 1 to 5 per axis) into the reported numbers."""
    if set(cells) != set(AXES):
        raise ValueError(f"expected axes {AXES}, got {sorted(cells)}")
    for axis, levels in cells.items():
        if len(levels) != NUM_LEVELS:
            raise ValueError(f"{axis}: expected {NUM_LEVELS} levels, got {len(levels)}")
    axes = {axis: _mean(cells[axis]) for axis in AXES}
    return {
        "faultladder": _mean([axes[a] for a in AXES]),
        "fault": _mean([axes[a] for a in FAULT_AXES]),
        "shift": _mean([axes[a] for a in SHIFT_AXES]),
        "nominal": nominal,
        "axes": axes,
    }


def gates(
    ppo_default: dict[str, Any],
    ppo_default_cells: dict[str, list[float]],
    zero: float,
    random: float,
    rescored: float,
) -> dict[str, Any]:
    """Gates G1, G2, G4 and G6 (the ones that need only the default policy).

    ``zero`` and ``random`` are those agents' FaultLadder scores; ``rescored``
    is the default policy's FaultLadder score from a second evaluation.
    """
    failing = [
        axis
        for axis in AXES
        if ppo_default_cells[axis][0] < G4_LEVEL1 - _EPS
        or ppo_default_cells[axis][-1] > G4_LEVEL5 + _EPS
    ]
    drift = abs(rescored - ppo_default["faultladder"])
    checks: dict[str, dict[str, Any]] = {
        "G1": {
            "condition": f"nominal >= {G1_NOMINAL}",
            "value": ppo_default["nominal"],
            "passed": ppo_default["nominal"] >= G1_NOMINAL - _EPS,
        },
        "G2": {
            "condition": f"zero and random <= {G2_FLOOR}",
            "value": {"zero": zero, "random": random},
            "passed": max(zero, random) <= G2_FLOOR + _EPS,
        },
        "G4": {
            "condition": f"level 1 >= {G4_LEVEL1} and level 5 <= {G4_LEVEL5} per axis",
            "failing_axes": failing,
            "passed": not failing,
        },
        "G6": {
            "condition": f"rescoring differs by <= {G6_REPEAT}",
            "value": drift,
            "passed": drift <= G6_REPEAT + _EPS,
        },
    }
    return {"checks": checks, "passed": all(c["passed"] for c in checks.values())}


def spread(values: list[float]) -> float:
    """Sample standard deviation across seeds; 0 for a single seed."""
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def baseline_gates(
    nodr: list[float], wide: list[float], privileged_level3: dict[str, list[float]]
) -> dict[str, Any]:
    """Gates G3 and G5, which need the trained baselines.

    ``nodr`` and ``wide`` are FaultLadder scores, one per seed. The seed spread
    is the larger of their two standard deviations, so G3 needs at least two
    seeds of each. ``privileged_level3`` maps
    each axis to the privileged policy's level-3 success, one per seed.
    """
    gap = _mean(wide) - _mean(nodr)
    seed_spread = max(spread(nodr), spread(wide))
    level3 = {axis: _mean(privileged_level3[axis]) for axis in AXES}
    failing = [axis for axis in AXES if level3[axis] < G5_LEVEL3 - _EPS]
    checks: dict[str, dict[str, Any]] = {
        "G3": {
            "condition": f"wide - no randomisation >= {G3_GAP} and > 2 x seed spread",
            "value": {"gap": gap, "seed_spread": seed_spread},
            "passed": min(len(nodr), len(wide)) >= 2
            and gap >= G3_GAP - _EPS
            and gap > 2 * seed_spread,
        },
        "G5": {
            "condition": f"privileged level 3 >= {G5_LEVEL3} per axis",
            "value": {"lowest": min(level3.values())},
            "failing_axes": failing,
            "passed": not failing,
        },
    }
    return {"checks": checks, "passed": all(c["passed"] for c in checks.values())}


MOTOR_AXES = ("motor_hip", "motor_thigh", "motor_calf")


def onset_scores(
    static: dict[str, list[float]], onset: dict[str, list[float]]
) -> dict[str, Any]:
    """Onset track: motor faults that start mid-episode.

    ``static`` and ``onset`` hold the five cell scores of each motor axis with
    the fault present from the start and appearing mid-episode, scored over
    the same window. ``onset`` is the track score; ``gap`` is how much a fault
    costs by arriving while the robot walks.
    """
    for scores in (static, onset):
        if set(scores) != set(MOTOR_AXES):
            raise ValueError(f"expected axes {MOTOR_AXES}, got {tuple(scores)}")
        if any(len(levels) != NUM_LEVELS for levels in scores.values()):
            raise ValueError(f"every axis needs {NUM_LEVELS} levels")
    axes = {
        axis: {"static": _mean(static[axis]), "onset": _mean(onset[axis])}
        for axis in MOTOR_AXES
    }
    static_mean = _mean([a["static"] for a in axes.values()])
    onset_mean = _mean([a["onset"] for a in axes.values()])
    return {
        "onset": onset_mean,
        "static": static_mean,
        "gap": static_mean - onset_mean,
        "axes": axes,
    }


def compound_prediction(single_a: float, single_b: float, nominal: float) -> float:
    """Success expected under two faults if they hurt independently.

    Each fault keeps a fraction ``single / nominal`` of the nominal successes;
    independent faults keep the product of the two fractions.
    """
    if nominal <= 0.0:
        return 0.0
    return min(single_a, single_b, single_a * single_b / nominal)


def compound_scores(pairs: list[dict[str, float]], nominal: float) -> dict[str, float]:
    """Compound track: two axes perturbed together.

    Each pair has the ``observed`` success and the two single-fault successes
    ``single_a`` and ``single_b``. ``compound`` is the track score;
    ``interaction`` is observed minus predicted, negative when two faults
    together are worse than their separate effects predict.
    """
    if not pairs:
        raise ValueError("no compound cells")
    observed = _mean([p["observed"] for p in pairs])
    predicted = _mean(
        [compound_prediction(p["single_a"], p["single_b"], nominal) for p in pairs]
    )
    return {
        "compound": observed,
        "predicted": predicted,
        "interaction": observed - predicted,
    }


def interval(values: list[float]) -> float:
    """Half-width of the 95% confidence interval of the mean across seeds."""
    # Two-sided Student t quantiles at 0.975 for 1 to 9 degrees of freedom.
    t = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262)
    n = len(values)
    if n < 2:
        return 0.0
    quantile = t[n - 2] if n - 2 < len(t) else 1.96
    return quantile * spread(values) / n**0.5


def rank_agreement(a: list[float], b: list[float]) -> float:
    """Kendall's tau between two score lists over the same policies."""
    pairs = [(i, j) for i in range(len(a)) for j in range(i + 1, len(a))]
    if not pairs:
        raise ValueError("ranking needs at least two policies")
    sign = lambda x: (x > 0) - (x < 0)  # noqa: E731
    return sum(sign(a[i] - a[j]) * sign(b[i] - b[j]) for i, j in pairs) / len(pairs)
