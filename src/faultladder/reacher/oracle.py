"""Closed-form inverse kinematics for the two-link reacher.

Pure geometry: nothing here reads the simulator, so the tests can check it
against the MuJoCo model instead of against itself.
"""

import math

import numpy as np

LINK_LENGTHS = (0.1, 0.1)


def forward_kinematics(q1: float, q2: float) -> tuple[float, float]:
    """Fingertip position for shoulder angle ``q1`` and elbow angle ``q2``."""
    l1, l2 = LINK_LENGTHS
    return (
        l1 * math.cos(q1) + l2 * math.cos(q1 + q2),
        l1 * math.sin(q1) + l2 * math.sin(q1 + q2),
    )


def inverse_kinematics(x: float, y: float) -> list[tuple[float, float]]:
    """Both joint solutions (elbow bent each way) that put the fingertip at x, y."""
    l1, l2 = LINK_LENGTHS
    cos_q2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    if abs(cos_q2) > 1:
        raise ValueError(f"target ({x}, {y}) is out of reach")
    solutions = []
    for q2 in (math.acos(cos_q2), -math.acos(cos_q2)):
        q1 = math.atan2(y, x) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
        solutions.append((q1, q2))
    return solutions


def wrap(angle: float) -> float:
    """Map an angle to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def joint_targets(q: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Joint angles that reach ``target``, choosing the solution nearest to ``q``.

    The shoulder is unlimited, so its target is expressed as the shortest turn
    from the current angle.
    """
    goals = [
        np.array([q[0] + wrap(q1 - q[0]), q2])
        for q1, q2 in inverse_kinematics(float(target[0]), float(target[1]))
    ]
    return min(goals, key=lambda goal: float(np.abs(goal - q).sum()))
