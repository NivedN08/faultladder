"""The frozen FaultLadder ladder: seven axes, five levels each, plus nominal."""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from mjlab.envs import ManagerBasedRlEnv

from faultladder.score import AXES, NUM_LEVELS

LADDER_FILE = Path(__file__).with_name("ladder_v1.json")
NOMINAL = "nominal"

# Event term names in the FaultLadder tasks (see ``faultladder.tasks``).
MOTOR, PAYLOAD, FRICTION, LATENCY, PUSH = (
    "fl_motor",
    "fl_payload",
    "fl_friction",
    "fl_latency",
    "fl_push",
)


@dataclass(frozen=True)
class Cell:
    """One axis at one severity level; level 0 is the unperturbed robot."""

    axis: str
    level: int
    value: float

    @property
    def name(self) -> str:
        return NOMINAL if self.level == 0 else f"{self.axis}/{self.level}"


@cache
def load(path: Path = LADDER_FILE) -> dict[str, Any]:
    spec = json.loads(path.read_text())
    if tuple(spec["axes"]) != AXES:
        raise ValueError(f"ladder axes {tuple(spec['axes'])} do not match {AXES}")
    for axis, entry in spec["axes"].items():
        if len(entry["levels"]) != NUM_LEVELS:
            raise ValueError(f"{axis} needs {NUM_LEVELS} levels")
    return spec


def nominal_cell() -> Cell:
    return Cell(NOMINAL, 0, 0.0)


def cells() -> list[Cell]:
    """The nominal cell followed by the 35 perturbed cells."""
    out = [nominal_cell()]
    for axis, entry in load()["axes"].items():
        out += [Cell(axis, i + 1, value) for i, value in enumerate(entry["levels"])]
    return out


def event_params(*cells: Cell) -> dict[str, dict[str, Any]]:
    """Parameters of every FaultLadder event term with ``cells`` applied together.

    Cells must be on different axes, and at most one may be a motor axis: the
    motor event weakens one kind of joint.
    """
    spec = load()
    params: dict[str, dict[str, Any]] = {
        MOTOR: {"joint_type": None, "strength": 1.0},
        PAYLOAD: {"mass_fraction": 0.0, "max_offset": spec["payload_max_offset"]},
        FRICTION: {"coefficient": None},
        LATENCY: {"steps": 0},
        PUSH: {"speed": 0.0},
    }
    used: set[str] = set()
    for cell in cells:
        if cell.level == 0:
            continue
        if cell.axis.startswith("motor_"):
            event = MOTOR
            update = {
                "joint_type": cell.axis.removeprefix("motor_"),
                "strength": cell.value,
            }
        elif cell.axis == "payload":
            event, update = PAYLOAD, {"mass_fraction": cell.value}
        elif cell.axis == "friction":
            event, update = FRICTION, {"coefficient": cell.value}
        elif cell.axis == "latency":
            event, update = LATENCY, {"steps": cell.value}
        elif cell.axis == "push":
            event, update = PUSH, {"speed": cell.value}
        else:
            raise ValueError(f"unknown axis {cell.axis!r}")
        if event in used:
            raise ValueError(f"two cells change {event}: {[c.name for c in cells]}")
        used.add(event)
        params[event].update(update)
    return params


def cell(axis: str, level: int) -> Cell:
    """The ladder cell of ``axis`` at ``level`` (1 to 5)."""
    return Cell(axis, level, load()["axes"][axis]["levels"][level - 1])


def set_cell(env: ManagerBasedRlEnv, *cells: Cell) -> None:
    """Select ``cells`` for a FaultLadder environment; they apply on the next reset."""
    for name, params in event_params(*cells).items():
        env.event_manager.get_term_cfg(name).params.update(params)
