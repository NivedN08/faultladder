"""The frozen FaultLadder ladder: seven axes, five levels each, plus nominal."""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from mjlab.envs import ManagerBasedRlEnv

from faultladder.score import AXES, NUM_LEVELS

LADDER_FILE = Path(__file__).with_name("ladder_v0.json")
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


def event_params(cell: Cell) -> dict[str, dict[str, Any]]:
    """Parameters of every FaultLadder event term for ``cell``."""
    spec = load()
    params: dict[str, dict[str, Any]] = {
        MOTOR: {"joint_type": None, "strength": 1.0},
        PAYLOAD: {"mass_fraction": 0.0, "max_offset": spec["payload_max_offset"]},
        FRICTION: {"coefficient": None},
        LATENCY: {"steps": 0},
        PUSH: {"speed": 0.0},
    }
    if cell.level == 0:
        return params
    if cell.axis.startswith("motor_"):
        params[MOTOR] = {
            "joint_type": cell.axis.removeprefix("motor_"),
            "strength": cell.value,
        }
    elif cell.axis == "payload":
        params[PAYLOAD]["mass_fraction"] = cell.value
    elif cell.axis == "friction":
        params[FRICTION]["coefficient"] = cell.value
    elif cell.axis == "latency":
        params[LATENCY]["steps"] = int(cell.value)
    elif cell.axis == "push":
        params[PUSH]["speed"] = cell.value
    else:
        raise ValueError(f"unknown axis {cell.axis!r}")
    return params


def set_cell(env: ManagerBasedRlEnv, cell: Cell) -> None:
    """Select ``cell`` for a FaultLadder environment; it applies on the next reset."""
    for name, params in event_params(cell).items():
        env.event_manager.get_term_cfg(name).params.update(params)
