"""The Modal app's pins and cost settings, checked without starting a job."""

import json
import tomllib
from pathlib import Path

import modal_app
import pytest

ROOT = Path(__file__).parent.parent


def test_image_versions_match_the_lockfile() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    locked: dict[str, set[str]] = {}
    for package in lock["package"]:
        version = package["version"].split("+")[0]
        locked.setdefault(package["name"], set()).add(version)
    for requirement in modal_app.IMAGE_PACKAGES:
        name, version = requirement.split("==")
        assert locked[name] == {version}, requirement


def test_image_has_one_mujoco() -> None:
    names = [requirement.split("==")[0] for requirement in modal_app.IMAGE_PACKAGES]
    assert "mujoco" in names
    assert "mujoco-core" not in names
    assert "mjviser==0.0.16" in modal_app.IMAGE_PACKAGES


def test_price_includes_gpu_cpu_and_memory() -> None:
    price = modal_app.price_per_second("L40S")
    assert price == pytest.approx(0.000542 + 4 * 0.0000131 + 16 * 0.00000222)
    assert modal_app.TRAIN_GPUS[0] == modal_app.TRAIN_GPU
    assert set(modal_app.TRAIN_GPUS) <= set(modal_app.GPU_PRICE)
    assert max(modal_app.GPU_PRICE.values()) == modal_app.GPU_PRICE["L40S"]


def test_iteration_budget_matches_the_recorded_curve() -> None:
    curve = json.loads((ROOT / "results/go1_baseline_curve.json").read_text())
    assert curve["iteration_budget"] == modal_app.ITERATION_BUDGET
