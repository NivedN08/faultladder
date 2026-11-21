"""Guards against two distributions installing into one package directory.

mujoco 3.x and mujoco-core 4.x both install into ``mujoco/``. When a transitive
dependency pulled in both, imports broke on Linux and silently mixed versions
on macOS. These tests fail as soon as the installed environment is in that
state, on any platform.
"""

import importlib
from collections import defaultdict
from importlib import metadata

import pytest

# Top-level package directories this project cannot work without.
GUARDED = ("mujoco", "mujoco_warp", "warp", "mjlab", "torch", "rsl_rl")

# Distributions that provide the same import package and must not coexist.
EXCLUSIVE = (("mujoco", "mujoco-core"),)


def _normalise(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def find_shared_files(
    files_by_dist: dict[str, list[tuple[str, ...]]],
    guarded: tuple[str, ...] = GUARDED,
) -> dict[str, list[str]]:
    """Return guarded files that more than one distribution installs."""
    owners: dict[str, set[str]] = defaultdict(set)
    for dist, files in files_by_dist.items():
        for parts in files:
            if parts and parts[0] in guarded and "__pycache__" not in parts:
                owners["/".join(parts)].add(dist)
    return {f: sorted(o) for f, o in owners.items() if len(o) > 1}


def installed_files() -> dict[str, list[tuple[str, ...]]]:
    return {
        _normalise(dist.metadata["Name"]): [f.parts for f in dist.files or []]
        for dist in metadata.distributions()
    }


def test_find_shared_files_reports_a_collision():
    files = {
        "mujoco": [("mujoco", "__init__.py"), ("mujoco", "renderer.py")],
        "mujoco-core": [("mujoco", "__init__.py"), ("mujoco", "rendering", "x.py")],
        "other": [("other", "__init__.py"), ("mujoco", "__pycache__", "a.pyc")],
    }
    assert find_shared_files(files) == {"mujoco/__init__.py": ["mujoco", "mujoco-core"]}


def test_find_shared_files_ignores_unguarded_packages():
    files = {"a": [("docs", "index.md")], "b": [("docs", "index.md")]}
    assert find_shared_files(files) == {}


def test_no_two_distributions_share_a_guarded_file():
    shared = find_shared_files(installed_files())
    sample = dict(list(shared.items())[:5])
    assert not shared, (
        f"{len(shared)} files are installed by more than one distribution, "
        f"e.g. {sample}. Bound the dependency that pulled in the second one."
    )


@pytest.mark.parametrize("group", EXCLUSIVE)
def test_exclusive_distributions_do_not_coexist(group):
    present = sorted(set(group) & set(installed_files()))
    assert len(present) <= 1, f"installed together: {present}"


@pytest.mark.parametrize(
    ("module", "dist"),
    [("mujoco", "mujoco"), ("mujoco_warp", "mujoco-warp"), ("warp", "warp-lang")],
)
def test_imported_version_matches_installed_distribution(module, dist):
    assert importlib.import_module(module).__version__ == metadata.version(dist)
