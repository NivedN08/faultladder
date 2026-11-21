#!/usr/bin/env bash
# The one definition of "the checks pass". CI runs this file and nothing else,
# so a green run here means a green run there. Run it before every push.
set -euo pipefail
cd "$(dirname "$0")/.."

# mjlab defaults MUJOCO_GL to egl on Linux and mujoco loads libEGL at import.
# These checks never render, so no GL backend is needed anywhere.
export MUJOCO_GL="${MUJOCO_GL:-disable}"

step() { printf '\n== %s\n' "$*"; "$@"; }

# --locked fails if uv.lock is out of date with pyproject.toml.
step uv sync --locked --all-extras
step uv run --locked pytest tests/test_install_integrity.py -q
step uv run --locked ruff format --check
step uv run --locked ruff check
step uv run --locked pyright
step uv run --locked pytest -m "not gpu" -q

printf '\nAll checks passed.\n'
