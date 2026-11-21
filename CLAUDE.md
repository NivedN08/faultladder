# FaultLadder: rules for agents

A held-out robustness benchmark for quadruped walking policies, built on mjlab
and MuJoCo Warp. Start every session by reading this file and
`results/STATUS.md`; end it by updating `STATUS.md`.

## Commands

Always use `uv run`, never bare `python` (and never the Anaconda Python).

```sh
uv sync --all-extras              # install; the lockfile is CPU-only
uv run ruff format && uv run ruff check --fix
uv run pyright
uv run pytest -m "not gpu"        # what CI runs
uv run pytest -m "not gpu and not slow"   # fast loop
uv run python -m faultladder.smoke --agent zero --num-envs 16   # mjlab smoke
uv run list-envs                  # mjlab's registered tasks
```

## Pinned versions

Python 3.12, mjlab 1.6.0, mujoco 3.11.0, mujoco-warp 3.11.x, warp-lang 1.17.0,
torch 2.14.0, rsl-rl-lib 5.4.2. Optional extra `reacher`: gymnasium 1.4.0,
stable-baselines3 2.9.2. Do not upgrade before the Phase 2 baseline is
reproduced.

Differences from the original plan:

- On Linux the lockfile takes torch from the PyTorch CPU index so CI stays
  small. The Modal image must install CUDA torch itself.
- The mjlab smoke rollout is headless (`faultladder.smoke`); mjlab's own
  `uv run play <task> --agent zero` opens a viewer and never exits.
- ffmpeg is not installed with Homebrew: mjlab depends on `imageio-ffmpeg`,
  which bundles a binary.

## Facts checked against the installed libraries

- Base task id: `Mjlab-Velocity-Flat-Unitree-Go1` (also a `Rough` variant).
- Build an env with `load_env_cfg(task_id)`, set `cfg.scene.num_envs`, then
  `RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cpu"))`.
- First CPU run compiles Warp kernels (about 30 s on an M2); later runs use
  the cache.
- MuJoCo Warp is not deterministic even with a fixed seed. Determinism tests
  run against CPU MuJoCo.
- In mjlab, XML `<option>` flags are ignored, and contact sensors miss brief
  contacts when decimation is above 1.

## Correctness rules

1. Write the test before the code for the score, the ladder and every env.
2. Report only numbers that were actually run, with the command that made them.
3. Every table in the repository is generated from JSON in `results/` by a
   script. Never type a result by hand.
4. The score has exactly one definition: `faultladder.score`.
5. Run the zero and random agents on every new or changed environment before
   training anything.
6. Do not edit mjlab's source. Extend it from this package.
7. mjlab, MuJoCo Warp and Modal changed heavily in 2026. Read the installed
   version's source or docs before writing against them. Context7 ids:
   `/mujocolab/mjlab`, `/google-deepmind/mujoco`, `/google-deepmind/mujoco_warp`,
   `/websites/modal`, `/farama-foundation/gymnasium`, `/nvidia/warp`,
   `/leggedrobotics/rsl_rl`.
8. When a gate fails, change ladder levels or thresholds, never baseline
   scores. After three failed calibration rounds, stop, write the numbers to
   `results/gate.md` and ask the user.

## Spending rules (Modal)

Project cap $130. Phase caps: 2 = $10, 3 = $25, 4 = $50, 5 = $40, 6 = $5.
A phase never borrows from another.

1. Nothing runs on Modal until its tests and a 16-env CPU smoke rollout pass.
2. One GPU per job, nothing above an L40S, every function sets `timeout`
   (training 7200 s, evaluation 1800 s, benchmark 600 s), training uses
   `modal run --detach` and `modal.Retries(max_retries=3, initial_delay=0.0)`.
3. Print the expected cost before each launch; append the actual duration and
   cost to `results/spend.csv` after.
4. Cancel a training run with no reward improvement after 20% of its
   iterations.
5. Do not poll a running job in a loop.
6. Volume `faultladder-runs` at `/runs` holds checkpoints (every 100
   iterations), TensorBoard logs, videos and result JSON. Set `MUJOCO_GL=egl`
   and `WANDB_MODE=disabled`.

## Repository hygiene

- Small commits, short imperative messages. From Phase 2 on: one branch and
  one pull request per phase, merged when CI is green.
- Never commit secrets, checkpoints, TensorBoard logs or videos.
- Planning documents and `CLAUDE.local.md` stay out of the repository.
- Robot models: use the copies mjlab bundles; check the licence before
  redistributing any model file.
- One phase per session. Phase gates and the current state are in
  `results/STATUS.md`.
