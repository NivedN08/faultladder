---
name: modal-train
description: Train, resume, monitor, fetch and render a FaultLadder policy on Modal. Use before launching any Modal GPU job, when a training run was interrupted, or when you need its logs, checkpoints, cost or a video.
---

# Train on Modal

Every `modal run` below spends credits. Measured on 2026-11-21: one Go1 run of
1,320 iterations on an L4 is about 24 minutes and $0.45; a 20 s video is about
$0.03. Rates are in `results/gpu_benchmark.json`.

1. Before launching: `./scripts/check.sh` passes, and you have stated the
   expected cost and checked it against the phase cap in `CLAUDE.md`.

2. Launch detached, so the job survives the terminal closing:

   ```sh
   uv run modal run --detach modal_app.py::train --seed 0
   ```

   Options: `--task`, `--iterations` (default `ITERATION_BUDGET`),
   `--num-envs`. Output goes to `<task>/seed<seed>/` on the `faultladder-runs`
   volume. The command keeps streaming logs; stopping it does not stop the job.

3. Check on it without holding a terminal open. Do not poll in a loop.

   ```sh
   uv run modal app list
   uv run modal app logs <app id> --since 2m
   uv run modal volume ls faultladder-runs <task>/seed<seed>
   ```

   The app state is `ephemeral (detached)` while running and `stopped` when
   done. Cancel with `uv run modal app stop <app id>` if reward has not
   improved after 20% of the iterations.

4. Resume or extend: run the same command again, with a larger `--iterations`
   to extend. Training restarts from the newest readable `model_<n>.pt`; a
   preempted job does this by itself through the retry policy.

5. Fetch the run. The download lands in a directory named after the last part
   of the remote path, inside the destination:

   ```sh
   cd runs && uv run modal volume get faultladder-runs <task>/seed<seed> .
   ```

   `train.json` in it has one entry per launch with seconds and cost.

6. Read the reward curve and the iteration budget:

   ```sh
   uv run python -m faultladder.curve runs/seed<seed> --out results/<name>.json
   ```

7. Render a 20 s video of the newest checkpoint and fetch it. The owner watches
   it; do not judge walking from numbers alone.

   ```sh
   uv run modal run modal_app.py::render --run <task>/seed<seed>
   uv run modal volume get faultladder-runs <path it prints> runs/<name>.mp4
   ```

8. Append each job's seconds and cost to `results/spend.csv`.

## Known failures

- `Please add a payment method to use <GPU> GPU functions`: the workspace has
  no card. Modal checks every GPU function in the app, not only the one
  called. Only the owner can fix it, under Settings, Usage and Billing; keep
  the spend limit at $0 so only credits are used.
- Jobs stop early with a budget message: the $130 usage limit counts all
  workspace usage in the billing period, not only this project.
- Image package versions must match `uv.lock`; `tests/test_modal_app.py`
  checks this. Never let `mujoco-core` into the image.
