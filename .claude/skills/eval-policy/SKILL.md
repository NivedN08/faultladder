---
name: eval-policy
description: Score a policy on the FaultLadder (36 cells), check the validity gates and read the result. Use when asked to evaluate, score or compare a checkpoint, the zero agent or the random agent, or after changing the ladder, the eval task or the score.
---

# Evaluate a policy

The score has one definition, `faultladder.score`. Never recompute it elsewhere.

1. Before spending: `./scripts/check.sh` passes. Measured on 2026-11-22, one
   full ladder (36 cells of 4,096 episodes) on an L4 takes 9 to 18 minutes and
   costs $0.17 to $0.33; runs with many falls are the slow ones.

2. Full ladder on Modal. Writes `results/<name>/ladder.json` locally and
   `eval/<name>/ladder.json` on the `faultladder-runs` volume:

   ```sh
   uv run modal run modal_app.py::ladder --name ppo_default_seed0
   uv run modal run modal_app.py::ladder --name zero --agent zero
   uv run modal run modal_app.py::ladder --name random --agent random
   ```

   Options: `--run <task>/seed<n>` and `--checkpoint model_<n>.pt` pick the
   policy (default: newest checkpoint of the Go1 seed 0 run), `--seed` changes
   the commands and randomness, `--only nominal,push/3` runs a subset (then
   `scores` is null), `--out` changes the local directory.

3. Small checks on the Mac CPU (no cost). 256 robots for one cell take about
   16 minutes, so keep it to a few cells:

   ```sh
   MUJOCO_GL=disable uv run python -m faultladder.evaluate --agent ppo \
     --checkpoint runs/go1_seed0/model_2999.pt --out scratch/cpu \
     --num-envs 64 --only nominal
   ```

4. Gates G1, G2, G4 and G6. Needs the default policy scored twice with
   different seeds, plus the zero and random agents. Exits non-zero on failure
   and writes `results/gate.md`:

   ```sh
   uv run python -m faultladder.gate
   ```

5. Read `ladder.json`: `scores.faultladder` is the headline, `scores.axes` the
   per-axis means over levels 1 to 5, `cells[*].score` the success rate and
   `cells[*].fall_rate` how many episodes ended in a fall. A low score with a
   low fall rate means the robot stayed up but missed the velocity tolerance.

6. Add each run to `results/spend.csv` (`cost_usd` in the JSON is list price
   times in-job time).

## Rules

- Zero and random agents are rerun whenever the ladder, the eval task or the
  score changes. Both must stay at or below 0.05.
- Changing a level in `ladder_v0.json` is a calibration round; the plan allows
  three. Record each in `results/STATUS.md` and rerun all four evaluations.
- Report only numbers from a `ladder.json` you produced.
