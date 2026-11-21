---
name: env-check
description: Validate a new or changed FaultLadder environment before any training. Use after editing an environment, its model file, reward, reset or observation, and before launching a training run.
---

# Environment check

Run these in order and stop at the first failure. Nothing trains until all pass.

1. Tests, including determinism and the Gymnasium checker:

   ```sh
   uv run pytest tests/test_reacher.py tests/test_results.py
   ```

2. Zero agent, random agent, oracle, determinism and speed, 1,000 episodes each:

   ```sh
   uv run python -m faultladder.checks --out results/reacher.json
   ```

   Expect: zero and random success at most 0.05, oracle at least 0.95,
   `max_abs_difference` exactly 0.0. A zero or random agent that scores means
   the success rule can be met by doing nothing; fix the rule, not the number.
   An oracle that fails means the task or the physics is wrong; do not train.

3. For the mjlab Go1 task, the same two dummy agents:

   ```sh
   uv run python -m faultladder.smoke --agent zero --num-envs 16
   uv run python -m faultladder.smoke --agent random --num-envs 16
   ```

4. Only then train, and train the ablation with the task information hidden:

   ```sh
   uv run python -m faultladder.reacher.train
   uv run python -m faultladder.reacher.train --hide-target
   ```

   Each takes about 3 minutes on an M2. Run them one after the other when both
   write to the same results file.

5. Read the gate from the results file; never type a result by hand:

   ```sh
   uv run python -c "import json; print(json.load(open('results/reacher.json'))['gate'])"
   ```

Before the full check script, `./scripts/check.sh` must pass.
