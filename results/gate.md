# FaultLadder validity gates

Unitree Go1, ladder v1, 4096 episodes per cell. G1, G4 and G6 use the default policy of training seed 0; G3 and G5 use training seeds 0, 1, 2, 3, 4. The seed spread in G3 is the larger of the two baselines' standard deviations across seeds; G5 uses the mean across seeds.

Every row is recomputed from the saved runs by `uv run python -m faultladder.gate`. The last column is the command that produced those runs.

| Gate | Condition | Measured | Result | Runs produced by |
| --- | --- | --- | --- | --- |
| G1 | nominal >= 0.95 | 1.0000 | pass | `uv run modal run modal_app.py::baselines` |
| G2 | zero and random <= 0.05 | zero 0.0000; random 0.0000 | pass | `uv run modal run modal_app.py::ladder --agent zero (and --agent random)` |
| G3 | wide - no randomisation >= 0.15 and > 2 x seed spread | gap 0.1796; seed_spread 0.0141 | pass | `uv run modal run modal_app.py::baselines` |
| G4 | level 1 >= 0.8 and level 5 <= 0.5 per axis | all axes pass | pass | `uv run modal run modal_app.py::baselines` |
| G5 | privileged level 3 >= 0.5 per axis | lowest 0.9957 | pass | `uv run modal run modal_app.py::baselines` |
| G6 | rescoring differs by <= 0.02 | 0.0003 | pass | `uv run modal run modal_app.py::ladder --name ppo_default_seed0_rescore --baseline default --seed 0 --eval-seed 1` |

All six gates pass.

## Default policy, success rate per cell

| Axis | L1 | L2 | L3 | L4 | L5 | Axis score |
| --- | --- | --- | --- | --- | --- | --- |
| motor_hip | 1.000 | 1.000 | 0.990 | 0.888 | 0.152 | 0.806 |
| motor_thigh | 1.000 | 1.000 | 1.000 | 1.000 | 0.265 | 0.853 |
| motor_calf | 1.000 | 1.000 | 0.596 | 0.164 | 0.000 | 0.552 |
| payload | 1.000 | 0.999 | 0.919 | 0.619 | 0.226 | 0.753 |
| friction | 1.000 | 1.000 | 1.000 | 0.988 | 0.296 | 0.857 |
| latency | 1.000 | 0.774 | 0.009 | 0.000 | 0.000 | 0.357 |
| push | 1.000 | 1.000 | 0.974 | 0.530 | 0.001 | 0.701 |

## Privileged policy at level 3, per seed

| Axis | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
| --- | --- | --- | --- | --- | --- |
| motor_hip | 1.000 | 0.999 | 0.996 | 0.999 | 0.999 |
| motor_thigh | 0.997 | 0.999 | 1.000 | 1.000 | 0.999 |
| motor_calf | 1.000 | 0.999 | 0.999 | 0.995 | 0.999 |
| payload | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 |
| friction | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| latency | 1.000 | 1.000 | 0.993 | 1.000 | 1.000 |
| push | 0.994 | 0.995 | 0.998 | 0.996 | 0.996 |
