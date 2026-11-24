# FaultLadder validity gates

Ladder v1, 4096 episodes per cell. G1, G4 and G6 use the default policy of training seed 0; G3 and G5 use training seeds 0, 1, 2. The seed spread in G3 is the larger of the two baselines' standard deviations across seeds; G5 uses the mean across seeds.

Every row is recomputed from the saved runs by `uv run python -m faultladder.gate`. The last column is the command that produced those runs.

| Gate | Condition | Measured | Result | Runs produced by |
| --- | --- | --- | --- | --- |
| G1 | nominal >= 0.95 | 1.0000 | pass | `uv run modal run modal_app.py::baselines` |
| G2 | zero and random <= 0.05 | zero 0.0000; random 0.0000 | pass | `uv run modal run modal_app.py::ladder --agent zero (and --agent random)` |
| G3 | wide - no randomisation >= 0.15 and > 2 x seed spread | gap 0.1810; seed_spread 0.0104 | pass | `uv run modal run modal_app.py::baselines` |
| G4 | level 1 >= 0.8 and level 5 <= 0.5 per axis | all axes pass | pass | `uv run modal run modal_app.py::baselines` |
| G5 | privileged level 3 >= 0.5 per axis | lowest 0.9956 | pass | `uv run modal run modal_app.py::baselines` |
| G6 | rescoring differs by <= 0.02 | 0.0005 | pass | `uv run modal run modal_app.py::ladder --name ppo_default_seed0_rescore --baseline default --seed 0 --eval-seed 1` |

All six gates pass.

## Default policy, success rate per cell

| Axis | L1 | L2 | L3 | L4 | L5 | Axis score |
| --- | --- | --- | --- | --- | --- | --- |
| motor_hip | 1.000 | 1.000 | 0.990 | 0.889 | 0.151 | 0.806 |
| motor_thigh | 1.000 | 1.000 | 1.000 | 1.000 | 0.266 | 0.853 |
| motor_calf | 1.000 | 1.000 | 0.594 | 0.167 | 0.001 | 0.552 |
| payload | 1.000 | 0.999 | 0.919 | 0.621 | 0.224 | 0.752 |
| friction | 1.000 | 1.000 | 1.000 | 0.987 | 0.293 | 0.856 |
| latency | 1.000 | 0.771 | 0.012 | 0.000 | 0.000 | 0.357 |
| push | 1.000 | 1.000 | 0.976 | 0.534 | 0.001 | 0.702 |

## Privileged policy at level 3, per seed

| Axis | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| motor_hip | 1.000 | 0.999 | 0.999 |
| motor_thigh | 1.000 | 0.999 | 0.999 |
| motor_calf | 0.998 | 0.999 | 0.998 |
| payload | 0.999 | 0.999 | 0.999 |
| friction | 1.000 | 1.000 | 1.000 |
| latency | 0.997 | 0.999 | 0.998 |
| push | 0.989 | 1.000 | 0.999 |
