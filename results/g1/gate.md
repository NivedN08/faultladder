# FaultLadder validity gates

Unitree G1, ladder v1, 4096 episodes per cell. G1, G4 and G6 use the default policy of training seed 0; G3 and G5 use training seeds 0, 1, 2. The seed spread in G3 is the larger of the two baselines' standard deviations across seeds; G5 uses the mean across seeds.

Every row is recomputed from the saved runs by `uv run python -m faultladder.gate`. The last column is the command that produced those runs.

| Gate | Condition | Measured | Result | Runs produced by |
| --- | --- | --- | --- | --- |
| G1 | nominal >= 0.95 | 1.0000 | pass | `uv run modal run modal_app.py::baselines` |
| G2 | zero and random <= 0.05 | zero 0.0000; random 0.0000 | pass | `uv run modal run modal_app.py::ladder --agent zero (and --agent random)` |
| G3 | wide - no randomisation >= 0.15 and > 2 x seed spread | gap 0.6891; seed_spread 0.0152 | pass | `uv run modal run modal_app.py::baselines` |
| G4 | level 1 >= 0.8 and level 5 <= 0.5 per axis | failing axes: motor_calf | FAIL | `uv run modal run modal_app.py::baselines` |
| G5 | privileged level 3 >= 0.5 per axis | lowest 0.8084 | pass | `uv run modal run modal_app.py::baselines` |
| G6 | rescoring differs by <= 0.02 | 0.0009 | pass | `uv run modal run modal_app.py::ladder --name ppo_default_seed0_rescore --baseline default --seed 0 --eval-seed 1` |

At least one gate FAILS.

## Default policy, success rate per cell

| Axis | L1 | L2 | L3 | L4 | L5 | Axis score |
| --- | --- | --- | --- | --- | --- | --- |
| motor_hip | 1.000 | 0.997 | 0.643 | 0.415 | 0.062 | 0.623 |
| motor_thigh | 1.000 | 0.963 | 0.196 | 0.000 | 0.000 | 0.432 |
| motor_calf | 1.000 | 0.996 | 0.911 | 0.789 | 0.736 | 0.886 |
| payload | 1.000 | 0.815 | 0.150 | 0.013 | 0.000 | 0.396 |
| friction | 1.000 | 1.000 | 1.000 | 0.865 | 0.064 | 0.786 |
| latency | 1.000 | 0.498 | 0.000 | 0.000 | 0.000 | 0.300 |
| push | 0.999 | 0.911 | 0.243 | 0.002 | 0.000 | 0.431 |

## Privileged policy at level 3, per seed

| Axis | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| motor_hip | 0.946 | 0.981 | 0.965 |
| motor_thigh | 0.995 | 0.986 | 0.982 |
| motor_calf | 1.000 | 1.000 | 1.000 |
| payload | 1.000 | 1.000 | 1.000 |
| friction | 1.000 | 1.000 | 1.000 |
| latency | 0.997 | 1.000 | 1.000 |
| push | 0.862 | 0.793 | 0.770 |
