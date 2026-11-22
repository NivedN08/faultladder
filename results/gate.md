# FaultLadder validity gates

Ladder v0, 4096 episodes per cell. Default policy: `/runs/Mjlab-Velocity-Flat-Unitree-Go1/seed0/model_2999.pt`, seed 0; rescored with seed 1.

| Gate | Condition | Measured | Result |
| --- | --- | --- | --- |
| G1 | nominal >= 0.95 | 0.9993 | pass |
| G2 | zero and random <= 0.05 | {'zero': 0.0, 'random': 0.0} | pass |
| G4 | level 1 >= 0.8 and level 5 <= 0.5 per axis | failing axes: [] | pass |
| G6 | rescoring differs by <= 0.02 | 0.0003 | pass |

G3 and G5 need the Phase 4 baselines and are not checked here.

## Default policy, success rate per cell

| Axis | L1 | L2 | L3 | L4 | L5 | Axis score |
| --- | --- | --- | --- | --- | --- | --- |
| motor_hip | 1.000 | 1.000 | 0.996 | 0.921 | 0.189 | 0.821 |
| motor_thigh | 1.000 | 1.000 | 0.999 | 0.998 | 0.258 | 0.851 |
| motor_calf | 1.000 | 0.997 | 0.818 | 0.136 | 0.003 | 0.591 |
| payload | 1.000 | 0.998 | 0.988 | 0.793 | 0.399 | 0.836 |
| friction | 1.000 | 1.000 | 1.000 | 0.956 | 0.219 | 0.835 |
| latency | 0.964 | 0.117 | 0.000 | 0.000 | 0.000 | 0.216 |
| push | 0.999 | 0.993 | 0.000 | 0.000 | 0.000 | 0.399 |

| Agent | FaultLadder | Fault | Shift | Nominal | Fall rate |
| --- | --- | --- | --- | --- | --- |
| ppo_default | 0.650 | 0.665 | 0.629 | 0.999 | 0.179 |
| zero | 0.000 | 0.000 | 0.000 | 0.000 | 0.276 |
| random | 0.000 | 0.000 | 0.000 | 0.000 | 0.800 |
