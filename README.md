# FaultLadder

A held-out robustness benchmark for legged walking policies. A policy is trained
any way its author likes, then scored on a frozen ladder of motor faults and
dynamics shifts it was not tuned on. The result is one number, with a per-axis
breakdown, that can be compared across papers.

FaultLadder runs on the Unitree Go1 quadruped and the Unitree G1 humanoid in
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) through
[mjlab](https://github.com/mujocolab/mjlab). Baselines are PPO from
[rsl_rl](https://github.com/leggedrobotics/rsl_rl).

## What it measures

The robot is given one velocity command per episode (at least 0.4 m/s) on flat
ground for 10 s. An episode succeeds if the robot does not fall and, over the
last 8 s, its mean planar velocity error is at most 0.25 m/s and its mean
yaw-rate error at most 0.5 rad/s.

The ladder has seven axes with five levels each, plus a cell with no
perturbation: 36 cells of 4,096 episodes.

| Axis | What changes | Levels 1 to 5 |
| --- | --- | --- |
| Motor, hip | Strength of one hip joint, as a fraction of normal | 0.8, 0.6, 0.4, 0.2, 0.0 |
| Motor, thigh | Strength of one thigh joint (G1: one knee) | 0.8, 0.6, 0.4, 0.2, 0.0 |
| Motor, calf | Strength of one calf joint (G1: one ankle joint) | 0.8, 0.6, 0.4, 0.2, 0.0 |
| Push | Sideways velocity kick every 2 s, in m/s | 0.5, 0.75, 1.0, 1.25, 1.5 |
| Payload | Mass added to the trunk (G1: torso), as a fraction of robot mass | 0.12, 0.24, 0.36, 0.48, 0.6 |
| Friction | Foot-to-floor friction coefficient | 0.6, 0.45, 0.3, 0.2, 0.1 |
| Latency | Action delay, in 20 ms control steps | 0.5, 1, 2, 3, 5 |

A cell's score is the fraction of successful episodes. An axis score is the
mean over its five levels, and the FaultLadder score is the mean over the seven
axes. Two sub-scores are reported with it: fault (the three motor axes and
push) and shift (payload, friction and latency). The levels are frozen in
`src/faultladder/ladder_v1.json` and are the same for both robots; the score is
defined once, in `src/faultladder/score.py`.

Levels 4 and 5 are held out: the strongest baselines here train on
perturbations up to level 3 only.

## Quickstart

Scoring needs a CUDA GPU. The commands below use [Modal](https://modal.com);
one ladder run takes 10 to 20 minutes on an A10 and costs about $0.40.

```sh
git clone https://github.com/NivedN08/faultladder && cd faultladder
uv sync --all-extras
uv run modal setup
gh release download v1.0.0 --pattern ppo_default_seed0.pt
uv run modal run modal_app.py::ladder --name mine --weights ppo_default_seed0.pt
```

The last command prints the scores and writes `results/mine/ladder.json`. The
released checkpoint is the seed 0 default baseline, which scores 0.697. To score
your own policy, pass its rsl_rl checkpoint as `--weights`; it must use the
observations and actions of mjlab's `Mjlab-Velocity-Flat-Unitree-Go1` task
(add `--robot g1` for `Mjlab-Velocity-Flat-Unitree-G1`).

A checkpoint is a PyTorch pickle file and can run code when loaded, so only
score checkpoints from a source you trust.

On a machine with its own GPU, without Modal:

```sh
uv run python -m faultladder.evaluate --agent ppo \
  --checkpoint ppo_default_seed0.pt --out results/mine --device cuda:0
```

A no-fault video and one 10 s video per axis at level 3 (written to `scratch/videos`):

```sh
uv run modal run modal_app.py::videos --baseline default --seed 0
```

To train and score the baselines yourself (about $0.80 per Go1 policy):

```sh
uv run modal run modal_app.py::baselines --names default --seeds 0
```

## Leaderboard

Mean over training seeds ± the half-width of the 95% confidence interval. The
full tables, with every axis, are generated from the saved runs:
[Go1](results/leaderboard.md), [G1](results/g1/leaderboard.md).

| Policy | What it trains on | Go1 (5 seeds) | G1 (3 seeds) |
| --- | --- | --- | --- |
| PPO privileged | Wide, and the policy is told the fault parameters | 0.874 ± 0.010 | 0.808 ± 0.001 |
| PPO wide, recurrent | Wide, with an LSTM policy | 0.868 ± 0.008 | not run |
| PPO wide | Every axis randomised up to level 3 | 0.846 ± 0.017 | 0.689 ± 0.038 |
| PPO default | mjlab's task as shipped | 0.719 ± 0.017 | 0.561 ± 0.026 |
| PPO no randomisation | No randomisation | 0.667 ± 0.017 | 0.000 ± 0.000 |
| Zero agent, random agent | Nothing | 0.000 | 0.000 |

Go1 policies train for 1,320 iterations and G1 policies for 5,000.

## Compound and onset tracks

Two further tracks are scored on the Go1 ([results/tracks.md](results/tracks.md)).

The compound track perturbs two axes at once, at levels 2 and 3, for the 18
axis pairs that do not combine two motor faults. Each result is compared with
the success predicted from the same policy's single-axis cells if the two
perturbations hurt independently. Every baseline does worse than predicted, and
the shortfall grows as the policy gets less robust:

| Policy | Compound | Predicted | Shortfall |
| --- | --- | --- | --- |
| PPO wide, recurrent | 0.976 | 0.999 | 0.023 |
| PPO privileged | 0.974 | 0.998 | 0.024 |
| PPO wide | 0.940 | 0.985 | 0.046 |
| PPO default | 0.689 | 0.787 | 0.098 |
| PPO no randomisation | 0.589 | 0.774 | 0.185 |

So single-axis scores overstate robustness to combined perturbations, most for
the weakest policies. Pushes combined with latency, payload or a weak calf
motor are the worst pairs.

The onset track starts a motor fault 5 s into the episode. For every baseline
the score is within 0.01 of the same fault present from the start, so the
ladder's static faults stand in for sudden ones on this task.

## Validity gates

A ladder version is frozen only if the benchmark passes six checks on itself.

| Gate | Condition | Go1 | G1 |
| --- | --- | --- | --- |
| G1 nominal | Default PPO succeeds at least 0.95 with no perturbation | pass | pass |
| G2 floor | Zero and random agents score at most 0.05 | pass | pass |
| G3 separation | Wide beats no randomisation by 0.15 and twice the seed spread | pass | pass |
| G4 informative | Level 1 at least 0.8 and level 5 at most 0.5 on every axis | pass | fails on calf |
| G5 solvable | Privileged policy at least 0.5 at level 3 on every axis | pass | pass |
| G6 repeatable | Rescoring with another seed differs by at most 0.02 | pass | pass |

Measured values: [Go1](results/gate.md), [G1](results/g1/gate.md).
[results/sensitivity.md](results/sensitivity.md) rescales the two success
tolerances from 0.5 to 2 times; the Go1 ranking is unchanged from 1 times
upward, and below that only the recurrent and privileged policies swap.

## Limitations

- Simulation only. No result here has been checked on a real robot.
- Flat ground only, and one simulator.
- The levels were calibrated on the Go1. On the G1 a dead ankle joint still
  succeeds 74% of the time, so that axis does not reach a hard level and G4
  fails there.
- The G1 policy trained without randomisation never tracks its command after
  5,000 iterations, so its score of 0 says that this training setup failed, not
  how robust a working policy of that kind would be. G3 passes on the G1
  trivially for the same reason.
- On the Go1 the wide policy scores above 0.9 on payload and latency, which
  leaves little room above it on those axes, and latency is steep for the
  default policy (0.77 at level 2, 0.01 at level 3).
- A motor fault weakens one joint's gains and torque limit. Locked joints,
  sensor faults and intermittent faults are not covered.
- The G1 has three seeds per baseline and no compound, onset or recurrent
  results.
- Baselines are PPO only, with one iteration budget per robot.

## Related work

[RoboGauge](https://github.com/wty-yy/RoboGauge) (RSS 2026,
[paper](https://arxiv.org/abs/2602.00678)) is the closest existing benchmark. It
evaluates Unitree Go2 policies in MuJoCo across five terrains and across
payload, friction and delay settings, gives one aggregate score, and was checked
against real-robot trials. FaultLadder does not cover terrain. It overlaps
RoboGauge on payload, friction and latency, and adds graded per-joint-type
motor faults, graded pushes, a compound track, a second robot morphology, and a
set of stated checks that the benchmark itself must pass.

[Robust-Gymnasium](https://github.com/SafeRL-Lab/Robust-Gymnasium) perturbs
observations, actions and dynamics, mostly on standard Gymnasium models.

Papers on fault-tolerant locomotion commonly test mid-episode and simultaneous
faults, each with its own protocol. The compound and onset tracks here are a
fixed protocol for those, not a new kind of test.

## Development

```sh
uv sync --all-extras
./scripts/check.sh    # format, lint, type check and tests, as CI runs them
```

Pinned: Python 3.12, mjlab 1.6.0, mujoco 3.11.0, mujoco-warp 3.11.0,
rsl-rl-lib 5.4.2. `src/faultladder/reacher` is a small two-joint arm
environment used to practise the workflow; it is not part of the benchmark.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
