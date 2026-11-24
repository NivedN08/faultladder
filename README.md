# FaultLadder

A held-out robustness benchmark for quadruped walking policies.

**Status: early. The benchmark itself does not exist yet.** So far the repository
holds the project setup and a small practice environment (a two-joint arm,
`src/faultladder/reacher`) with its checks; its numbers are in
`results/reacher.json`. Everything below describes what the project intends to
build; none of it has been implemented or measured.

## The idea

Reinforcement-learning walking controllers are trained with randomised physics,
but how they cope with a failing motor is rarely measured in a comparable way.
FaultLadder aims to give a policy one number for that.

A policy is trained any way its author likes. It is then scored on a frozen
ladder of faults and shifts it was not tuned on.

| Axis | What changes | Levels (ladder v1) |
| --- | --- | --- |
| Motor, hip | Strength of one hip joint, as a fraction of normal | 0.8, 0.6, 0.4, 0.2, 0.0 |
| Motor, thigh | Strength of one thigh joint, as a fraction of normal | 0.8, 0.6, 0.4, 0.2, 0.0 |
| Motor, calf | Strength of one calf joint, as a fraction of normal | 0.8, 0.6, 0.4, 0.2, 0.0 |
| Push | Sideways velocity kick every 2 s, in m/s | 0.5, 0.75, 1.0, 1.25, 1.5 |
| Payload | Mass added to the trunk, as a fraction of robot mass | 0.12, 0.24, 0.36, 0.48, 0.6 |
| Friction | Foot-to-floor friction coefficient | 0.6, 0.45, 0.3, 0.2, 0.1 |
| Latency | Action delay, in 20 ms control steps | 0.5, 1, 2, 3, 5 |

Each cell is scored by the fraction of episodes in which the robot stays up and
keeps tracking its commanded velocity. The FaultLadder score is the mean over
the seven axes, reported with two sub-scores: a fault score (the three motor
axes and push) and a shift score (payload, friction and latency). The levels
are frozen in `src/faultladder/ladder_v1.json`. Baseline scores are in
[results/leaderboard.md](results/leaderboard.md) and the validity gates in
[results/gate.md](results/gate.md).

## Related work

[RoboGauge](https://github.com/wty-yy/RoboGauge) (RSS 2026,
[paper](https://arxiv.org/abs/2602.00678)) is the closest existing benchmark. It
evaluates Unitree Go2 policies in MuJoCo across five terrains and across
payload, friction and delay settings, gives one aggregate score, and was checked
against real-robot trials. FaultLadder does not cover terrain. It overlaps
RoboGauge on payload, friction and latency, and adds graded motor faults, graded
pushes, and a set of stated checks that the benchmark itself must pass before a
ladder version is frozen.

[Robust-Gymnasium](https://github.com/SafeRL-Lab/Robust-Gymnasium) perturbs
observations, actions and dynamics, mostly on standard Gymnasium models.

## Planned stack

- Robot: Unitree Go1
- Simulator: [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)
- Environment framework: [mjlab](https://github.com/mujocolab/mjlab)
- Training: PPO from [rsl_rl](https://github.com/leggedrobotics/rsl_rl)

## Licence

Apache-2.0. See [LICENSE](LICENSE).
