# FaultLadder

A held-out robustness benchmark for quadruped walking policies.

**Status: planning. No code has been written yet.** Everything below describes
what the project intends to build; none of it has been implemented or measured.

## The idea

Reinforcement-learning walking controllers are usually trained with randomised
physics and then judged informally: someone watches a video, or reports reward
under the same randomisation the policy was trained on. FaultLadder aims to give
a policy one number instead.

A policy is trained any way its author likes. It is then scored on a frozen
ladder of faults and shifts it was not tuned on.

| Axis | What changes | Levels (draft) |
| --- | --- | --- |
| Motor | Strength of one leg joint, as a fraction of normal | 0.8, 0.6, 0.4, 0.2, 0.0 |
| Payload | Mass added to the trunk, as a fraction of robot mass | 0.1, 0.2, 0.3, 0.4, 0.5 |
| Friction | Foot-to-floor friction coefficient | 0.6, 0.45, 0.3, 0.2, 0.1 |
| Latency | Action delay, in 20 ms control steps | 1, 2, 3, 4, 5 |
| Push | Sideways velocity kick every 2 s, in m/s | 0.5, 1.0, 1.5, 2.0, 2.5 |

Each cell is scored by the fraction of episodes in which the robot stays up and
keeps tracking its commanded velocity. The FaultLadder score is the mean over
the five axes. These levels are a first draft and will be calibrated before a
versioned ladder is frozen.

## Planned stack

- Robot: Unitree Go1
- Simulator: [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)
- Environment framework: [mjlab](https://github.com/mujocolab/mjlab)
- Training: PPO from [rsl_rl](https://github.com/leggedrobotics/rsl_rl)

## Licence

Apache-2.0. See [LICENSE](LICENSE).
