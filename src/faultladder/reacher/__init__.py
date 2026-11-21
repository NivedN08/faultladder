"""Phase 1: a two-joint reacher written directly on MuJoCo and Gymnasium."""

import gymnasium as gym

ENV_ID = "FaultLadder/Reacher-v0"

if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="faultladder.reacher.env:ReacherEnv")
