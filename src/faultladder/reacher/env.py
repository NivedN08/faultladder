"""A two-joint arm that reaches a target, written directly on MuJoCo."""

# mujoco's compiled bindings ship without type stubs.
# pyright: reportAttributeAccessIssue=false

from importlib import resources
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from faultladder.reacher.oracle import LINK_LENGTHS

__all__ = ["LINK_LENGTHS", "ReacherEnv"]

CONTROL_HZ = 50
MAX_EPISODE_STEPS = 200

# One unit of action moves a joint's position target this far from where the
# joint is now.
ACTION_SCALE = 0.2  # rad

# Targets are drawn from a ring both elbow solutions can reach.
TARGET_RADIUS_RANGE = (0.06, 0.19)  # m

SUCCESS_DISTANCE = 0.02  # m
SUCCESS_SPEED = 0.05  # m/s
SUCCESS_STEPS = 10

ACTION_CHANGE_WEIGHT = 0.01
SUCCESS_BONUS = 1.0

# Lowest: fingertip and target at opposite ends of the workspace, and both
# actions flipping between -1 and 1. Highest: the success bonus at zero distance.
REWARD_BOUNDS = (
    -(sum(LINK_LENGTHS) + TARGET_RADIUS_RANGE[1]) - ACTION_CHANGE_WEIGHT * 8.0,
    SUCCESS_BONUS,
)


class ReacherEnv(gym.Env):
    """Planar two-link arm; the task is to hold the fingertip on a target.

    Observation (10): cos and sin of both joints, both joint velocities, target
    position, fingertip-to-target vector. With ``hide_target`` the last four
    are zero, which is the ablation that shows the policy needs the target.

    Action (2): in [-1, 1], scaled to an offset from each joint's current angle
    and sent to a position actuator.
    """

    metadata = {"render_modes": []}

    def __init__(self, hide_target: bool = False):
        xml = resources.files("faultladder.reacher").joinpath("reacher.xml").read_text()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.hide_target = hide_target

        self.frame_skip = round(1 / (CONTROL_HZ * self.model.opt.timestep))
        assert abs(self.frame_skip * self.model.opt.timestep * CONTROL_HZ - 1) < 1e-9

        self._tip_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "fingertip"
        )
        # The shoulder turns freely; resets draw it from one full turn.
        self.joint_range = np.array([[-np.pi, np.pi], self.model.jnt_range[1]])

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (10,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

        self.target = np.zeros(2)
        self._last_action = np.zeros(2)
        self._steps = 0
        self._still_steps = 0

    @property
    def fingertip(self) -> np.ndarray:
        return self.data.site_xpos[self._tip_id, :2].copy()

    @property
    def fingertip_speed(self) -> float:
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac, None, self._tip_id)
        return float(np.linalg.norm(jac @ self.data.qvel))

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        low, high = TARGET_RADIUS_RANGE
        # Square root of a uniform draw makes targets uniform over the ring's area.
        radius = np.sqrt(self.np_random.uniform(low**2, high**2))
        angle = self.np_random.uniform(-np.pi, np.pi)
        self.target = radius * np.array([np.cos(angle), np.sin(angle)])
        self.data.mocap_pos[0, :2] = self.target

        qpos = self.np_random.uniform(self.joint_range[:, 0], self.joint_range[:, 1])
        self.set_state(qpos, np.zeros(2))

        self._last_action = np.zeros(2)
        self._steps = 0
        self._still_steps = 0
        return self._observation(), {"is_success": False}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = self.data.qpos + ACTION_SCALE * action
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._steps += 1

        distance = float(np.linalg.norm(self.target - self.fingertip))
        on_target = distance < SUCCESS_DISTANCE and self.fingertip_speed < SUCCESS_SPEED
        self._still_steps = self._still_steps + 1 if on_target else 0
        success = self._still_steps >= SUCCESS_STEPS

        change = action - self._last_action
        self._last_action = action
        reward = -distance - ACTION_CHANGE_WEIGHT * float(change @ change)
        if success:
            reward += SUCCESS_BONUS

        # Success ends the task (terminated). Running out of time only cuts the
        # episode short (truncated), so a learner still bootstraps from it.
        terminated = success
        truncated = not terminated and self._steps >= MAX_EPISODE_STEPS
        info = {"is_success": success, "distance": distance}
        return self._observation(), reward, terminated, truncated, info

    def _observation(self) -> np.ndarray:
        q, qd = self.data.qpos, self.data.qvel
        if self.hide_target:
            target = to_target = np.zeros(2)
        else:
            target, to_target = self.target, self.target - self.fingertip
        obs = np.concatenate([np.cos(q), np.sin(q), qd, target, to_target])
        return obs.astype(np.float32)
