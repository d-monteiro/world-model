"""Custom Gymnasium MuJoCo environment for a 3-joint robotic arm."""

import os
import numpy as np
import gymnasium
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle
from gymnasium import spaces


class ThreeJointArmEnv(MujocoEnv, EzPickle):
    """3-joint robotic arm environment for object manipulation.

    Observation space (9D):
        [joint1, joint2, joint3, obj_x, obj_y, obj_z, target_x, target_y, target_z]

    Action space (3D):
        Motor torques for each joint, clipped to [-1, 1].

    Reward:
        Negative distance between object and target.

    Termination:
        Object within 0.05 of target.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, **kwargs):
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float64
        )
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs",
            "robot_world.xml",
        )
        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=5,
            observation_space=observation_space,
            **kwargs,
        )
        EzPickle.__init__(self, **kwargs)

    def _get_obs(self):
        # Joint angles (3 robot joints)
        joint_angles = self.data.qpos[:3].copy()

        # Object position (free joint: 3 pos + 4 quat, we take pos only)
        obj_pos = self.data.xpos[self.model.body("object").id].copy()

        # Target position
        target_pos = self.data.site("target").xpos.copy()

        return np.concatenate([joint_angles, obj_pos, target_pos])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()
        obj_pos = obs[3:6]
        target_pos = obs[6:9]

        dist = np.linalg.norm(obj_pos - target_pos)
        reward = -dist

        terminated = dist < 0.05
        truncated = False
        info = {"distance": dist, "success": terminated}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def reset_model(self):
        # Reset robot joints to small random values
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Randomize joint angles slightly
        qpos[:3] += self.np_random.uniform(low=-0.1, high=0.1, size=3)

        # Randomize object position within reachable workspace
        # Arm reach ~0.55m, keep positions within r<0.45 to be safe
        obj_angle = self.np_random.uniform(-np.pi, np.pi)
        obj_r = self.np_random.uniform(0.15, 0.4)
        obj_x = obj_r * np.cos(obj_angle)
        obj_y = obj_r * np.sin(obj_angle)
        obj_z = self.np_random.uniform(0.025, 0.15)
        # Object free joint: pos(3) + quat(4)
        qpos[3] = obj_x
        qpos[4] = obj_y
        qpos[5] = obj_z
        qpos[6:10] = [1, 0, 0, 0]  # Identity quaternion

        # Randomize target position (same reachable bounds)
        tgt_angle = self.np_random.uniform(-np.pi, np.pi)
        tgt_r = self.np_random.uniform(0.15, 0.4)
        target_x = tgt_r * np.cos(tgt_angle)
        target_y = tgt_r * np.sin(tgt_angle)
        target_z = self.np_random.uniform(0.025, 0.15)
        self.model.site("target").pos[:] = [target_x, target_y, target_z]

        # Zero velocities
        qvel[:] = 0

        self.set_state(qpos, qvel)
        return self._get_obs()


gymnasium.register(
    id="ThreeJointArm-v0",
    entry_point="physical_ai.envs.arm_env:ThreeJointArmEnv",
    max_episode_steps=200,
)
