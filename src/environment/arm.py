import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional


class Arm3(gym.Env):
    """
    State = [q1, q2, q3, obj_x, obj_y, goal_x, goal_y]
        - q_i are joint angles, in radians, limited to [-pi/2, pi/2]
        - obj_x, obj_y: object position on the floor (fixed)
        - goal_x, goal_y: goal position on the floor (fixed)

    Action = [dq1, dq2, dq3]
        - small changes to joint angles

    No rendering. Just state transitions and a simple reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        joint_limit: float = np.pi / 2,   # each joint in [-pi/2, pi/2]
        max_dq: float = 0.05,             # max change per step (radians)
        workspace_radius: float = 1.5,
        max_episode_steps: int = 200,
    ):
        super().__init__()

        self.n_joints = 3
        self.max_dq = float(max_dq)
        self.workspace_radius = float(workspace_radius)
        self.max_episode_steps = int(max_episode_steps)

        # --- Spaces ---

        # Per‑joint angle limits
        joint1_low = np.pi / 6.0         # ~30 degrees
        joint1_high = 5.0 * np.pi / 6.0  # ~150 degrees

        joint2_low = np.pi / 4.0        # 45 degrees
        joint2_high = (7.0 * np.pi) / 4.0    # 135 degrees

        joint3_low = np.pi / 4.0
        joint3_high = (7.0 * np.pi) / 4.0 

        self.joint_low = np.array(
            [joint1_low, joint2_low, joint3_low], dtype=np.float32
        )
        self.joint_high = np.array(
            [joint1_high, joint2_high, joint3_high], dtype=np.float32
        )

        # Simple 3-link planar arm: all links same length
        self.link_lengths = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # Action: delta joint angles in [-max_dq, max_dq]
        self.action_space = spaces.Box(
            low=-self.max_dq,
            high=self.max_dq,
            shape=(self.n_joints,),
            dtype=np.float32,
        )

        # State: [q1, q2, q3, obj_x, obj_y, goal_x, goal_y]
        # All positions in a disc of radius `workspace_radius`
        obs_low = np.concatenate(
            [
                self.joint_low,                          # q
                -workspace_radius * np.ones(4, np.float32),  # obj_x,obj_y,goal_x,goal_y
            ]
        )
        obs_high = np.concatenate(
            [
                self.joint_high,                         # q
                workspace_radius * np.ones(4, np.float32),
            ]
        )

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        # --- Internal state ---
        self.q = None            # joint angles (3,)
        self.obj_pos = None      # (2,)
        self.goal_pos = None     # (2,)
        self.step_count = 0

    # ------------------ core API ------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Random joint angles in [-joint_limit, joint_limit]
        self.q = self.np_random.uniform(
            low=self.joint_low,
            high=self.joint_high,
        ).astype(np.float32)

        # Random object and goal in workspace disc
        self.obj_pos = self._sample_in_disc(self.workspace_radius).astype(np.float32)
        self.goal_pos = self._sample_in_disc(self.workspace_radius).astype(np.float32)

        self.step_count = 0

        return self._get_obs(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Clip action
        action = np.clip(action, -self.max_dq, self.max_dq).astype(np.float32)

        # Update joint angles and clip to limits
        self.q = np.clip(self.q + action, self.joint_low, self.joint_high)

        self.step_count += 1

        # Simple reward: negative distance between object and goal
        dist = float(np.linalg.norm(self.obj_pos - self.goal_pos))
        reward = -dist

        # Termination when object "reaches" goal (purely geometric here)
        terminated = dist < 0.05
        truncated = self.step_count >= self.max_episode_steps

        info = {"distance_obj_goal": dist}

        return self._get_obs(), reward, terminated, truncated, info

    # ------------------ helpers ------------------

    def _get_obs(self) -> np.ndarray:
        """
        Observation: [q1, q2, q3, obj_x, obj_y, goal_x, goal_y]
        """
        return np.concatenate(
            [self.q, self.obj_pos, self.goal_pos],
        ).astype(np.float32)

    def _sample_in_disc(self, radius: float) -> np.ndarray:
        """
        Uniform sample in a 2D disc of given radius.
        """
        r = radius * np.sqrt(self.np_random.random())
        theta = 2 * np.pi * self.np_random.random()
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.array([x, y], dtype=np.float32)

    # ------------------ geometry / validity ------------------

    def _joint_positions(self, q: np.ndarray) -> np.ndarray:
        """
        Forward kinematics: return joint positions in 2D.

        Returns an array of shape (4, 2):
        [base, joint1, joint2, end_effector]
        """
        q = np.asarray(q, dtype=np.float32)

        x, y = 0.0, 0.0
        pts = [(x, y)]
        angle_sum = 0.0

        for angle, length in zip(q, self.link_lengths):
            angle_sum += float(angle)
            x += float(length) * np.cos(angle_sum)
            y += float(length) * np.sin(angle_sum)
            pts.append((x, y))

        return np.array(pts, dtype=np.float32)

    @staticmethod
    def _segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
        """
        Check if line segments p1-p2 and p3-p4 intersect (excluding shared endpoints).
        Small, standard 2D segment intersection test.
        """

        def orient(a, b, c) -> float:
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        o1 = orient(p1, p2, p3)
        o2 = orient(p1, p2, p4)
        o3 = orient(p3, p4, p1)
        o4 = orient(p3, p4, p2)

        # Proper intersection if orientations differ
        return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)

    def is_valid_q(self, q: np.ndarray) -> bool:
        """
        Check if joint configuration q is valid:
        - within per-joint limits
        - basic self-collision check: base->joint1 must not cross joint2->end_effector
        """
        q = np.asarray(q, dtype=np.float32)

        # Per-joint limits
        if np.any(q < self.joint_low) or np.any(q > self.joint_high):
            return False

        # Simple self-collision test
        pts = self._joint_positions(q)
        # base, joint1, joint2, end-effector
        p0, p1, p2, p3 = pts

        # Only check non-adjacent segments: base-joint1 vs joint2-end_effector
        if self._segments_intersect(p0, p1, p2, p3):
            return False

        return True


    def close(self):
        pass