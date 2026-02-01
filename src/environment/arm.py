import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Arm3(gym.Env):
    """
    State = [q1, q2, q3, obj_x, obj_y, goal_x, goal_y]
        - q_i are joint angles, in radians, limited to [-pi/2, pi/2]
        - obj_x, obj_y: object position on the floor (fixed)
        - goal_x, goal_y: goal position on the floor (fixed)

    Action = [dq1, dq2, dq3]
        - small changes to joint angles

    Rendering modes: "human" (interactive window), "rgb_array" (numpy array)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        joint_limit: float = np.pi / 2,   # each joint in [-pi/2, pi/2]
        max_dq: float = 0.05,             # max change per step (radians)
        workspace_radius: float = 1.5,
        max_episode_steps: int = 200,
        grasp_threshold: float = 0.15,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.n_joints = 3
        self.grasp_threshold = float(grasp_threshold)
        self.max_dq = float(max_dq)
        self.workspace_radius = float(workspace_radius)
        self.max_episode_steps = int(max_episode_steps)

        # Rendering
        self.fig = None
        self.ax = None

        # --- Spaces ---

        # Per-joint angle limits (relative to previous link)
        joint1_low = np.pi / 6.0         # 30 degrees
        joint1_high = 5.0 * np.pi / 6.0  # 150 degrees

        joint2_low = -np.pi / 2.0        # -90 degrees (fold inward)
        joint2_high = np.pi / 2.0        #  90 degrees (fold outward)

        joint3_low = -np.pi / 2.0        # -90 degrees
        joint3_high = np.pi / 2.0        #  90 degrees

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
        self.grasping = False    # is the arm holding the object?
        self.step_count = 0

    # ------------------ core API ------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Random valid joint angles (reject self-collisions)
        while True:
            q = self.np_random.uniform(
                low=self.joint_low,
                high=self.joint_high,
            ).astype(np.float32)
            if self.is_valid_q(q):
                break
        self.q = q

        # Object and goal: sample reachable positions (end-effector of random valid configs)
        self.obj_pos = self._sample_reachable_pos().astype(np.float32)
        self.goal_pos = self._sample_reachable_pos().astype(np.float32)

        self.grasping = False
        self.step_count = 0

        return self._get_obs(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Clip action
        action = np.clip(action, -self.max_dq, self.max_dq).astype(np.float32)

        # Update joint angles and clip to limits
        new_q = np.clip(self.q + action, self.joint_low, self.joint_high)

        # Only accept if no self-collision
        if self.is_valid_q(new_q):
            self.q = new_q

        self.step_count += 1

        # Grasping: if end-effector is close to object, grab it
        ee_pos = self._joint_positions(self.q)[-1]  # end-effector position
        ee_to_obj = float(np.linalg.norm(ee_pos - self.obj_pos))

        if ee_to_obj < self.grasp_threshold:
            self.grasping = True

        # If grasping, object follows the end-effector
        if self.grasping:
            self.obj_pos = ee_pos.copy()

        # Reward: negative distance between object and goal
        dist_obj_goal = float(np.linalg.norm(self.obj_pos - self.goal_pos))
        reward = -dist_obj_goal

        # Termination when object reaches goal
        terminated = dist_obj_goal < 0.05
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "distance_obj_goal": dist_obj_goal,
            "distance_ee_obj": ee_to_obj,
            "grasping": self.grasping,
        }

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

    def _sample_reachable_pos(self) -> np.ndarray:
        """Sample a position the arm can actually reach."""
        while True:
            q = self.np_random.uniform(
                low=self.joint_low, high=self.joint_high,
            ).astype(np.float32)
            if self.is_valid_q(q):
                return self._joint_positions(q)[-1]

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

    # ------------------ rendering ------------------

    def render(self):
        """
        Render the current state of the environment.
        
        Returns:
            - If render_mode is "rgb_array": returns numpy array of shape (H, W, 3)
            - If render_mode is "human": displays the figure and returns None
        """
        if self.render_mode is None:
            return None

        # Create figure if it doesn't exist
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            if self.render_mode == "human":
                plt.ion()  # Interactive mode for human rendering

        self.ax.clear()
        
        # Set up the plot
        r = self.workspace_radius
        self.ax.set_xlim(-r, r)
        self.ax.set_ylim(-r, r)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title(f'3-Link Arm (Step {self.step_count})')

        # Draw workspace boundary
        workspace_circle = plt.Circle((0, 0), r, color='gray', fill=False, 
                                     linestyle='--', linewidth=1, alpha=0.5)
        self.ax.add_patch(workspace_circle)

        # Draw the arm
        joint_positions = self._joint_positions(self.q)
        
        # Draw links
        for i in range(len(joint_positions) - 1):
            x_vals = [joint_positions[i][0], joint_positions[i+1][0]]
            y_vals = [joint_positions[i][1], joint_positions[i+1][1]]
            self.ax.plot(x_vals, y_vals, 'o-', linewidth=3, markersize=8, 
                        color='darkblue', label='Arm' if i == 0 else '')

        # Draw base (origin)
        self.ax.plot(0, 0, 'ks', markersize=12, label='Base')

        # Draw end-effector
        end_effector = joint_positions[-1]
        self.ax.plot(end_effector[0], end_effector[1], 'ro', markersize=12, 
                    label='End-Effector')

        # Draw object (orange if grasped, blue if free)
        obj_color = '#e67e22' if self.grasping else '#3498db'
        obj_label = 'Object (grasped)' if self.grasping else 'Object'
        self.ax.plot(self.obj_pos[0], self.obj_pos[1], 'o', color=obj_color,
                    markersize=15, label=obj_label, zorder=5)

        # Draw goal
        self.ax.plot(self.goal_pos[0], self.goal_pos[1], 'g*', markersize=20,
                    label='Goal')

        # Draw grasp radius around end-effector
        grasp_circle = plt.Circle((end_effector[0], end_effector[1]),
                                  self.grasp_threshold, color='red',
                                  fill=False, linestyle=':', linewidth=1, alpha=0.4)
        self.ax.add_patch(grasp_circle)

        # Info text
        dist = float(np.linalg.norm(self.obj_pos - self.goal_pos))
        status = "GRASPING" if self.grasping else "free"
        self.ax.text(0.02, 0.98, f'Obj→Goal: {dist:.3f}\nStatus: {status}',
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        self.ax.legend(loc='upper right')

        if self.render_mode == "human":
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
            return None
        elif self.render_mode == "rgb_array":
            # Convert canvas to RGB array
            self.fig.canvas.draw()
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None