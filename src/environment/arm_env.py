"""
2D Robotic Arm Environment for World Model Learning

A custom Gymnasium environment featuring:
- Multi-link robotic arm (2-3 joints)
- Movable objects
- Goal positions
- Vector-based observations
- Continuous action space
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any
import math
from .renderer import ArmRenderer


class Arm2DEnv(gym.Env):
    """
    2D Robotic Arm Environment
    
    The arm can move objects to goal positions. Observations include:
    - Joint angles and velocities
    - Object positions
    - Goal positions
    - End-effector position
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        num_joints: int = 2,
        num_objects: int = 1,
        workspace_size: float = 2.0,
        max_episode_steps: int = 200,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the 2D robotic arm environment.
        
        Args:
            num_joints: Number of arm joints (2-3 recommended)
            num_objects: Number of movable objects
            workspace_size: Size of the workspace (half-width/height)
            max_episode_steps: Maximum steps per episode
            render_mode: "human" or "rgb_array" for rendering
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.num_objects = num_objects
        self.workspace_size = workspace_size
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        # Arm parameters
        self.link_lengths = np.ones(num_joints) * 0.5  # Each link is 0.5 units
        self.joint_limits = np.array([[-np.pi, np.pi]] * num_joints)  # Full rotation
        self.max_joint_velocity = 0.2  # Radians per step
        
        # Object parameters
        self.object_radius = 0.1
        self.grasp_distance = 0.15  # Distance for "grasping" an object
        
        # Action space: joint velocities (change in joint angles)
        # Range: [-max_velocity, max_velocity] for each joint
        self.action_space = spaces.Box(
            low=-self.max_joint_velocity,
            high=self.max_joint_velocity,
            shape=(num_joints,),
            dtype=np.float32
        )
        
        # Observation space: [joint_angles, joint_velocities, object_positions, goal_positions, end_effector_pos]
        # Joint angles: num_joints
        # Joint velocities: num_joints
        # Object positions: num_objects * 2 (x, y)
        # Goal positions: num_objects * 2 (x, y)
        # End-effector position: 2 (x, y)
        obs_dim = (
            num_joints +  # joint angles
            num_joints +  # joint velocities
            num_objects * 2 +  # object positions
            num_objects * 2 +  # goal positions
            2  # end-effector position
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # State variables (initialized in reset)
        self.joint_angles = None
        self.joint_velocities = None
        self.object_positions = None
        self.goal_positions = None
        self.step_count = None
        self.grasped_object = None  # Index of currently grasped object (-1 if none)
        
        # Initialize renderer if needed
        self.renderer = None
        if render_mode is not None:
            self.renderer = ArmRenderer(workspace_size=workspace_size)
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize joint angles (random or zero)
        self.joint_angles = self.np_random.uniform(
            low=-0.5, high=0.5, size=(self.num_joints,)
        )
        self.joint_velocities = np.zeros(self.num_joints)
        
        # Initialize object positions (random in workspace)
        self.object_positions = self.np_random.uniform(
            low=-self.workspace_size * 0.6,
            high=self.workspace_size * 0.6,
            size=(self.num_objects, 2)
        )
        
        # Initialize goal positions (random, different from objects)
        self.goal_positions = self.np_random.uniform(
            low=-self.workspace_size * 0.6,
            high=self.workspace_size * 0.6,
            size=(self.num_objects, 2)
        )
        
        # Ensure goals are not too close to initial object positions
        for i in range(self.num_objects):
            min_dist = 0.5
            while np.linalg.norm(self.goal_positions[i] - self.object_positions[i]) < min_dist:
                self.goal_positions[i] = self.np_random.uniform(
                    low=-self.workspace_size * 0.6,
                    high=self.workspace_size * 0.6,
                    size=(2,)
                )
        
        self.step_count = 0
        self.grasped_object = -1  # No object grasped initially
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Joint velocity commands (change in joint angles)
            
        Returns:
            observation: Current state observation
            reward: Reward for this step
            terminated: Whether episode ended (goal reached)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        # Clip action to valid range
        action = np.clip(action, -self.max_joint_velocity, self.max_joint_velocity)
        
        # Update joint angles
        self.joint_angles += action
        self.joint_angles = np.clip(
            self.joint_angles,
            self.joint_limits[:, 0],
            self.joint_limits[:, 1]
        )
        
        # Update velocities (simple model: velocity = action)
        self.joint_velocities = action.copy()
        
        # Compute end-effector position
        end_effector_pos = self._forward_kinematics(self.joint_angles)
        
        # Check for object grasping
        self._update_grasping(end_effector_pos)
        
        # Update object positions if grasped
        if self.grasped_object >= 0:
            # Move object with end-effector
            self.object_positions[self.grasped_object] = end_effector_pos.copy()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check termination conditions
        terminated = self._check_goal_reached()
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute end-effector position from joint angles.
        
        Args:
            joint_angles: Array of joint angles (radians)
            
        Returns:
            End-effector position [x, y]
        """
        x, y = 0.0, 0.0  # Base position
        angle_sum = 0.0
        
        for i, (angle, length) in enumerate(zip(joint_angles, self.link_lengths)):
            angle_sum += angle
            x += length * np.cos(angle_sum)
            y += length * np.sin(angle_sum)
        
        return np.array([x, y])
    
    def _update_grasping(self, end_effector_pos: np.ndarray):
        """Update which object (if any) is currently grasped."""
        # If already grasping, check if still close enough
        if self.grasped_object >= 0:
            dist = np.linalg.norm(
                end_effector_pos - self.object_positions[self.grasped_object]
            )
            if dist > self.grasp_distance * 1.5:  # Release if too far
                self.grasped_object = -1
        
        # Check if end-effector is close to any object
        if self.grasped_object < 0:
            for i in range(self.num_objects):
                dist = np.linalg.norm(end_effector_pos - self.object_positions[i])
                if dist < self.grasp_distance:
                    self.grasped_object = i
                    break
    
    def _compute_reward(self) -> float:
        """
        Compute reward for current state.
        
        Reward structure:
        - Small negative reward per step (encourage efficiency)
        - Large positive reward when object reaches goal
        - Bonus for grasping objects
        """
        reward = -0.01  # Small step penalty
        
        # Check if objects are at goals
        for i in range(self.num_objects):
            dist_to_goal = np.linalg.norm(
                self.object_positions[i] - self.goal_positions[i]
            )
            
            # Reward for being close to goal
            if dist_to_goal < 0.1:
                reward += 10.0  # Large reward for reaching goal
            elif dist_to_goal < 0.3:
                reward += 1.0 * (0.3 - dist_to_goal) / 0.3  # Gradual reward
            
            # Small reward for grasping
            if self.grasped_object == i:
                reward += 0.1
        
        return reward
    
    def _check_goal_reached(self) -> bool:
        """Check if all objects are at their goal positions."""
        for i in range(self.num_objects):
            dist = np.linalg.norm(
                self.object_positions[i] - self.goal_positions[i]
            )
            if dist > 0.1:  # Threshold for "at goal"
                return False
        return True
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        end_effector_pos = self._forward_kinematics(self.joint_angles)
        
        # Normalize joint angles to [-1, 1] range
        normalized_angles = self.joint_angles / np.pi
        
        # Normalize velocities
        normalized_velocities = self.joint_velocities / self.max_joint_velocity
        
        # Normalize positions by workspace size
        normalized_objects = self.object_positions.flatten() / self.workspace_size
        normalized_goals = self.goal_positions.flatten() / self.workspace_size
        normalized_ee = end_effector_pos / self.workspace_size
        
        # Concatenate all observations
        obs = np.concatenate([
            normalized_angles,
            normalized_velocities,
            normalized_objects,
            normalized_goals,
            normalized_ee,
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state."""
        end_effector_pos = self._forward_kinematics(self.joint_angles)
        
        # Compute distances to goals
        goal_distances = [
            np.linalg.norm(self.object_positions[i] - self.goal_positions[i])
            for i in range(self.num_objects)
        ]
        
        return {
            "end_effector_pos": end_effector_pos.copy(),
            "grasped_object": self.grasped_object,
            "goal_distances": goal_distances,
            "step_count": self.step_count,
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        if self.renderer is None:
            self.renderer = ArmRenderer(workspace_size=self.workspace_size)
        
        if self.render_mode == "human":
            self.renderer.render(
                joint_angles=self.joint_angles,
                link_lengths=self.link_lengths,
                object_positions=self.object_positions,
                goal_positions=self.goal_positions,
                grasped_object=self.grasped_object,
                show=True
            )
            return None
        elif self.render_mode == "rgb_array":
            rgb_array = self.renderer.render(
                joint_angles=self.joint_angles,
                link_lengths=self.link_lengths,
                object_positions=self.object_positions,
                goal_positions=self.goal_positions,
                grasped_object=self.grasped_object,
                show=False
            )
            return rgb_array
    
    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
