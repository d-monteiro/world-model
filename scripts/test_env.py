"""
Test script for the 2D Robotic Arm Environment

This script demonstrates the environment and shows how to interact with it.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.arm_env import Arm2DEnv


def test_environment():
    """Test the environment with random actions."""
    print("Creating 2D Robotic Arm Environment...")
    
    # Create environment
    env = Arm2DEnv(
        num_joints=2,
        num_objects=1,
        workspace_size=2.0,
        max_episode_steps=200,
        render_mode="human"
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()
    
    # Reset environment
    print("Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    print()
    
    # Initial render
    print("Opening visualization window...")
    env.render()
    time.sleep(1)  # Give window time to appear
    
    # Run episode with random actions
    print("Running episode with random actions...")
    print("Watch the robot move in the window!")
    print()
    
    for step in range(50):  # Run more steps to see movement
        # Random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render every step
        env.render()
        
        # Print status every 10 steps
        if step % 10 == 0:
            print(f"Step {step + 1}:")
            print(f"  Reward: {reward:.3f}")
            print(f"  Grasped object: {info['grasped_object']}")
            print(f"  Goal distance: {info['goal_distances'][0]:.3f}")
            print(f"  End-effector: ({info['end_effector_pos'][0]:.2f}, {info['end_effector_pos'][1]:.2f})")
            print()
        
        if terminated:
            print(f"  ✓ Goal reached at step {step + 1}!")
            break
        if truncated:
            print(f"  ✗ Max steps reached at step {step + 1}")
            break
    
    print("\nTest completed!")
    print("Close the visualization window to exit...")
    input("Press Enter to close...")
    env.close()


def test_observation_space():
    """Test observation space properties."""
    print("\n" + "="*50)
    print("Testing Observation Space")
    print("="*50)
    
    env = Arm2DEnv(num_joints=2, num_objects=1)
    obs, _ = env.reset(seed=42)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    print()
    
    # Break down observation components
    print("Observation breakdown:")
    print(f"  Joint angles (2): {obs[0:2]}")
    print(f"  Joint velocities (2): {obs[2:4]}")
    print(f"  Object position (2): {obs[4:6]}")
    print(f"  Goal position (2): {obs[6:8]}")
    print(f"  End-effector position (2): {obs[8:10]}")


def test_action_space():
    """Test action space properties."""
    print("\n" + "="*50)
    print("Testing Action Space")
    print("="*50)
    
    env = Arm2DEnv(num_joints=2, num_objects=1)
    
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Action space low: {env.action_space.low}")
    print(f"Action space high: {env.action_space.high}")
    print()
    
    # Sample some actions
    print("Sample actions:")
    for i in range(5):
        action = env.action_space.sample()
        print(f"  Action {i+1}: {action}")


if __name__ == "__main__":
    print("="*50)
    print("2D Robotic Arm Environment Test")
    print("="*50)
    print()
    
    # Test observation and action spaces
    test_observation_space()
    test_action_space()
    
    # Test full environment (with rendering)
    print("\n" + "="*50)
    print("Testing Full Environment (with rendering)")
    print("="*50)
    print()
    print("Note: A matplotlib window will open showing the arm.")
    print("Close the window to continue...")
    print()
    
    test_environment()
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)
