"""
Interactive visualization of the 2D Robotic Arm

This script shows the robot moving in real-time with a longer episode.
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


def visualize_robot_movement():
    """Show the robot moving with random actions."""
    print("="*60)
    print("2D Robotic Arm - Interactive Visualization")
    print("="*60)
    print()
    
    # Create environment
    print("Creating environment...")
    env = Arm2DEnv(
        num_joints=2,
        num_objects=1,
        workspace_size=2.0,
        max_episode_steps=500,
        render_mode="human"
    )
    
    # Reset
    obs, info = env.reset(seed=42)
    
    print("Environment ready!")
    print(f"Object position: ({info['end_effector_pos'][0]:.2f}, {info['end_effector_pos'][1]:.2f})")
    print(f"Goal distance: {info['goal_distances'][0]:.3f}")
    print()
    print("Opening visualization window...")
    print("Watch the robot move!")
    print()
    
    # Initial render
    env.render()
    time.sleep(1.5)  # Give window time to appear
    
    step_count = 0
    print("Starting episode...")
    print("Press Ctrl+C to stop early")
    print()
    
    try:
        while True:
            # Random action
            action = env.action_space.sample()
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render
            env.render()
            
            step_count += 1
            
            # Print status every 20 steps
            if step_count % 20 == 0:
                print(f"Step {step_count}: "
                      f"Reward={reward:.3f}, "
                      f"Grasped={info['grasped_object']}, "
                      f"Goal_dist={info['goal_distances'][0]:.3f}")
            
            if terminated:
                print(f"\n✓ Success! Goal reached at step {step_count}")
                break
            
            if truncated:
                print(f"\n✗ Episode ended at step {step_count} (max steps)")
                break
            
            # Small delay for smoother animation
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print(f"\n\nStopped by user at step {step_count}")
    
    print("\nEpisode complete!")
    print("Close the visualization window when done viewing...")
    input("Press Enter to close...")
    
    env.close()
    print("Done!")


if __name__ == "__main__":
    visualize_robot_movement()
