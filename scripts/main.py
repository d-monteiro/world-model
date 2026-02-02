"""
Main script to control the Arm robot using the learned World Model (MPC).

This script:
1. Loads the environment (Arm3).
2. Loads the MPC Planner (which uses VAE + RNN).
3. At each step, uses the Planner to find the best action to reach the object.
4. Executes the action and visualizes the result.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environment.arm import Arm3
from src.planning.mpc import MPCPlanner

def main():
    print("Initializing Environment and World Model Planner...")
    
    # 1. Setup Environment
    # Use "human" for visualization, or None for speed
    env = Arm3(render_mode="human")
    
    # 2. Setup Planner
    # Ensure models exist
    if not (ROOT / "src/models/vae_weights.pth").exists():
        print("❌ Model weights not found! Please run 'run_pipeline.py' first.")
        return

    # Initialize Planner
    # horizon=15: look 15 steps ahead
    # n_candidates=500: test 500 parallel random action sequences
    planner = MPCPlanner(
        vae_path="src/models/vae_weights.pth",
        rnn_path="src/models/rnn_weights.pth",
        horizon=15,
        n_candidates=500
    )
    
    print("\nStarting Control Loop. Goal: Reach the Blue Object!")
    
    try:
        # Run 3 episodes
        for ep in range(3):
            print(f"\n--- Episode {ep+1} ---")
            
            # Reset Env
            obs, info = env.reset(seed=ep*100) # Different seed per episode
            
            # Show initial state
            if env.render_mode == "human":
                env.render()
                
            # Control Loop (200 steps max)
            for t in range(200):
                # Get Object Position from state [q1, q2, q3, ox, oy, gx, gy]
                # Object X,Y are at indices 3 and 4
                obj_pos = obs[3:5]
                
                # --- PLANNING STEP ---
                # The planner uses the World Model to choose the best action
                # to get the end-effector close to obj_pos
                action = planner.plan(obs, target_pos=obj_pos)
                
                # Execute in Reality
                obs, reward, terminated, truncated, info = env.step(action)
                
                if env.render_mode == "human":
                    env.render()
                
                # Check status
                dist = info['distance_obj_goal'] # In arm.py this is obj-goal distance
                # But we want hand-obj distance.
                # Let's trust the visual or calculate manually if needed.
                
                # Simple log
                if t % 10 == 0:
                    print(f"Step {t}: Planning... Action: {action.round(3)}")
                
                if terminated:
                    print("🎉 Success! Task completed based on env criteria.")
                    break
                    
                if truncated:
                    print("⏰ Time limit reached.")
                    break
                    
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        env.close()

if __name__ == "__main__":
    main()
