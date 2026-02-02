"""
Visualize the World Model 'dreaming' in parallel with reality.

This script:
1. Loads the trained VAE and RNN.
2. Runs the real environment (Arm3).
3. Runs the World Model (RNN) purely based on actions (Open-Loop).
4. Visualizes both:
   - BLUE Arm: Real Physics (Ground Truth)
   - RED Arm: World Model Prediction (Dream)

If the Red arm follows the Blue arm closely, your World Model works!
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environment.arm import Arm3
    # --- 1. Load Models ---
    device = torch.device('cpu') # Inference is fast enough on CPU
    print(f"Loading models on {device}...")
    
    # VAE
    vae = VAE(input_dim=3, latent_dim=2).to(device)
    vae.load_state_dict(torch.load(ROOT / 'src/models/vae_weights.pth', map_location=device))
    vae.eval()
    
    # RNN
    # Input dim for RNN was (z_dim + context_dim) = 2 + 4 = 6
    # But wait, in train_rnn.py we might have set it up differently. 
    # Let's check the logic we implemented in train_rnn.py...
    # We decided to Concatenate Context to Action.
    # So rnn input size was latent_dim=2.
    # And action_dim was 3 + 4 = 7.
    
    rnn = MDNRNN(
        latent_dim=2,
        action_dim=7, # 3 (action) + 4 (context)
        hidden_size=256,
        num_gaussians=5
    ).to(device)
    
    try:
        rnn.load_state_dict(torch.load(ROOT / 'src/models/rnn_weights.pth', map_location=device))
    except FileNotFoundError:
        print("RNN weights not found! Please run the pipeline first.")
        return
        
    rnn.eval()
    print("Models loaded successfully.")

    # --- 2. Setup Environment ---
    env = Arm3(render_mode="human")
    obs, info = env.reset()
    
    # Initialize Dream State (Warm start with real first state)
    z_curr, context = get_latent_state(vae, obs)
    
    # RNN Hidden state
    rnn_hidden = None
    
    print("\nStarting Simulation...")
    print("BLUE = REALITY")
    print("RED  = DREAM (Model Prediction)")
    
    # Access plot axes for custom drawing
    if env.render_mode == "human":
        env.render() # Spawns the window
        ax = env.ax
    
    try:
        for t in range(500):
            # 1. Action (Random smooth exploration or policy)
            # Let's use the same smoothing logic as collect_data for nice movements
            target_action = env.action_space.sample()
            if t % 20 == 0:
                 current_target = target_action
            
            # Simple random action for now, or user can control?
            action = env.action_space.sample() * 0.5 # Slow down a bit
            
            # 2. Step Reality
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # 3. Step Dream (RNN)
            # Prepare inputs
            # RNN expects: z (1, 1, 2) and action_with_context (1, 1, 7)
            z_in = torch.FloatTensor(z_curr).view(1, 1, 2)
            
            # Context is assumed constant for the dream (we know obj/goal don't move)
            # Construct action vector: [dq1, dq2, dq3, ox, oy, gx, gy]
            act_ctx = np.concatenate([action, context])
            act_in = torch.FloatTensor(act_ctx).view(1, 1, 7)
            
            with torch.no_grad():
                (pi, mu, sigma), rnn_hidden = rnn(z_in, act_in, rnn_hidden)
                
                # Deterministic prediction: Take the Gaussian with highest weight (or just mean of main one)
                # For visualization, we usually pick the mode or sample.
                # Let's take the mean of the component with highest pi
                k = torch.argmax(pi[0, 0])
                z_next_pred = mu[0, 0, k].numpy()
                
            # 4. Decode Dream to Joint Angles
            q_dream = decode_latent_state(vae, z_next_pred)
            
            # 5. Render
            env.render()
            
            # -- Custom Overlay Drawing --
            if env.render_mode == "human" and ax is not None:
                # Calculate positions for Dream Arm
                # We need access to helper functions in env, but they are instance methods.
                # We can cheat and use the env instance
                
                # Save real state temporarily
                real_q = env.q.copy()
                
                # Pretend env is in dream state to get positions
                # (This is a hack to reuse the _joint_positions logic without rewriting it)
                pts = env._joint_positions(q_dream)
                
                # Draw Dream Arm (Red, dashed)
                # Remove previous dream lines if we stored them? 
                # Matplotlib animation is tricky in a simple loop.
                # Easiest way: Plot on top, simple call.
                # But env.render() clears the ax! So we must draw AFTER env.render()
                
                for i in range(len(pts) - 1):
                    x_vals = [pts[i][0], pts[i+1][0]]
                    y_vals = [pts[i][1], pts[i+1][1]]
                    ax.plot(x_vals, y_vals, 'o--', linewidth=2, markersize=5, 
                            color='red', alpha=0.6, label='Dream' if i==0 else "")
                
                # Draw Dream End-Effector
                ee = pts[-1]
                ax.plot(ee[0], ee[1], 'r*', markersize=10, alpha=0.8)
                
                # Add Legend once
                # ax.legend() 
                
                # Refresh
                env.fig.canvas.draw()
                env.fig.canvas.flush_events()
            
            # Update states
            obs = next_obs
            z_curr = z_next_pred # CLOSED LOOP DREAMING! We feed prediction back to model
            # Note: We do NOT use the real z from next_obs. That would be cheating (Teacher Forcing).
            # We want to see if the model drifts away.
            
            if terminated or truncated:
                print("Episode finished, resetting...")
                obs, info = env.reset()
                z_curr, context = get_latent_state(vae, obs)
                rnn_hidden = None
                
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    run_dream_simulation()
