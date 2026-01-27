# Great visual, worked as expected

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from src.environment.arm import Arm3
import os

def main():
    # Initialize the environment
    env = Arm3()
    
    # Number of samples for the dense cloud
    N_SAMPLES = 100000
    
    print(f"Generating {N_SAMPLES} random configurations...")
    
    # Sample random joint angles uniformly within limits
    # Shape: (N_SAMPLES, 3)
    q_samples = np.random.uniform(
        low=env.joint_low, 
        high=env.joint_high, 
        size=(N_SAMPLES, env.n_joints)
    )
    
    valid_end_effectors = []
    
    print("Computing forward kinematics and filtering collisions...")
    for q in q_samples:
        if env.is_valid_q(q):
            # _joint_positions returns [base, joint1, joint2, end_effector]
            # We want the last one: end_effector (x, y)
            positions = env._joint_positions(q)
            end_effector = positions[-1]
            valid_end_effectors.append(end_effector)
            
    valid_end_effectors = np.array(valid_end_effectors)
    
    print(f"Found {len(valid_end_effectors)} valid configurations out of {N_SAMPLES}.")
    
    if len(valid_end_effectors) == 0:
        print("No valid configurations found. Check constraints.")
        return

    # Extract X and Y coordinates
    x = valid_end_effectors[:, 0]
    y = valid_end_effectors[:, 1]
    
    # Plotting
    print("Plotting...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Calculate distance from center for coloring
    distances = np.sqrt(x**2 + y**2)
    
    # Scatter plot
    scatter = ax.scatter(x, y, c=distances, cmap='viridis', s=0.5, alpha=0.6)
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Distance from Base', color='white')
    
    # Aesthetics
    ax.set_title(f'Reachable Workspace of Arm3 ({len(valid_end_effectors)} points)', color='white', fontsize=16)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Save the figure
    output_file = 'reachable_workspace.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {os.path.abspath(output_file)}")
    
    # Clean up
    plt.close(fig)

if __name__ == "__main__":
    main()
