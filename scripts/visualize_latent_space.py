"""
Visualize the latent space of the VAE.

This script:
1. Generates random valid arm configurations
2. Encodes them into latent space using a VAE
3. Visualizes the latent space (using PCA if latent_dim > 2)
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

from src.environment.arm import Arm3
from src.models.vae import VAE


def main():
    # Initialize environment
    env = Arm3()
    
    # Generate random valid configurations
    N_SAMPLES = 5000
    print(f"Generating {N_SAMPLES} random valid configurations...")
    
    states = []
    for _ in range(N_SAMPLES):
        # Reset environment to get a random state
        state, _ = env.reset()
        states.append(state)
    
    states = np.array(states)
    print(f"Generated {len(states)} states with shape {states.shape}")
    
    # Load trained VAE
    latent_dim = 4
    vae = VAE(input_dim=7, latent_dim=latent_dim, hidden_dim=128)
    checkpoint_path = ROOT / "src" / "checkpoints" / "vae.pt"
    vae.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    vae.eval()
    print(f"Loaded trained VAE from {checkpoint_path}")

    # Encode states to latent space
    print("Encoding states to latent space...")
    with torch.no_grad():
        states_tensor = torch.FloatTensor(states)
        mu, logvar = vae.encode(states_tensor)
        latent_codes_full = mu.numpy()

    # PCA to project 4D latent → 2D for visualization
    pca = PCA(n_components=2)
    latent_codes = pca.fit_transform(latent_codes_full)
    print(f"Latent codes shape: {latent_codes_full.shape} → PCA → {latent_codes.shape}")
    print(f"PCA explained variance: {pca.explained_variance_ratio_.round(3)}")
    
    # Visualize latent space
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Latent space colored by first joint angle (q1)
    ax1 = axes[0]
    q1_values = states[:, 0]  # First joint angle
    scatter1 = ax1.scatter(latent_codes[:, 0], latent_codes[:, 1], 
                          c=q1_values, cmap='viridis', s=10, alpha=0.6)
    ax1.set_xlabel('Latent Dimension 1', fontsize=12)
    ax1.set_ylabel('Latent Dimension 2', fontsize=12)
    ax1.set_title('Latent Space (colored by q1)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Joint 1 Angle (q1)', fontsize=10)
    
    # Plot 2: Latent space colored by end-effector distance from origin
    ax2 = axes[1]
    # Compute end-effector positions
    distances = []
    for state in states:
        q = state[:3]
        positions = env._joint_positions(q)
        end_effector = positions[-1]
        dist = np.linalg.norm(end_effector)
        distances.append(dist)
    
    distances = np.array(distances)
    scatter2 = ax2.scatter(latent_codes[:, 0], latent_codes[:, 1], 
                          c=distances, cmap='plasma', s=10, alpha=0.6)
    ax2.set_xlabel('Latent Dimension 1', fontsize=12)
    ax2.set_ylabel('Latent Dimension 2', fontsize=12)
    ax2.set_title('Latent Space (colored by end-effector distance)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Distance from Base', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'latent_space_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {output_file}")
    
    plt.show()


if __name__ == "__main__":
    main()
