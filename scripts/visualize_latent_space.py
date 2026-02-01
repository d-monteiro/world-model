"""
Visualize the latent space of the trained VAE.

The VAE compresses joint angles (q1, q2, q3) → 2D latent space.
No PCA needed since latent is already 2D.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.environment.arm import Arm3
from src.models.vae import VAE


def main():
    env = Arm3()

    # Generate random valid configurations
    N_SAMPLES = 5000
    print(f"Generating {N_SAMPLES} random configurations...")

    states = []
    for _ in range(N_SAMPLES):
        state, _ = env.reset()
        states.append(state)

    states = np.array(states)
    joint_angles = states[:, :3]  # Only q1, q2, q3
    print(f"Joint angles shape: {joint_angles.shape}")

    # Load trained VAE (3D → 2D)
    vae = VAE(input_dim=3, latent_dim=2, hidden_dim=256)
    checkpoint_path = ROOT / "src" / "checkpoints" / "vae.pt"
    vae.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    vae.eval()
    print(f"Loaded VAE from {checkpoint_path}")

    # Encode to 2D latent space (no PCA needed)
    with torch.no_grad():
        mu, _ = vae.encode(torch.FloatTensor(joint_angles))
        latent_codes = mu.numpy()

    print(f"Latent codes shape: {latent_codes.shape}")

    # Compute end-effector distances
    distances = []
    for state in states:
        q = state[:3]
        positions = env._joint_positions(q)
        distances.append(np.linalg.norm(positions[-1]))
    distances = np.array(distances)

    # Plot
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: colored by q1
    sc1 = axes[0].scatter(latent_codes[:, 0], latent_codes[:, 1],
                          c=states[:, 0], cmap='viridis', s=10, alpha=0.6)
    axes[0].set_title('Colored by q1 (joint 1)')
    plt.colorbar(sc1, ax=axes[0], label='q1 (rad)')

    # Plot 2: colored by q2
    sc2 = axes[1].scatter(latent_codes[:, 0], latent_codes[:, 1],
                          c=states[:, 1], cmap='viridis', s=10, alpha=0.6)
    axes[1].set_title('Colored by q2 (joint 2)')
    plt.colorbar(sc2, ax=axes[1], label='q2 (rad)')

    # Plot 3: colored by end-effector distance
    sc3 = axes[2].scatter(latent_codes[:, 0], latent_codes[:, 1],
                          c=distances, cmap='plasma', s=10, alpha=0.6)
    axes[2].set_title('Colored by end-effector distance')
    plt.colorbar(sc3, ax=axes[2], label='Distance from base')

    for ax in axes:
        ax.set_xlabel('Latent dim 1')
        ax.set_ylabel('Latent dim 2')
        ax.grid(True, alpha=0.3)

    plt.suptitle('VAE Latent Space (joint angles 3D → 2D)', fontsize=14)
    plt.tight_layout()

    output_file = ROOT / "scripts" / "latent_space_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved to {output_file}")
    plt.show()


if __name__ == "__main__":
    main()
