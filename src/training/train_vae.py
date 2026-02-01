"""
Train the VAE on collected transition data.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from models.vae import VAE, vae_loss, train_step


def train_vae(
    data_path="data/transitions.pkl",
    latent_dim=4,
    hidden_dim=128,
    batch_size=256,
    epochs=200,
    lr=1e-3,
    beta=0.001,
    save_path="checkpoints/vae.pt",
):
    # Load data
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    states = torch.tensor(data["states"], dtype=torch.float32)
    print(f"Loaded {len(states)} states, shape: {states.shape}")

    # Create model
    vae = VAE(input_dim=states.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim)
    vae.fit_normalization(states)

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # DataLoader
    dataset = TensorDataset(states)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    print(f"\nTraining VAE: {epochs} epochs, batch_size={batch_size}, latent_dim={latent_dim}, beta={beta}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        n_batches = 0

        for (batch,) in loader:
            tl, rl, kl = train_step(vae, optimizer, batch, beta=beta)
            total_loss_sum += tl
            recon_loss_sum += rl
            kl_loss_sum += kl
            n_batches += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs}  |  "
                f"loss: {total_loss_sum/n_batches:.4f}  "
                f"recon: {recon_loss_sum/n_batches:.4f}  "
                f"kl: {kl_loss_sum/n_batches:.4f}"
            )

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(vae.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    # Quick test: reconstruct a few states
    vae.eval()
    with torch.no_grad():
        sample = states[:5]
        recon, _, _ = vae(sample)
        print("\n=== Reconstruction test ===")
        for i in range(5):
            print(f"  Original:      {sample[i].numpy().round(3)}")
            print(f"  Reconstructed: {recon[i].numpy().round(3)}")
            print()


if __name__ == "__main__":
    train_vae()
