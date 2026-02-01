"""
Train the VAE on collected transition data.

Only compresses joint angles (q1, q2, q3) — the dynamic part of the state.
Object and goal positions are static per episode and passed directly.
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
    latent_dim=2,
    hidden_dim=256,
    batch_size=256,
    epochs=500,
    lr=1e-3,
    beta_start=0.0,
    beta_end=0.005,
    warmup_epochs=200,
    save_path="checkpoints/vae.pt",
):
    # Load data
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    all_states = torch.tensor(data["states"], dtype=torch.float32)
    print(f"Loaded {len(all_states)} states, shape: {all_states.shape}")

    # Only use joint angles (first 3 dims) — the dynamic part
    joint_angles = all_states[:, :3]
    print(f"Training on joint angles only: {joint_angles.shape}")
    print(f"  q1 range: [{joint_angles[:, 0].min():.2f}, {joint_angles[:, 0].max():.2f}]")
    print(f"  q2 range: [{joint_angles[:, 1].min():.2f}, {joint_angles[:, 1].max():.2f}]")
    print(f"  q3 range: [{joint_angles[:, 2].min():.2f}, {joint_angles[:, 2].max():.2f}]")

    # Create model: 3D input → 2D latent
    vae = VAE(input_dim=3, latent_dim=latent_dim, hidden_dim=hidden_dim)
    vae.fit_normalization(joint_angles)

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # DataLoader
    dataset = TensorDataset(joint_angles)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop with beta warmup
    print(f"\nTraining VAE: 3D → {latent_dim}D, {epochs} epochs, hidden={hidden_dim}")
    print(f"Beta warmup: {beta_start} → {beta_end} over {warmup_epochs} epochs")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        if epoch <= warmup_epochs:
            beta = beta_start + (beta_end - beta_start) * (epoch / warmup_epochs)
        else:
            beta = beta_end

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

        if epoch % 25 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs}  |  "
                f"beta: {beta:.4f}  "
                f"loss: {total_loss_sum/n_batches:.5f}  "
                f"recon: {recon_loss_sum/n_batches:.5f}  "
                f"kl: {kl_loss_sum/n_batches:.4f}"
            )

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(vae.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    # Reconstruction test
    vae.eval()
    idx = torch.randperm(len(joint_angles))[:5]
    with torch.no_grad():
        sample = joint_angles[idx]
        recon, _, _ = vae(sample)
        print("\n=== Reconstruction test (5 random joint configs) ===")
        print(f"{'':>4}  {'q1':>8} {'q2':>8} {'q3':>8}")
        for i in range(5):
            orig = sample[i].numpy()
            rec = recon[i].numpy()
            err = np.abs(orig - rec)
            print(f"Orig: {orig[0]:8.3f} {orig[1]:8.3f} {orig[2]:8.3f}")
            print(f"Rec:  {rec[0]:8.3f} {rec[1]:8.3f} {rec[2]:8.3f}")
            print(f"Err:  {err[0]:8.3f} {err[1]:8.3f} {err[2]:8.3f}")
            print()


if __name__ == "__main__":
    train_vae()
