"""Visualize the VAE latent space with PCA and t-SNE."""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from physical_ai.models.vae import StateVAE, LATENT_DIM
from physical_ai.utils.preprocessing import load_scaler, transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Load
scaler = load_scaler(os.path.join(CHECKPOINT_DIR, "scaler.pkl"))
vae = StateVAE(latent_dim=LATENT_DIM).to(DEVICE)
vae.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "vae.pth"), map_location=DEVICE, weights_only=True))
vae.eval()

data = np.load(os.path.join(DATA_DIR, "vae_dataset.npy"))

# Sample 10K points for visualization
rng = np.random.RandomState(42)
idx = rng.choice(len(data), 10000, replace=False)
sample = data[idx]
sample_norm = transform(scaler, sample).astype(np.float32)

# Encode
with torch.no_grad():
    mu, logvar = vae.encode(torch.from_numpy(sample_norm).to(DEVICE))
    z = mu.cpu().numpy()

# Also get reconstructions for error analysis
with torch.no_grad():
    recon, _, _ = vae(torch.from_numpy(sample_norm).to(DEVICE))
    recon = recon.cpu().numpy()

# --- Figure 1: Latent space PCA colored by object-target distance ---
obj_pos = sample[:, 3:6]
target_pos = sample[:, 6:9]
distances = np.linalg.norm(obj_pos - target_pos, axis=1)

pca = PCA(n_components=2)
z_2d = pca.fit_transform(z)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sc = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=distances, cmap="viridis", s=1, alpha=0.5)
axes[0].set_title("Latent space (PCA) — color = obj-target distance")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.colorbar(sc, ax=axes[0], label="Distance")

# --- Figure 2: Reconstruction error per dimension ---
errors = np.abs(sample_norm - recon)
mean_errors = errors.mean(axis=0)
labels = ["j1", "j2", "j3", "ox", "oy", "oz", "tx", "ty", "tz"]
axes[1].bar(labels, mean_errors)
axes[1].set_title("Mean reconstruction error per dimension")
axes[1].set_ylabel("MAE (normalized)")

# --- Figure 3: Latent dimension activity ---
z_std = z.std(axis=0)
axes[2].bar(range(LATENT_DIM), sorted(z_std, reverse=True))
axes[2].set_title("Latent dimension activity (std, sorted)")
axes[2].set_xlabel("Latent dimension")
axes[2].set_ylabel("Std")

plt.tight_layout()
output_path = os.path.join(CHECKPOINT_DIR, "latent_space.png")
plt.savefig(output_path, dpi=150)
plt.close()
print(f"Saved to {output_path}")
print(f"\nLatent dims with std > 0.5: {(z_std > 0.5).sum()} / {LATENT_DIM}")
print(f"PCA explained variance (2 components): {pca.explained_variance_ratio_.sum():.1%}")
