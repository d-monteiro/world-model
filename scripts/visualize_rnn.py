"""Visualize MDN-RNN predictions: 1-step and multi-step rollouts."""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from physical_ai.models.vae import StateVAE, LATENT_DIM
from physical_ai.models.mdnrnn import MDNRNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Load models
vae = StateVAE(latent_dim=LATENT_DIM).to(DEVICE)
vae.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "vae.pth"), map_location=DEVICE, weights_only=True))
vae.eval()

mdnrnn = MDNRNN().to(DEVICE)
mdnrnn.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "mdnrnn.pth"), map_location=DEVICE, weights_only=True))
mdnrnn.eval()

# Load a few episodes
latents = np.load(os.path.join(DATA_DIR, "rnn_latents.npy"))
actions = np.load(os.path.join(DATA_DIR, "rnn_actions.npy"))

fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# --- Row 1: 1-step prediction for 3 random episodes ---
rng = np.random.RandomState(42)
episodes = rng.choice(len(latents), 3, replace=False)

for col, ep_idx in enumerate(episodes):
    z_seq = torch.from_numpy(latents[ep_idx]).float().unsqueeze(0).to(DEVICE)
    a_seq = torch.from_numpy(actions[ep_idx]).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        (pi, sigma, mu), _ = mdnrnn(z_seq[:, :-1], a_seq[:, :-1])
        best_idx = pi.argmax(dim=-1)
        best_idx_exp = best_idx.unsqueeze(-1).unsqueeze(-1).expand(*best_idx.shape, 1, LATENT_DIM)
        pred = mu.gather(-2, best_idx_exp).squeeze(-2)

    actual = latents[ep_idx, 1:]
    predicted = pred[0].cpu().numpy()

    mse_per_step = ((actual - predicted) ** 2).mean(axis=1)
    axes[0, col].plot(mse_per_step)
    axes[0, col].set_title(f"Episode {ep_idx} — 1-step MSE")
    axes[0, col].set_xlabel("Step")
    axes[0, col].set_ylabel("MSE")
    axes[0, col].set_ylim(0, max(0.05, mse_per_step.max() * 1.1))


def rollout_multistep(ep_idx, deterministic):
    """Run multi-step free-running rollout."""
    z_seq = torch.from_numpy(latents[ep_idx]).float().unsqueeze(0).to(DEVICE)
    a_seq = torch.from_numpy(actions[ep_idx]).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        z_t = z_seq[:, 0:1]
        hidden = mdnrnn.init_hidden(1, DEVICE)
        predicted_steps = []

        for t in range(99):
            a_t = a_seq[:, t:t+1]
            (pi, sigma, mu), hidden = mdnrnn(z_t, a_t, hidden)

            if deterministic:
                # Mean of best Gaussian (what controller sees in dreams)
                pi_flat = pi.squeeze(1)
                mu_flat = mu.squeeze(1)
                best_idx = pi_flat.argmax(dim=-1)
                best_idx_exp = best_idx.unsqueeze(-1).unsqueeze(-1).expand(1, 1, LATENT_DIM)
                z_next = mu_flat.gather(-2, best_idx_exp).squeeze(-2).unsqueeze(1)
            else:
                # Stochastic sampling
                z_next = mdnrnn.sample(pi.squeeze(1), sigma.squeeze(1), mu.squeeze(1)).unsqueeze(1)

            predicted_steps.append(z_next[0, 0].cpu().numpy())
            z_t = z_next

    actual = latents[ep_idx, 1:]
    predicted = np.array(predicted_steps)

    with torch.no_grad():
        actual_states = vae.decode(torch.from_numpy(actual).float().to(DEVICE)).cpu().numpy()
        pred_states = vae.decode(torch.from_numpy(predicted).float().to(DEVICE)).cpu().numpy()

    return actual_states, pred_states


# --- Row 2: Multi-step DETERMINISTIC (what controller sees) ---
for col, ep_idx in enumerate(episodes):
    actual_states, pred_states = rollout_multistep(ep_idx, deterministic=True)
    axes[1, col].plot(actual_states[:, 3], label="Actual obj_x", linewidth=2)
    axes[1, col].plot(pred_states[:, 3], label="Dream obj_x", linestyle="--", linewidth=2)
    axes[1, col].plot(actual_states[:, 6], label="Actual tgt_x", linewidth=1, alpha=0.5)
    axes[1, col].plot(pred_states[:, 6], label="Dream tgt_x", linestyle=":", linewidth=1, alpha=0.5)
    axes[1, col].set_title(f"Episode {ep_idx} — Deterministic rollout")
    axes[1, col].set_xlabel("Step")
    axes[1, col].set_ylabel("Position (normalized)")
    axes[1, col].legend(fontsize=8)

# --- Row 3: Multi-step STOCHASTIC (for comparison) ---
for col, ep_idx in enumerate(episodes):
    actual_states, pred_states = rollout_multistep(ep_idx, deterministic=False)
    axes[2, col].plot(actual_states[:, 3], label="Actual obj_x", linewidth=2)
    axes[2, col].plot(pred_states[:, 3], label="Dream obj_x", linestyle="--", linewidth=2)
    axes[2, col].plot(actual_states[:, 6], label="Actual tgt_x", linewidth=1, alpha=0.5)
    axes[2, col].plot(pred_states[:, 6], label="Dream tgt_x", linestyle=":", linewidth=1, alpha=0.5)
    axes[2, col].set_title(f"Episode {ep_idx} — Stochastic rollout")
    axes[2, col].set_xlabel("Step")
    axes[2, col].set_ylabel("Position (normalized)")
    axes[2, col].legend(fontsize=8)

plt.tight_layout()
output_path = os.path.join(CHECKPOINT_DIR, "rnn_predictions.png")
plt.savefig(output_path, dpi=150)
plt.close()
print(f"Saved to {output_path}")

# Summary stats
print(f"\n1-step MSE across 100 random episodes:")
mses = []
for ep_idx in rng.choice(len(latents), 100, replace=False):
    z_seq = torch.from_numpy(latents[ep_idx]).float().unsqueeze(0).to(DEVICE)
    a_seq = torch.from_numpy(actions[ep_idx]).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        (pi, sigma, mu), _ = mdnrnn(z_seq[:, :-1], a_seq[:, :-1])
        best_idx = pi.argmax(dim=-1)
        best_idx_exp = best_idx.unsqueeze(-1).unsqueeze(-1).expand(*best_idx.shape, 1, LATENT_DIM)
        pred = mu.gather(-2, best_idx_exp).squeeze(-2)
    mse = ((latents[ep_idx, 1:] - pred[0].cpu().numpy()) ** 2).mean()
    mses.append(mse)
print(f"  Mean: {np.mean(mses):.6f}")
print(f"  Std:  {np.std(mses):.6f}")
