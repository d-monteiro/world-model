"""Step 3: Collect episode sequences and encode with trained VAE for RNN training."""

import os
import numpy as np
import torch

from physical_ai.data.collector import collect_episodes
from physical_ai.models.vae import StateVAE, LATENT_DIM
from physical_ai.utils.preprocessing import load_scaler, transform

N_EPISODES = 10_000
EPISODE_LENGTH = 100
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load VAE and scaler
    scaler = load_scaler(os.path.join(CHECKPOINT_DIR, "scaler.pkl"))
    vae = StateVAE(latent_dim=LATENT_DIM).to(DEVICE)
    vae.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "vae.pth"), map_location=DEVICE, weights_only=True))
    vae.eval()

    print(f"Collecting {N_EPISODES} episodes of {EPISODE_LENGTH} steps...")
    observations, actions = collect_episodes(
        n_episodes=N_EPISODES,
        episode_length=EPISODE_LENGTH,
        seed=SEED,
    )
    print(f"Observations: {observations.shape}, Actions: {actions.shape}")

    # Encode observations to latent space
    print("Encoding observations with VAE...")
    obs_flat = observations.reshape(-1, 9)
    obs_norm = transform(scaler, obs_flat).astype(np.float32)

    latents = []
    batch_size = 4096
    with torch.no_grad():
        for i in range(0, len(obs_norm), batch_size):
            batch = torch.from_numpy(obs_norm[i : i + batch_size]).to(DEVICE)
            mu, _ = vae.encode(batch)
            latents.append(mu.cpu().numpy())

    latents = np.concatenate(latents, axis=0)
    latents = latents.reshape(N_EPISODES, EPISODE_LENGTH, LATENT_DIM)

    # Save sequences: z_t, a_t for each timestep
    np.save(os.path.join(DATA_DIR, "rnn_latents.npy"), latents)
    np.save(os.path.join(DATA_DIR, "rnn_actions.npy"), actions.astype(np.float32))
    print(f"Saved RNN data: latents {latents.shape}, actions {actions.shape}")


if __name__ == "__main__":
    main()
