"""Linear controller trained with CMA-ES."""

import numpy as np
import torch
import torch.nn as nn

from physical_ai.models.vae import LATENT_DIM
from physical_ai.models.mdnrnn import HIDDEN_SIZE, ACTION_DIM


class Controller(nn.Module):
    """Simple linear controller: [z, h] -> action.

    Input: concatenation of latent state z (32) and LSTM hidden state h (256) = 288.
    Output: action (3) with tanh activation.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_size: int = HIDDEN_SIZE,
        action_dim: int = ACTION_DIM,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.fc = nn.Linear(latent_dim + hidden_size, action_dim)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, h], dim=-1)
        return torch.tanh(self.fc(x))

    def get_n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_params(self, flat_params: np.ndarray):
        """Set parameters from a flat numpy array (for CMA-ES)."""
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(
                torch.from_numpy(flat_params[offset : offset + n])
                .float()
                .reshape(p.shape)
            )
            offset += n

    def get_params(self) -> np.ndarray:
        """Get parameters as a flat numpy array."""
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.parameters()])


def dream_evaluate(
    controller: Controller,
    vae,
    mdnrnn,
    n_rollouts: int = 16,
    horizon: int = 50,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Evaluate controller inside the MDN-RNN dream.

    Uses deterministic predictions (mean of best Gaussian) to avoid
    compounding stochastic noise. Short horizon (50 steps) keeps dreams
    within the reliable prediction range.

    Returns negative mean reward (CMA-ES minimizes).
    """
    vae.eval()
    mdnrnn.eval()
    controller.eval()

    total_reward = 0.0

    with torch.no_grad():
        for _ in range(n_rollouts):
            # Initialize z from VAE prior (standard normal)
            z = torch.randn(1, 1, vae.latent_dim, device=device)
            hidden = mdnrnn.init_hidden(1, device)

            episode_reward = 0.0
            for _ in range(horizon):
                h = hidden[0].squeeze(0)  # (1, hidden_size)
                z_flat = z.squeeze(1)  # (1, latent_dim)

                action = controller(z_flat, h)  # (1, action_dim)

                # Step in dream world
                action_seq = action.unsqueeze(1)  # (1, 1, action_dim)
                z_input = z  # (1, 1, latent_dim)
                (pi, sigma, mu), hidden = mdnrnn(z_input, action_seq, hidden)

                # Deterministic: use mean of most likely Gaussian (no sampling)
                pi_flat = pi.squeeze(1)  # (1, n_gaussians)
                mu_flat = mu.squeeze(1)  # (1, n_gaussians, latent_dim)
                best_idx = pi_flat.argmax(dim=-1)  # (1,)
                best_idx_exp = best_idx.unsqueeze(-1).unsqueeze(-1).expand(
                    1, 1, vae.latent_dim
                )
                z_next = mu_flat.gather(-2, best_idx_exp).squeeze(-2).unsqueeze(1)

                # Decode z to get state and compute reward
                state = vae.decode(z_next.squeeze(1))  # (1, 9)
                obj_pos = state[0, 3:6]
                target_pos = state[0, 6:9]
                dist = torch.norm(obj_pos - target_pos).item()
                episode_reward += -dist

                z = z_next

            total_reward += episode_reward

    mean_reward = total_reward / n_rollouts
    # CMA-ES minimizes, so return negative reward
    return -mean_reward
