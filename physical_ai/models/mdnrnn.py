"""Mixture Density Network RNN for world model dynamics prediction."""

import math
import torch
import torch.nn as nn

LATENT_DIM = 8
ACTION_DIM = 3
HIDDEN_SIZE = 256
N_GAUSSIANS = 5


class MDNRNN(nn.Module):
    """LSTM-based world model with Mixture Density Network output.

    Predicts next latent state z_{t+1} given (z_t, a_t) as a mixture of Gaussians.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        action_dim: int = ACTION_DIM,
        hidden_size: int = HIDDEN_SIZE,
        n_gaussians: int = N_GAUSSIANS,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.n_gaussians = n_gaussians

        self.lstm = nn.LSTM(
            input_size=latent_dim + action_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # MDN head: for each Gaussian we output pi (1), mu (latent_dim), sigma (latent_dim)
        mdn_output_size = n_gaussians * (1 + 2 * latent_dim)
        self.mdn_head = nn.Linear(hidden_size, mdn_output_size)

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            z: Latent states, shape (batch, seq_len, latent_dim)
            a: Actions, shape (batch, seq_len, action_dim)
            hidden: Optional (h, c) LSTM hidden state

        Returns:
            (pi, sigma, mu): MDN parameters
            (h, c): New hidden state
        """
        x = torch.cat([z, a], dim=-1)
        lstm_out, hidden_next = self.lstm(x, hidden)

        mdn_out = self.mdn_head(lstm_out)
        pi, sigma, mu = self._split_mdn_params(mdn_out)

        return (pi, sigma, mu), hidden_next

    def _split_mdn_params(
        self, mdn_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split MDN output into mixing coefficients, sigmas, and mus."""
        n_g = self.n_gaussians
        ld = self.latent_dim

        # Shape: (..., n_gaussians * (1 + 2 * latent_dim))
        pi = mdn_out[..., :n_g]
        sigma = mdn_out[..., n_g : n_g + n_g * ld]
        mu = mdn_out[..., n_g + n_g * ld :]

        pi = torch.softmax(pi, dim=-1)
        sigma = torch.exp(sigma).reshape(*sigma.shape[:-1], n_g, ld)
        mu = mu.reshape(*mu.shape[:-1], n_g, ld)

        return pi, sigma, mu

    def sample(
        self,
        pi: torch.Tensor,
        sigma: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """Sample z_next from the mixture distribution."""
        # Select Gaussian component
        categorical = torch.distributions.Categorical(pi)
        idx = categorical.sample()  # (...,)

        # Gather the selected Gaussian's mu and sigma
        batch_shape = idx.shape
        idx_expanded = idx.unsqueeze(-1).unsqueeze(-1).expand(*batch_shape, 1, self.latent_dim)
        selected_mu = mu.gather(-2, idx_expanded).squeeze(-2)
        selected_sigma = sigma.gather(-2, idx_expanded).squeeze(-2)

        # Sample from selected Gaussian
        eps = torch.randn_like(selected_mu)
        return selected_mu + selected_sigma * eps

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h, c)


def mdn_loss(
    pi: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute negative log-likelihood of target under the mixture model.

    Args:
        pi: Mixing coefficients, shape (..., n_gaussians)
        sigma: Standard deviations, shape (..., n_gaussians, latent_dim)
        mu: Means, shape (..., n_gaussians, latent_dim)
        target: Target z values, shape (..., latent_dim)
    """
    target = target.unsqueeze(-2)  # (..., 1, latent_dim)

    # Log probability of target under each Gaussian
    var = sigma.pow(2)
    log_prob = -0.5 * (
        math.log(2 * math.pi)
        + torch.log(var)
        + (target - mu).pow(2) / var
    )
    log_prob = log_prob.sum(dim=-1)  # Sum over latent dims -> (..., n_gaussians)

    # Log mixture probability
    log_pi = torch.log(pi + 1e-8)
    log_mix_prob = torch.logsumexp(log_pi + log_prob, dim=-1)  # (...,)

    return -log_mix_prob.mean()
