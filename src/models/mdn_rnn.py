"""
MDN-RNN Model (Mixture Density Network + LSTM).

This model acts as the "Brain" of the World Model.
It predicts the next latent state z_{t+1} given current state z_t and action a_t.
Instead of a single deterministic prediction, it outputs a mixture of Gaussians
to model probability distributions (uncertainty).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MDNRNN(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_size=256, num_gaussians=5, num_layers=1):
        super(MDNRNN, self).__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians
        self.num_layers = num_layers
        
        # LSTM Input: Latent State + Action
        # Note: If passing Object/Goal positions directly, add their dimensions here!
        # input_size = latent_dim + action_dim + context_dim
        self.input_size = latent_dim + action_dim 
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # MDN Heads (output parameters for the Mixture of Gaussians)
        # We predict a distribution for EACH latent dimension INDEPENDENTLY or Jointly?
        # Usually standard MDN-RNN predicts a mixture for the whole vector.
        # Output size per gaussian component:
        # - mu: latent_dim
        # - sigma: latent_dim (diagonal covariance)
        # - pi: 1 (weight of this component)
        
        self.fc_pi = nn.Linear(hidden_size, num_gaussians)            # Logits for mixing coefficients
        self.fc_mu = nn.Linear(hidden_size, num_gaussians * latent_dim)   # Means
        self.fc_sigma = nn.Linear(hidden_size, num_gaussians * latent_dim) # Log std deviations

    def forward(self, z, action, hidden=None):
        """
        Args:
            z: Latent state (batch, seq_len, latent_dim)
            action: Action (batch, seq_len, action_dim)
            hidden: LSTM hidden state (h, c)
            
        Returns:
            (pi, mu, sigma), hidden
        """
        # Concatenate z and action
        x = torch.cat([z, action], dim=-1) # (batch, seq, latent+act)
        
        # LSTM forward
        output, hidden = self.lstm(x, hidden) # output: (batch, seq, hidden)
        
        # Process outputs for MDN
        # Pi: (batch, seq, num_gaussians)
        pi = F.softmax(self.fc_pi(output), dim=-1)
        
        # Mu: (batch, seq, num_gaussians, latent_dim)
        mu = self.fc_mu(output)
        mu = mu.view(output.shape[0], output.shape[1], self.num_gaussians, self.latent_dim)
        
        # Sigma: (batch, seq, num_gaussians, latent_dim)
        # Use exp to ensure positivity
        log_sigma = self.fc_sigma(output)
        log_sigma = log_sigma.view(output.shape[0], output.shape[1], self.num_gaussians, self.latent_dim)
        sigma = torch.exp(log_sigma)
        
        return (pi, mu, sigma), hidden

    def get_initial_state(self, batch_size, device):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        )


def mdn_loss_function(pi, mu, sigma, target):
    """
    Computes the Negative Log Likelihood (NLL) loss for MDN.
    
    Args:
        pi: (batch, seq, K) mixing coefficients
        mu: (batch, seq, K, L) means
        sigma: (batch, seq, K, L) standard deviations
        target: (batch, seq, L) actual next latent states
        
    Returns:
        nll: scalar loss
    """
    # Expand target to match K components
    # target: (B, S, L) -> (B, S, 1, L)
    target = target.unsqueeze(2)
    
    # Calculate probability density of target under each Gaussian component
    # Normal distribution formula: 
    # P(x) = 1/sqrt(2pi*sigma^2) * exp(-(x-mu)^2 / 2sigma^2)
    # Log P(x) = -0.5 * log(2pi) - log(sigma) - 0.5 * ((x-mu)/sigma)^2
    
    # We sum over Latent Dimensions (L) assuming diagonal covariance (independent dims given component)
    # Log prob for each component k: sum(log_prob_dim_l)
    
    var = sigma ** 2
    log_scale = torch.log(sigma)
    
    # (B, S, K, L)
    exponent = -0.5 * ((target - mu) ** 2) / var
    log_prob_per_dim = exponent - log_scale - 0.5 * np.log(2 * np.pi)
    
    # Sum across latent dimensions (L) -> log probability of the full vector for component K
    # (B, S, K)
    log_prob_comp = torch.sum(log_prob_per_dim, dim=-1)
    
    # Add mixing coefficients (in log space for stability)
    # P(y) = sum_k(pi_k * P_k(y))
    # Log Sum Exp trick: log(sum(exp(x)))
    
    # log(pi * N(mu, sigma)) = log(pi) + log_prob_comp
    log_weighted_prob = torch.log(pi + 1e-8) + log_prob_comp
    
    # Sum over components (K) using LogSumExp
    log_prob_final = torch.logsumexp(log_weighted_prob, dim=-1)
    
    # Negative Log Likelihood
    nll = -torch.mean(log_prob_final)
    
    return nll
