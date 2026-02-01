"""
Variational Autoencoder (VAE) for compressing robotic arm states.

The VAE learns to encode environment states (joint angles, object position, goal position)
into a compact latent representation and reconstruct them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAE(nn.Module):
    """
    Simple Variational Autoencoder for state compression.
    
    Args:
        input_dim: Dimension of the input state (e.g., 7 for [q1, q2, q3, obj_x, obj_y, goal_x, goal_y])
        latent_dim: Dimension of the latent space (e.g., 4 or 8)
        hidden_dim: Dimension of hidden layers (default: 64)
    """
    
    def __init__(self, input_dim: int = 7, latent_dim: int = 4, hidden_dim: int = 64):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Normalization parameters (will be fitted to data)
        # Using register_buffer so they're saved with the model but not trained
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        self.normalization_fitted = False
        
        # Encoder: state -> hidden -> (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Latent distribution parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent -> hidden -> state
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def fit_normalization(self, data: torch.Tensor, eps: float = 1e-8):
        """
        Fit normalization parameters (mean and std) to the data.
        
        Args:
            data: Training data tensor of shape (N, input_dim)
            eps: Small constant for numerical stability
        """
        self.input_mean = data.mean(dim=0)
        self.input_std = data.std(dim=0) + eps
        self.normalization_fitted = True
        print(f"Normalization fitted: mean={self.input_mean.numpy()}, std={self.input_std.numpy()}")
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using fitted statistics."""
        return (x - self.input_mean) / self.input_std
    
    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize output back to original scale."""
        return x_norm * self.input_std + self.input_mean
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input state to latent distribution parameters.
        
        Args:
            x: Input state tensor of shape (batch_size, input_dim)
            
        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        # Normalize input if normalization is fitted
        if self.normalization_fitted:
            x = self.normalize(x)
        
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstructed state.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            x_recon: Reconstructed state (batch_size, input_dim)
        """
        x_recon = self.decoder(z)
        
        # Denormalize output if normalization is fitted
        if self.normalization_fitted:
            x_recon = self.denormalize(x_recon)
        
        return x_recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input state tensor (batch_size, input_dim)
            
        Returns:
            x_recon: Reconstructed state
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Sample random states from the learned latent distribution.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            samples: Generated states (num_samples, input_dim)
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


def vae_loss(x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
             logvar: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss = Reconstruction Loss + Beta * KL Divergence.
    
    Args:
        x_recon: Reconstructed state
        x: Original state
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term (default: 1.0)
        
    Returns:
        total_loss: Total VAE loss
        recon_loss: Reconstruction loss (MSE)
        kl_loss: KL divergence loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x.size(0)  # Average over batch
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def train_step(model: VAE, optimizer: torch.optim.Optimizer, x: torch.Tensor, 
               beta: float = 1.0) -> Tuple[float, float, float]:
    """
    Single training step for the VAE.
    
    Args:
        model: VAE model
        optimizer: Optimizer
        x: Batch of states
        beta: Weight for KL divergence
        
    Returns:
        total_loss: Total loss value
        recon_loss: Reconstruction loss value
        kl_loss: KL divergence value
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    x_recon, mu, logvar = model(x)
    
    # Compute loss
    total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar, beta)
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), recon_loss.item(), kl_loss.item()


if __name__ == "__main__":
    # Simple test
    print("Testing VAE...")
    
    # Create a simple VAE
    vae = VAE(input_dim=7, latent_dim=4, hidden_dim=64)
    print(f"VAE created with {sum(p.numel() for p in vae.parameters())} parameters")
    
    # Generate some test data with different scales (like real arm states)
    batch_size = 100
    # Simulate: [q1, q2, q3, obj_x, obj_y, goal_x, goal_y]
    # q1: [0.5, 2.6], q2,q3: [-0.8, 0.8], positions: [-1.5, 1.5]
    x = torch.cat([
        torch.rand(batch_size, 1) * 2.1 + 0.5,  # q1
        torch.rand(batch_size, 2) * 1.6 - 0.8,  # q2, q3
        torch.rand(batch_size, 4) * 3.0 - 1.5,  # positions
    ], dim=1)
    
    print(f"\n--- Test WITHOUT normalization ---")
    print(f"Input mean: {x.mean(dim=0).numpy()}")
    print(f"Input std: {x.std(dim=0).numpy()}")
    
    x_recon, mu, logvar = vae(x)
    total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)
    print(f"Loss - Total: {total_loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}")
    
    # Now fit normalization
    print(f"\n--- Fitting normalization ---")
    vae.fit_normalization(x)
    
    print(f"\n--- Test WITH normalization ---")
    x_recon_norm, mu_norm, logvar_norm = vae(x)
    total_loss_norm, recon_loss_norm, kl_loss_norm = vae_loss(x_recon_norm, x, mu_norm, logvar_norm)
    print(f"Loss - Total: {total_loss_norm.item():.4f}, Recon: {recon_loss_norm.item():.4f}, KL: {kl_loss_norm.item():.4f}")
    
    # Test sampling
    samples = vae.sample(num_samples=5)
    print(f"\nGenerated samples shape: {samples.shape}")
    print(f"Sample mean: {samples.mean(dim=0).detach().numpy()}")
    
    print("\n✓ VAE test passed!")

