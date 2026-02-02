"""
Dynamics Model: predicts next state from current state + action.

Works directly on joint angles + object position (no VAE needed).
Learns the physics: how joints change with actions, and how grasping works.

Input:  [q1, q2, q3, obj_x, obj_y, a1, a2, a3]  (8D)
Output: [q1', q2', q3', obj_x', obj_y']           (5D)

Goal position is NOT included — it never changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DynamicsModel(nn.Module):

    def __init__(self, state_dim: int = 5, action_dim: int = 3, hidden_dim: int = 256):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Normalization buffers
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        self.register_buffer('action_mean', torch.zeros(action_dim))
        self.register_buffer('action_std', torch.ones(action_dim))
        self.normalization_fitted = False

        # Network predicts state DELTA (next_state - current_state)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def fit_normalization(self, states: torch.Tensor, actions: torch.Tensor, eps: float = 1e-8):
        self.state_mean = states.mean(dim=0)
        self.state_std = states.std(dim=0) + eps
        self.action_mean = actions.mean(dim=0)
        self.action_std = actions.std(dim=0) + eps
        self.normalization_fitted = True

    def normalize_state(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.state_mean) / self.state_std

    def normalize_action(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.action_mean) / self.action_std

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next state.

        Args:
            state: [batch, 5] — q1, q2, q3, obj_x, obj_y
            action: [batch, 3] — dq1, dq2, dq3

        Returns:
            next_state: [batch, 5]
        """
        if self.normalization_fitted:
            s_norm = self.normalize_state(state)
            a_norm = self.normalize_action(action)
        else:
            s_norm = state
            a_norm = action

        x = torch.cat([s_norm, a_norm], dim=-1)
        delta = self.net(x)

        # Predict delta, add to current state (residual connection)
        next_state = state + delta
        return next_state
