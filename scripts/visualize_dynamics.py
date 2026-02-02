"""
Visualize dynamics model accuracy.

Scatter plots: predicted vs real for each state dimension.
Perfect model = all points on the diagonal.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle

from src.models.dynamics import DynamicsModel


def main():
    # Load dynamics model
    model = DynamicsModel(state_dim=5, action_dim=3, hidden_dim=256)
    model.load_state_dict(torch.load(ROOT / "src" / "checkpoints" / "dynamics.pt", weights_only=True))
    model.eval()

    # Load data
    with open(ROOT / "src" / "data" / "transitions.pkl", "rb") as f:
        data = pickle.load(f)

    states = torch.tensor(data["states"][:, :5], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.float32)
    next_states = torch.tensor(data["next_states"][:, :5], dtype=torch.float32)

    # Predict on a random subset
    idx = torch.randperm(len(states))[:5000]
    with torch.no_grad():
        pred = model(states[idx], actions[idx])

    true = next_states[idx].numpy()
    pred = pred.numpy()

    # Plot
    labels = ["q1", "q2", "q3", "obj_x", "obj_y"]
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    for i, (ax, label) in enumerate(zip(axes, labels)):
        errors = np.abs(true[:, i] - pred[:, i])
        sc = ax.scatter(true[:, i], pred[:, i], c=errors, cmap='plasma',
                       s=3, alpha=0.5, vmin=0, vmax=np.percentile(errors, 95))

        # Diagonal (perfect prediction)
        lims = [min(true[:, i].min(), pred[:, i].min()),
                max(true[:, i].max(), pred[:, i].max())]
        ax.plot(lims, lims, 'w--', linewidth=1, alpha=0.5)

        ax.set_xlabel(f'Real {label}')
        ax.set_ylabel(f'Predicted {label}')
        ax.set_title(f'{label}  (err: {errors.mean():.4f})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    plt.colorbar(sc, ax=axes[-1], label='Abs error', fraction=0.05)
    plt.suptitle('Dynamics Model — Predicted vs Real (5000 samples)', fontsize=14)
    plt.tight_layout()

    output_file = ROOT / "scripts" / "dynamics_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_file}")
    plt.show()


if __name__ == "__main__":
    main()
