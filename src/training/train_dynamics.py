"""
Train the dynamics model on collected transition data.

The model learns: f(state_t, action_t) → state_{t+1}
where state = [q1, q2, q3, obj_x, obj_y] (no goal — it never changes)
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from models.dynamics import DynamicsModel


def train_dynamics(
    data_path="data/transitions.pkl",
    hidden_dim=256,
    batch_size=256,
    epochs=300,
    lr=1e-3,
    save_path="checkpoints/dynamics.pt",
):
    # Load data
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    raw_states = torch.tensor(data["states"], dtype=torch.float32)
    raw_actions = torch.tensor(data["actions"], dtype=torch.float32)
    raw_next_states = torch.tensor(data["next_states"], dtype=torch.float32)

    # Extract dynamic part: [q1, q2, q3, obj_x, obj_y] (drop goal_x, goal_y)
    states = raw_states[:, :5]
    next_states = raw_next_states[:, :5]
    actions = raw_actions

    print(f"Loaded {len(states)} transitions")
    print(f"  States:      {states.shape}  (q1, q2, q3, obj_x, obj_y)")
    print(f"  Actions:     {actions.shape}  (dq1, dq2, dq3)")
    print(f"  Next states: {next_states.shape}")

    # Check how many transitions have grasping (obj_pos changed)
    obj_moved = (states[:, 3:5] - next_states[:, 3:5]).abs().sum(dim=1) > 0.001
    print(f"  Transitions with obj movement (grasping): {obj_moved.sum().item()} ({100*obj_moved.float().mean():.1f}%)")

    # Create model
    model = DynamicsModel(state_dim=5, action_dim=3, hidden_dim=hidden_dim)
    model.fit_normalization(states, actions)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    # Split train/val (90/10)
    n = len(states)
    perm = torch.randperm(n)
    n_train = int(0.9 * n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_dataset = TensorDataset(states[train_idx], actions[train_idx], next_states[train_idx])
    val_dataset = TensorDataset(states[val_idx], actions[val_idx], next_states[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"\nTraining dynamics: {epochs} epochs, hidden={hidden_dim}")
    print(f"  Train: {n_train}, Val: {n - n_train}")
    print("-" * 60)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for s, a, s_next in train_loader:
            optimizer.zero_grad()
            pred = model(s, a)
            loss = torch.nn.functional.mse_loss(pred, s_next)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1

        train_loss = train_loss_sum / n_batches

        # Validate
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for s, a, s_next in val_loader:
                pred = model(s, a)
                loss = torch.nn.functional.mse_loss(pred, s_next)
                val_loss_sum += loss.item()
                n_val += 1

        val_loss = val_loss_sum / n_val
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 25 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{epochs}  |  "
                f"train: {train_loss:.6f}  val: {val_loss:.6f}  "
                f"best: {best_val_loss:.6f}  lr: {lr_now:.1e}"
            )

    # Load best model and test
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

    print(f"\nModel saved to {save_path} (best val loss: {best_val_loss:.6f})")

    # Test on random samples
    idx = torch.randperm(len(states))[:5]
    with torch.no_grad():
        s = states[idx]
        a = actions[idx]
        s_next_true = next_states[idx]
        s_next_pred = model(s, a)

        print("\n=== Prediction test (5 random transitions) ===")
        labels = ["q1", "q2", "q3", "ox", "oy"]
        for i in range(5):
            true = s_next_true[i].numpy()
            pred = s_next_pred[i].numpy()
            err = np.abs(true - pred)
            print(f"  True:  {' '.join(f'{labels[j]}={true[j]:+.3f}' for j in range(5))}")
            print(f"  Pred:  {' '.join(f'{labels[j]}={pred[j]:+.3f}' for j in range(5))}")
            print(f"  Error: {' '.join(f'{labels[j]}={err[j]:.4f}' for j in range(5))}")
            print()


if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    train_dynamics()
