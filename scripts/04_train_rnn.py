"""Step 4: Train the MDN-RNN world model with scheduled sampling."""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from physical_ai.models.mdnrnn import MDNRNN, mdn_loss, LATENT_DIM, ACTION_DIM, HIDDEN_SIZE

# Hyperparameters
SEQ_LEN = 50
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 100
GRAD_CLIP = 1.0
# Scheduled sampling: probability of using own prediction instead of ground truth
# Linearly increases from 0 to SS_MAX over training
SS_MAX = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")


def create_subsequences(latents: np.ndarray, actions: np.ndarray, seq_len: int):
    """Create overlapping subsequences from episodes."""
    n_episodes, ep_len, _ = latents.shape
    seqs_z, seqs_a, seqs_target = [], [], []

    for i in range(n_episodes):
        for start in range(0, ep_len - seq_len, seq_len // 2):
            end = start + seq_len
            seqs_z.append(latents[i, start:end])
            seqs_a.append(actions[i, start:end])
            seqs_target.append(latents[i, start + 1 : end + 1])

    # Handle last target that goes beyond episode
    valid = []
    for idx, t in enumerate(seqs_target):
        if len(t) == seq_len:
            valid.append(idx)

    seqs_z = np.array([seqs_z[i] for i in valid])
    seqs_a = np.array([seqs_a[i] for i in valid])
    seqs_target = np.array([seqs_target[i] for i in valid])

    return seqs_z, seqs_a, seqs_target


def train_step_scheduled(model, z_batch, a_batch, target_batch, ss_prob):
    """Training step with scheduled sampling.

    With probability ss_prob, replace ground truth z_t with model's own
    prediction from the previous step. This makes the model robust to
    its own errors during free-running inference.
    """
    batch_size, seq_len, latent_dim = z_batch.shape

    # Step-by-step forward with scheduled sampling
    hidden = model.init_hidden(batch_size, z_batch.device)
    all_pi, all_sigma, all_mu = [], [], []

    z_t = z_batch[:, 0:1, :]  # Start with ground truth first step

    for t in range(seq_len):
        a_t = a_batch[:, t:t+1, :]
        (pi, sigma, mu), hidden = model(z_t, a_t, hidden)
        all_pi.append(pi)
        all_sigma.append(sigma)
        all_mu.append(mu)

        # Detach hidden state to avoid backprop through full sequence at once
        hidden = (hidden[0].detach(), hidden[1].detach())

        # Decide next input: ground truth or own prediction
        if t < seq_len - 1:
            use_own = torch.rand(1).item() < ss_prob
            if use_own:
                # Use deterministic prediction (mean of best Gaussian)
                with torch.no_grad():
                    best_idx = pi.squeeze(1).argmax(dim=-1)
                    best_idx_exp = best_idx.unsqueeze(-1).unsqueeze(-1).expand(
                        batch_size, 1, latent_dim
                    )
                    z_pred = mu.squeeze(1).gather(-2, best_idx_exp).squeeze(-2)
                z_t = z_pred.unsqueeze(1)
            else:
                z_t = z_batch[:, t+1:t+2, :]

    # Stack predictions and compute loss
    all_pi = torch.cat(all_pi, dim=1)
    all_sigma = torch.cat(all_sigma, dim=1)
    all_mu = torch.cat(all_mu, dim=1)

    loss = mdn_loss(all_pi, all_sigma, all_mu, target_batch)
    return loss


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load data
    latents = np.load(os.path.join(DATA_DIR, "rnn_latents.npy"))
    actions = np.load(os.path.join(DATA_DIR, "rnn_actions.npy"))
    print(f"Loaded: latents {latents.shape}, actions {actions.shape}")

    # Create subsequences
    z_seqs, a_seqs, target_seqs = create_subsequences(latents, actions, SEQ_LEN)
    print(f"Created {len(z_seqs)} subsequences of length {SEQ_LEN}")

    # Train/val split
    n_val = int(len(z_seqs) * 0.1)
    indices = np.random.RandomState(42).permutation(len(z_seqs))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(z_seqs[train_idx]).float(),
            torch.from_numpy(a_seqs[train_idx]).float(),
            torch.from_numpy(target_seqs[train_idx]).float(),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(z_seqs[val_idx]).float(),
            torch.from_numpy(a_seqs[val_idx]).float(),
            torch.from_numpy(target_seqs[val_idx]).float(),
        ),
        batch_size=BATCH_SIZE,
    )

    # Model
    model = MDNRNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # Scheduled sampling probability: linearly increase from 0 to SS_MAX
        ss_prob = SS_MAX * epoch / max(EPOCHS - 1, 1)

        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for z_batch, a_batch, target_batch in train_loader:
            z_batch = z_batch.to(DEVICE)
            a_batch = a_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)

            loss = train_step_scheduled(model, z_batch, a_batch, target_batch, ss_prob)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        # Validate (always with teacher forcing for comparable metrics)
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for z_batch, a_batch, target_batch in val_loader:
                z_batch = z_batch.to(DEVICE)
                a_batch = a_batch.to(DEVICE)
                target_batch = target_batch.to(DEVICE)

                (pi, sigma, mu), _ = model(z_batch, a_batch)
                loss = mdn_loss(pi, sigma, mu, target_batch)
                val_loss += loss.item()

                # 1-step MSE
                best_idx = pi.argmax(dim=-1)
                best_idx_exp = best_idx.unsqueeze(-1).unsqueeze(-1).expand(
                    *best_idx.shape, 1, LATENT_DIM
                )
                pred_mu = mu.gather(-2, best_idx_exp).squeeze(-2)
                mse = ((pred_mu - target_batch) ** 2).mean().item()
                val_mse += mse
                n_val_batches += 1

        val_loss /= n_val_batches
        val_mse /= n_val_batches
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Train NLL: {train_loss:.4f} | "
                f"Val NLL: {val_loss:.4f} | "
                f"Val MSE: {val_mse:.6f} | "
                f"SS: {ss_prob:.2f}"
            )

    # Save model
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "mdnrnn.pth"))
    print(f"Saved MDN-RNN to {CHECKPOINT_DIR}/mdnrnn.pth")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train NLL")
    plt.plot(val_losses, label="Val NLL")
    plt.xlabel("Epoch")
    plt.ylabel("NLL Loss")
    plt.title("MDN-RNN Training (with Scheduled Sampling)")
    plt.legend()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "mdnrnn_training.png"), dpi=100)
    plt.close()
    print("Saved training plot.")


if __name__ == "__main__":
    main()
