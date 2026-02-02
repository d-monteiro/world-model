"""Step 2: Train the VAE on collected state data."""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from physical_ai.models.vae import StateVAE, vae_loss
from physical_ai.utils.preprocessing import fit_scaler, transform, save_scaler

# Hyperparameters
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 150
BETA = 0.01
LATENT_DIM = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load data
    data = np.load(os.path.join(DATA_DIR, "vae_dataset.npy"))
    print(f"Loaded data: {data.shape}")

    # Fit scaler and normalize
    scaler = fit_scaler(data)
    save_scaler(scaler, os.path.join(CHECKPOINT_DIR, "scaler.pkl"))
    data_norm = transform(scaler, data).astype(np.float32)

    # Train/val split
    n_val = int(len(data_norm) * 0.1)
    val_data = data_norm[:n_val]
    train_data = data_norm[n_val:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_data)),
        batch_size=BATCH_SIZE,
    )

    # Model
    model = StateVAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # Train
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            x_recon, mu, logvar = model(batch)
            loss, recon, kl = vae_loss(x_recon, batch, mu, logvar, beta=BETA)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        train_loss = epoch_loss / len(train_data)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(DEVICE)
                x_recon, mu, logvar = model(batch)
                loss, recon, kl = vae_loss(x_recon, batch, mu, logvar, beta=BETA)
                val_loss += loss.item() * len(batch)
                val_recon += recon.item() * len(batch)

        val_loss /= len(val_data)
        val_recon /= len(val_data)
        val_losses.append(val_loss)
        scheduler.step(val_recon)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Recon MSE: {val_recon:.6f}"
            )

    # Save model
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "vae.pth"))
    print(f"Saved VAE to {CHECKPOINT_DIR}/vae.pth")

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training")
    plt.legend()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "vae_training.png"), dpi=100)
    plt.close()
    print("Saved training plot.")


if __name__ == "__main__":
    main()
