"""
Train the MDN-RNN (World Model) using pre-collected data and a trained VAE.

Pipeline in this script:
1. Load raw transitions (obs, action, next_obs).
2. Load VAE model to encode observations into latent space 'z'.
3. Create dataset of sequences (z, context, action) -> next_z.
4. Train MDN-RNN to minimize NLL loss.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Models
from src.models.vae import VAE
from src.models.mdn_rnn import MDNRNN, mdn_loss_function

def train_rnn(
    data_path="src/training/data/transitions.pkl",
    vae_path="src/models/vae_weights.pth",
    save_path="src/models/rnn_weights.pth",
    epochs=50,
    batch_size=64,
    hidden_size=256,
    num_gaussians=5,
    lr=0.001
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    # Need to define full absolute path because we are running from root usually
    if not Path(data_path).is_absolute():
        data_path = ROOT / data_path
    
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    # data['states'] shape: (N, 7) -> [q1, q2, q3, ox, oy, gx, gy]
    # data['actions'] shape: (N, 3)
    # data['next_states'] shape: (N, 7)
    
    raw_states = torch.FloatTensor(data['states']).to(device)
    actions = torch.FloatTensor(data['actions']).to(device)
    raw_next_states = torch.FloatTensor(data['next_states']).to(device)
    
    N = len(raw_states)
    print(f"Loaded {N} transitions.")

    # --- 2. Load & Apply VAE ---
    # We need to split states into Arm(3) and Context(4)
    # Arm: q1, q2, q3 (cols 0-3)
    # Context: obj_x, obj_y, goal_x, goal_y (cols 3-7)
    
    print("Loading VAE...")
    vae = VAE(input_dim=3, latent_dim=2).to(device) # Updated dims per user request
    
    if not Path(vae_path).is_absolute():
        vae_path = ROOT / vae_path
        
    try:
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        print("VAE weights loaded.")
    except FileNotFoundError:
        print(f"ERROR: VAE weights not found at {vae_path}. Train VAE first!")
        return

    vae.eval()
    
    # We also need to fit normalization to the new data to be safe, 
    # OR we assume the loaded VAE already has buffers. 
    # Since buffers are saved in state_dict, we are good.
    # But wait, input_dim=3 means we must only pass the first 3 columns.
    
    arm_states = raw_states[:, :3]
    context = raw_states[:, 3:]     # (N, 4)
    
    arm_next_states = raw_next_states[:, :3]
    
    print("Encoding states to latent space...")
    with torch.no_grad():
        # Encode current states
        # The VAE expects normalized inputs internally if it has normalization layers, 
        # but the encode() method handles that if normalization_fitted is True.
        # BUT, did we save the normalization stats? Yes, register_buffer saves them.
        
        # We want the MEAN (mu) of the encoding as the representation z
        z_mu, _ = vae.encode(arm_states)
        z_next_mu, _ = vae.encode(arm_next_states)
        
    z = z_mu.detach()           # (N, 2)
    z_next = z_next_mu.detach() # (N, 2)
    
    print(f"Latent state (z) shape: {z.shape}")
    
    # --- 3. Prepare RNN Data ---
    # For a simple RNN training regarding single-step transitions:
    # Input: z_t, action_t, context_t
    # Target: z_{t+1}
    # Note: The MDN-RNN expects sequential data (Batch, Seq, Feat).
    # Since our data is disjoint transitions (s, a, s'), sequence length = 1.
    
    # Combine z and context for input? No, the MDNRNN class takes z and action separate usually,
    # but we need to inject context.
    # Let's check MDNRNN definition: forward(z, action, hidden) -> x = cat(z, action)
    # We need to Modify inputs to include context.
    
    # TRICK: treat 'z' input to RNN as 'concat(z, context)'
    rnn_input_z = torch.cat([z, context], dim=1) # (N, 2+4=6)
    
    # Reshape for LSTM: (N, 1, Features)
    rnn_input_z_seq = rnn_input_z.unsqueeze(1)
    actions_seq = actions.unsqueeze(1)        # (N, 1, 3)
    target_seq = z_next.unsqueeze(1)          # (N, 1, 2) - Only predicting arm movement!
    
    dataset = TensorDataset(rnn_input_z_seq, actions_seq, target_seq)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # --- 4. Initialize MDN-RNN ---
    latent_dim = 2 # z is 2D
    context_dim = 4 # obj(2) + goal(2)
    action_dim = 3
    
    # The RNN sees: Z(2) + Context(4) as its "state" input
    # So effectively latent_dim passed to constructor should correspond to input size part 1
    # MDNRNN code: input_size = latent_dim + action_dim
    # We will pass (latent_dim + context_dim) as the first arg.
    
    effective_z_dim = latent_dim + context_dim
    
    print(f"Initializing MDN-RNN with input z_dim={effective_z_dim}, action_dim={action_dim}...")
    
    rnn = MDNRNN(
        latent_dim=effective_z_dim, 
        action_dim=action_dim,
        hidden_size=hidden_size,
        num_gaussians=num_gaussians
    ).to(device)
    
    optimizer = optim.Adam(rnn.parameters(), lr=lr)
    
    # --- 5. Training Loop ---
    print(f"\nStarting RNN training for {epochs} epochs...")
    
    min_loss = float('inf')
    
    for epoch in range(epochs):
        rnn.train()
        total_loss = 0
        
        for b_z, b_act, b_target in train_loader:
            optimizer.zero_grad()
            
            # Forward
            # b_z includes context!
            (pi, mu, sigma), _ = rnn(b_z, b_act)
            
            # Loss
            # CAREFUL: b_target is only 2D (next z), but MDN outputs match input size by default?
            # Let's check MDNRNN: fc_mu -> num_gaussians * latent_dim
            # We initialized it with latent_dim=6 (z+context).
            # So it predicts 6D output (next_z + next_context).
            # But context is constant! We only care about the first 2 dimensions of the output 
            # corresponding to z.
            
            # Actually, we should probably change MDNRNN to allow different input_dim vs output_dim.
            # OR simpler: We pass context as part of 'action' (conditioning), not as state to be predicted.
            pass
            
            # STOP. Better approach:
            # Let's pass Context as part of Action so it's treated as Input-Only.
            # RNN State = z (2D)
            # RNN Input = z (2D) + [Action (3D) + Context (4D)]
            
        # ... logic flow interrupt ...
        # I need to adjust the call to handle the dimension mismatch described above.
        # I will implement the loop with the Quick Fix: Concatenate Context to Action.
        
        pass

    # RESTARTING LOGIC CORRECTLY
    # Define Dimensions
    z_dim = 2
    act_dim_plus_context = 3 + 4 # 7
    
    # Re-init model with these dims
    rnn = MDNRNN(
        latent_dim=z_dim,          # It will predict z_dim outputs
        action_dim=act_dim_plus_context, # It takes this as cond input
        hidden_size=hidden_size,
        num_gaussians=num_gaussians
    ).to(device)
    
    optimizer = optim.Adam(rnn.parameters(), lr=lr)
    
    # Update Dataset
    # z inputs: just z (N, 1, 2)
    rnn_z_in = z.unsqueeze(1)
    
    # action inputs: concat(action, context) -> (N, 1, 7)
    rnn_act_in = torch.cat([actions, context], dim=1).unsqueeze(1)
    
    dataset = TensorDataset(rnn_z_in, rnn_act_in, target_seq)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # NOW LOOP
    for epoch in range(epochs):
        rnn.train()
        epoch_loss = 0
        batches = 0
        
        for b_z, b_act_ctx, b_target in train_loader:
            optimizer.zero_grad()
            
            # Forward
            (pi, mu, sigma), _ = rnn(b_z, b_act_ctx)
            
            # Loss
            loss = mdn_loss_function(pi, mu, sigma, b_target)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        avg_loss = epoch_loss / batches
        
        if avg_loss < min_loss:
            min_loss = avg_loss
            # Save absolute path
            torch.save(rnn.state_dict(), ROOT / save_path)
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
            
    print(f"\nTraining Complete. Best Loss: {min_loss:.4f}")
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_rnn()
