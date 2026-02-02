"""
Model Predictive Control (MPC) using the learned World Model.

This script implements a 'Random Shooting' planner:
1. Generate K random action sequences.
2. Predict trajectories using the RNN (World Model).
3. Evaluate trajectories based on a cost function (Distance to Goal).
4. Execute the best action.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.vae import VAE
from src.models.mdn_rnn import MDNRNN

class MPCPlanner:
    def __init__(
        self, 
        vae_path="src/models/vae_weights.pth",
        rnn_path="src/models/rnn_weights.pth",
        horizon=10, 
        n_candidates=1000,
        device='cpu'
    ):
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.device = device
        
        # Load Models
        if not Path(vae_path).is_absolute():
            vae_path = ROOT / vae_path
        if not Path(rnn_path).is_absolute():
            rnn_path = ROOT / rnn_path
            
        print(f"Loading MPC models from:\n VAE: {vae_path}\n RNN: {rnn_path}")
        
        # Load VAE
        self.vae = VAE(input_dim=3, latent_dim=2).to(device)
        try:
            self.vae.load_state_dict(torch.load(vae_path, map_location=device))
        except FileNotFoundError:
            print("❌ Error: VAE weights not found.")
        self.vae.eval()
        
        # Load RNN
        # Context(4) + Action(3) = 7 dims
        self.rnn = MDNRNN(latent_dim=2, action_dim=7, hidden_size=256).to(device)
        try:
            self.rnn.load_state_dict(torch.load(rnn_path, map_location=device))
        except FileNotFoundError:
            print("❌ Error: RNN weights not found.")
        self.rnn.eval()
        
    def encode_obs(self, obs):
        """Encodes observation: [q, qs, qe, ox, oy, gx, gy] -> z (2D), context (4D)"""
        # obs shape needs to be (1, 7) or (7,)
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            
        arm_state = obs[:, :3]
        context = obs[:, 3:]
        
        with torch.no_grad():
            mu, _ = self.vae.encode(arm_state)
        return mu, context

    def forward_kinematics(self, joints):
        """
        Compute End-Effector (x,y) from joint angles (batch format).
        Joints: (Batch, Horizon, 3) or (Batch, 3)
        Returns: (Batch, Horizon, 2)
        """
        # Link lengths from arm.py (0.5, 0.5, 0.5)
        l1, l2, l3 = 0.5, 0.5, 0.5
        
        q1 = joints[..., 0]
        q2 = joints[..., 1]
        q3 = joints[..., 2]
        
        # Relative angles -> Absolute angles
        # In arm.py logic, the angles are usually relative/accumulated? 
        # Let's check arm.py _joint_positions:
        # angle_sum += angle
        # x += length * cos(angle_sum)
        
        a1 = q1
        x1 = l1 * torch.cos(a1)
        y1 = l1 * torch.sin(a1)
        
        a2 = a1 + q2
        x2 = x1 + l2 * torch.cos(a2)
        y2 = y1 + l2 * torch.sin(a2)
        
        a3 = a2 + q3
        x3 = x2 + l3 * torch.cos(a3)
        y3 = y2 + l3 * torch.sin(a3)
        
        return torch.stack([x3, y3], dim=-1)

    def plan(self, obs, target_pos=(1.0, 1.0)):
        """
        Plan the best action sequence to reach target_pos.
        obs: Current observation
        target_pos: Tuple (x, y) we want the hand to reach
        """
        z_curr, context = self.encode_obs(obs) # z: (1, 2), ctx: (1, 4)
        
        # 1. Generate Candidates
        # Shape: (N_candidates, Horizon, 3)
        # Action space is roughly [-0.05, 0.05]
        actions = torch.rand(self.n_candidates, self.horizon, 3).to(self.device) * 0.1 - 0.05
        
        # Expand context for batch
        # context: (1, 4) -> (N, Horizon, 4)
        batch_ctx = context.unsqueeze(1).expand(self.n_candidates, self.horizon, 4)
        
        # Concatenate actions + context -> RNN input
        # rnn_in: (N, H, 7)
        rnn_in_act = torch.cat([actions, batch_ctx], dim=-1)
        
        # 2. Rollout latent trajectories
        # We need to loop over horizon because RNN state updates
        curr_z = z_curr.expand(self.n_candidates, 1, 2) # (N, 1, 2)
        hidden = None
        
        traj_z = []
        
        with torch.no_grad():
            for t in range(self.horizon):
                step_act = rnn_in_act[:, t:t+1, :] # (N, 1, 7)
                
                (pi, mu, sigma), hidden = self.rnn(curr_z, step_act, hidden)
                
                # Deterministic prediction: Pick mean of strongest component
                # Simplified: Just pick first component or weighted mean?
                # Let's pick the component with max weight for each batch item
                # pi: (N, 1, K)
                k = torch.argmax(pi, dim=-1) # (N, 1) indices
                
                # Gather mu corresponding to k
                # mu: (N, 1, K, L)
                # This gather is tricky in pytorch without complex indexing
                # Let's just take the first component for speed/simplicity as they often collapse
                # OR: take weighted average
                
                # mu: (N, 1, K, 2)
                # pi: (N, 1, K) -> unsqueeze -> (N, 1, K, 1)
                weighted_mu = torch.sum(mu * pi.unsqueeze(-1), dim=2) # (N, 1, 2)
                
                curr_z = weighted_mu
                traj_z.append(curr_z)
                
        # Stack trajectory: (N, H, 2)
        full_traj_z = torch.cat(traj_z, dim=1)
        
        # 3. Decode to Joint Space to check Real Physics cost
        # We only care about the Final State (Reach task) or Cumulative?
        # Let's check Final State distance
        final_z = full_traj_z[:, -1, :] # (N, 2)
        
        with torch.no_grad():
            final_joints = self.vae.decode(final_z) # (N, 3)
            
        # 4. Compute Cost
        # Forward Kinematics
        end_effector_pos = self.forward_kinematics(final_joints) # (N, 2)
        
        target = torch.tensor(target_pos).to(self.device).expand(self.n_candidates, 2)
        
        # Euclidean distance
        dists = torch.norm(end_effector_pos - target, dim=1)
        
        # 5. Pick Best
        best_idx = torch.argmin(dists)
        best_action_seq = actions[best_idx]
        
        # Return first action of best sequence
        return best_action_seq[0].cpu().numpy()

if __name__ == "__main__":
    # Simple Test
    planner = MPCPlanner()
    print("Planner initialized. Testing random plan...")
    
    dummy_obs = np.array([0.0, 0.0, 0.0, 0.5, 0.5, -0.5, -0.5], dtype=np.float32)
    action = planner.plan(dummy_obs, target_pos=(0.5, 0.5))
    print(f"Selected Action: {action}")
