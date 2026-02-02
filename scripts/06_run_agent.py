"""Step 6: Run the trained agent in the real MuJoCo environment."""

import os
import argparse
import numpy as np
import torch
import gymnasium

import physical_ai.envs  # noqa: F401
from physical_ai.models.vae import StateVAE, LATENT_DIM
from physical_ai.models.mdnrnn import MDNRNN
from physical_ai.controller.controller import Controller
from physical_ai.utils.preprocessing import load_scaler, transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")


def run_episode(env, vae, mdnrnn, controller, scaler, render: bool = False):
    """Run a single episode with the trained agent."""
    obs, _ = env.reset()

    hidden = mdnrnn.init_hidden(1, DEVICE)
    total_reward = 0.0
    steps = 0

    for step in range(200):
        # Encode observation
        obs_norm = transform(scaler, obs.reshape(1, -1)).astype(np.float32)
        obs_tensor = torch.from_numpy(obs_norm).to(DEVICE)

        with torch.no_grad():
            mu, _ = vae.encode(obs_tensor)
            z = mu  # Use mean, no sampling

            h = hidden[0].squeeze(0)  # (1, hidden_size)
            action = controller(z, h)  # (1, action_dim)

            # Update MDN-RNN hidden state
            z_seq = z.unsqueeze(1)  # (1, 1, latent_dim)
            a_seq = action.unsqueeze(1)  # (1, 1, action_dim)
            _, hidden = mdnrnn(z_seq, a_seq, hidden)

        action_np = action.cpu().numpy().squeeze()
        obs, reward, terminated, truncated, info = env.step(action_np)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    return total_reward, steps, info.get("success", False), info.get("distance", float("inf"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render with MuJoCo viewer")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    args = parser.parse_args()

    # Load models
    scaler = load_scaler(os.path.join(CHECKPOINT_DIR, "scaler.pkl"))

    vae = StateVAE(latent_dim=LATENT_DIM).to(DEVICE)
    vae.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "vae.pth"), map_location=DEVICE, weights_only=True))
    vae.eval()

    mdnrnn = MDNRNN().to(DEVICE)
    mdnrnn.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "mdnrnn.pth"), map_location=DEVICE, weights_only=True))
    mdnrnn.eval()

    controller = Controller().to(DEVICE)
    controller.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "controller.pth"), map_location=DEVICE, weights_only=True))
    controller.eval()

    render_mode = "human" if args.render else None
    env = gymnasium.make("ThreeJointArm-v0", render_mode=render_mode)

    for ep in range(args.episodes):
        reward, steps, success, dist = run_episode(env, vae, mdnrnn, controller, scaler, args.render)
        status = "SUCCESS" if success else "FAIL"
        print(f"Episode {ep+1}: reward={reward:.4f}, steps={steps}, dist={dist:.4f}, {status}")

    env.close()


if __name__ == "__main__":
    main()
