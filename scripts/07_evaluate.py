"""Step 7: Evaluate the trained agent with comprehensive metrics."""

import os
import numpy as np
import torch
import gymnasium

import physical_ai.envs  # noqa: F401
from physical_ai.models.vae import StateVAE, LATENT_DIM
from physical_ai.models.mdnrnn import MDNRNN
from physical_ai.controller.controller import Controller
from physical_ai.utils.preprocessing import load_scaler, transform

N_EPISODES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")


def compute_jerk(actions: list[np.ndarray]) -> float:
    """Compute mean jerk (third derivative of position ~ derivative of acceleration)."""
    if len(actions) < 3:
        return 0.0
    actions = np.array(actions)
    acc = np.diff(actions, axis=0)
    jerk = np.diff(acc, axis=0)
    return float(np.mean(np.abs(jerk)))


def evaluate_episode(env, vae, mdnrnn, controller, scaler):
    """Run one evaluation episode and return metrics."""
    obs, _ = env.reset()
    hidden = mdnrnn.init_hidden(1, DEVICE)

    total_reward = 0.0
    actions_history = []
    steps = 0
    success = False
    success_step = None
    final_dist = float("inf")

    for step in range(200):
        obs_norm = transform(scaler, obs.reshape(1, -1)).astype(np.float32)
        obs_tensor = torch.from_numpy(obs_norm).to(DEVICE)

        with torch.no_grad():
            mu, _ = vae.encode(obs_tensor)
            z = mu
            h = hidden[0].squeeze(0)
            action = controller(z, h)
            z_seq = z.unsqueeze(1)
            a_seq = action.unsqueeze(1)
            _, hidden = mdnrnn(z_seq, a_seq, hidden)

        action_np = action.cpu().numpy().squeeze()
        actions_history.append(action_np)

        obs, reward, terminated, truncated, info = env.step(action_np)
        total_reward += reward
        steps += 1
        final_dist = info.get("distance", float("inf"))

        if terminated:
            success = True
            success_step = step + 1
            break
        if truncated:
            break

    jerk = compute_jerk(actions_history)
    return {
        "reward": total_reward,
        "steps": steps,
        "success": success,
        "success_step": success_step,
        "final_distance": final_dist,
        "jerk": jerk,
    }


def main():
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

    env = gymnasium.make("ThreeJointArm-v0")

    print(f"Evaluating over {N_EPISODES} episodes...\n")

    results = []
    for ep in range(N_EPISODES):
        metrics = evaluate_episode(env, vae, mdnrnn, controller, scaler)
        results.append(metrics)
        if (ep + 1) % 20 == 0:
            print(f"  Completed {ep+1}/{N_EPISODES} episodes")

    env.close()

    # Aggregate metrics
    successes = [r["success"] for r in results]
    final_dists = [r["final_distance"] for r in results]
    jerks = [r["jerk"] for r in results]
    rewards = [r["reward"] for r in results]
    success_steps = [r["success_step"] for r in results if r["success_step"] is not None]

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Success rate:         {100 * np.mean(successes):.1f}% ({sum(successes)}/{N_EPISODES})")
    print(f"Mean final distance:  {np.mean(final_dists):.4f} +/- {np.std(final_dists):.4f}")
    print(f"Mean reward:          {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}")
    print(f"Mean jerk:            {np.mean(jerks):.4f} +/- {np.std(jerks):.4f}")
    if success_steps:
        print(f"Mean steps to success: {np.mean(success_steps):.1f} +/- {np.std(success_steps):.1f}")
    else:
        print("Mean steps to success: N/A (no successes)")
    print("=" * 50)


if __name__ == "__main__":
    main()
