"""
Run the MPC planner on the arm environment with visualization.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.environment.arm import Arm3
from src.models.dynamics import DynamicsModel
from src.planning.mpc import MPCPlanner


def main():
    # Load dynamics model
    model = DynamicsModel(state_dim=5, action_dim=3, hidden_dim=256)
    model.load_state_dict(torch.load(ROOT / "src" / "checkpoints" / "dynamics.pt", weights_only=True))
    model.eval()
    print("Loaded dynamics model")

    env = Arm3(render_mode="human", max_episode_steps=400)

    planner = MPCPlanner(
        dynamics_model=model,
        env=env,
    )

    for ep in range(5):
        obs, _ = env.reset()
        env.render()

        state = obs[:5]
        goal = obs[5:7]

        print(f"\nEpisode {ep+1}")
        print(f"  Object: ({state[3]:.2f}, {state[4]:.2f})")
        print(f"  Goal:   ({goal[0]:.2f}, {goal[1]:.2f})")

        for t in range(400):
            action = planner.plan(state, goal, env.grasping)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            state = obs[:5]

            if t % 20 == 0:
                print(f"  step={t:3d}  ee→obj={info['distance_ee_obj']:.3f}  obj→goal={info['distance_obj_goal']:.3f}  grasp={info['grasping']}")

            if terminated:
                print(f"  GOAL REACHED at step {t}!")
                break
            if truncated:
                print(f"  Timeout. Final dist={info['distance_obj_goal']:.3f}")
                break

    env.close()


if __name__ == "__main__":
    main()
