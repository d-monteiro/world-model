"""Visualize the environment with random actions."""

import physical_ai.envs  # noqa: F401
import gymnasium

env = gymnasium.make("ThreeJointArm-v0", render_mode="human")
obs, _ = env.reset(seed=42)

for i in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
