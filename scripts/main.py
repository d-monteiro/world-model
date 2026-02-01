import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environment.arm import Arm3

env = Arm3(render_mode="human")
obs, info = env.reset(seed=42)
env.render()

print("Starting simulation... Press Ctrl+C to stop.")

try:
    for ep in range(5):
        obs, info = env.reset()
        env.render()
        for t in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if info["grasping"] and t % 10 == 0:
                print(f"  ep={ep} step={t} GRASPING! obj→goal={info['distance_obj_goal']:.3f}")

            if terminated:
                print(f"  ep={ep} step={t} GOAL REACHED!")
                break
            if truncated:
                break

        print(f"Episode {ep} done. Final dist={info['distance_obj_goal']:.3f} grasped={info['grasping']}")

except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    env.close()
