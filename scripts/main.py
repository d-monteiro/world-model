import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environment.arm import Arm3

env = Arm3(render_mode="human")
obs, info = env.reset(seed=0)

print("Starting simulation... Press Ctrl+C to stop.")

try:
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        
        if terminated or truncated:
            print("Episode finished")
            obs, info = env.reset()
            
except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    env.close()