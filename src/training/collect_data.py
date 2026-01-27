"""
Collect experience data from the robotic arm environment.

This script runs random episodes and stores (state, action, next_state) transitions
to train the VAE and dynamics model later.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm
import pickle

from environment.arm import Arm3


def collect_data(num_episodes=1000, max_steps=200, save_path='data/transitions.pkl'):
    """
    Collect experience data from random episodes.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        save_path: Path to save the collected data
    """
    # Initialize environment
    env = Arm3()
    
    # Storage for transitions
    states = []
    actions = []
    next_states = []
    
    print(f"Collecting data from {num_episodes} episodes...")
    
    for episode in tqdm(range(num_episodes), desc="Episodes"):
        # Reset environment
        s, _ = env.reset()
        
        for t in range(max_steps):
            # Random action (sample from action space)
            a = env.action_space.sample()
            
            # Take step in environment
            s2, reward, terminated, truncated, info = env.step(a)
            
            # Store transition
            states.append(s)
            actions.append(a)
            next_states.append(s2)
            
            # Update state
            s = s2
            
            # Break if episode is done
            if terminated or truncated:
                break
    
    # Convert to numpy arrays
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    
    print(f"\nCollected {len(states)} transitions")
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Next states shape: {next_states.shape}")
    
    # Create data directory if it doesn't exist
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save data
    data = {
        'states': states,
        'actions': actions,
        'next_states': next_states,
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nData saved to {save_path.absolute()}")
    
    # Print some statistics
    print("\n=== Data Statistics ===")
    print(f"State mean: {states.mean(axis=0)}")
    print(f"State std: {states.std(axis=0)}")
    print(f"Action mean: {actions.mean(axis=0)}")
    print(f"Action std: {actions.std(axis=0)}")
    
    return data


if __name__ == "__main__":
    # Collect data
    data = collect_data(
        num_episodes=1000,
        max_steps=200,
        save_path='data/transitions.pkl'
    )
    
    print("\n✓ Data collection complete!")
