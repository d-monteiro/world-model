# Environment & Simulation Walkthrough

This document explains the 2D Robotic Arm environment setup and how to use it.

## Overview

The environment consists of:
1. **Custom Gymnasium Environment** (`src/environment/arm_env.py`) - The main environment
2. **Renderer** (`src/environment/renderer.py`) - Visualization tools
3. **Test Script** (`scripts/test_env.py`) - Example usage

## Environment Components

### 1. Robotic Arm

- **Multi-link arm**: Configurable number of joints (default: 2)
- **Link lengths**: Each link is 0.5 units long
- **Joint limits**: Full rotation (-π to π radians)
- **Joint velocities**: Maximum 0.2 radians per step

### 2. Objects & Goals

- **Movable objects**: Circular objects that can be picked up
- **Goal positions**: Target locations for objects
- **Grasping mechanism**: End-effector can "grasp" objects within 0.15 units

### 3. Observation Space

The observation is a **10-dimensional vector** (for 2 joints, 1 object):

```
[θ₁, θ₂,           # Joint angles (normalized by π)
 v₁, v₂,           # Joint velocities (normalized by max_velocity)
 x_obj, y_obj,     # Object position (normalized by workspace_size)
 x_goal, y_goal,   # Goal position (normalized by workspace_size)
 x_ee, y_ee]       # End-effector position (normalized by workspace_size)
```

**Why normalized?** Normalization helps neural networks learn faster by keeping values in a similar range.

### 4. Action Space

- **Continuous actions**: Joint velocity commands
- **Shape**: `(num_joints,)` - one value per joint
- **Range**: `[-0.2, 0.2]` radians per step
- **Meaning**: Change in joint angle for each joint

### 5. Reward Structure

- **Step penalty**: -0.01 per step (encourages efficiency)
- **Goal reward**: +10.0 when object reaches goal (within 0.1 units)
- **Proximity reward**: Gradual reward (up to +1.0) as object approaches goal
- **Grasping bonus**: +0.1 when grasping an object

### 6. Termination Conditions

- **Success**: All objects are within 0.1 units of their goals
- **Max steps**: Episode ends after 200 steps (truncated)

## How It Works

### Forward Kinematics

The arm's end-effector position is computed using forward kinematics:

```python
x, y = 0.0, 0.0  # Base position
angle_sum = 0.0

for angle, length in zip(joint_angles, link_lengths):
    angle_sum += angle
    x += length * cos(angle_sum)
    y += length * sin(angle_sum)
```

This calculates where the end of the arm is based on joint angles.

### Grasping Mechanism

1. **Check distance**: If end-effector is within 0.15 units of an object, it's "grasped"
2. **Move object**: When grasped, object moves with the end-effector
3. **Release**: If end-effector moves more than 0.225 units away, object is released

### Physics Simulation

- **Simple dynamics**: Joint angles update directly by action (velocity)
- **No friction or inertia**: Simplified for hackathon scope
- **Collision detection**: Objects can overlap (simplified physics)

## Usage Examples

### Basic Usage

```python
from src.environment.arm_env import Arm2DEnv

# Create environment
env = Arm2DEnv(
    num_joints=2,
    num_objects=1,
    workspace_size=2.0,
    render_mode="human"  # or "rgb_array" or None
)

# Reset environment
obs, info = env.reset(seed=42)

# Run episode
for step in range(200):
    # Get action (random, or from policy)
    action = env.action_space.sample()
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render (if render_mode is set)
    env.render()
    
    if terminated or truncated:
        break

env.close()
```

### Understanding Observations

```python
obs, info = env.reset()

# Observation breakdown (for 2 joints, 1 object):
# obs[0:2]   = joint angles
# obs[2:4]   = joint velocities  
# obs[4:6]   = object position
# obs[6:8]   = goal position
# obs[8:10]  = end-effector position

# Get actual positions (denormalized)
workspace_size = 2.0
object_pos = obs[4:6] * workspace_size
goal_pos = obs[6:8] * workspace_size
end_effector_pos = obs[8:10] * workspace_size
```

### Collecting Data for Training

```python
import numpy as np

# Collect trajectories
states = []
actions = []
next_states = []
rewards = []

env = Arm2DEnv()
obs, _ = env.reset()

for step in range(1000):
    action = env.action_space.sample()  # Random policy
    next_obs, reward, done, truncated, _ = env.step(action)
    
    states.append(obs)
    actions.append(action)
    next_states.append(next_obs)
    rewards.append(reward)
    
    obs = next_obs
    
    if done or truncated:
        obs, _ = env.reset()

# Convert to numpy arrays
states = np.array(states)
actions = np.array(actions)
next_states = np.array(next_states)
```

## Testing the Environment

Run the test script to verify everything works:

```bash
# Activate virtual environment
source venv/bin/activate

# Run test
python scripts/test_env.py
```

This will:
1. Test observation and action spaces
2. Run a short episode with random actions
3. Display the environment (if matplotlib backend supports it)

## Customization

### Change Number of Joints

```python
env = Arm2DEnv(num_joints=3)  # 3-joint arm
```

### Change Number of Objects

```python
env = Arm2DEnv(num_objects=2)  # 2 objects to move
```

### Adjust Workspace Size

```python
env = Arm2DEnv(workspace_size=3.0)  # Larger workspace
```

### Change Episode Length

```python
env = Arm2DEnv(max_episode_steps=500)  # Longer episodes
```

## Next Steps

Now that the environment is set up, you can:

1. **Collect training data**: Run random/heuristic policies to collect `(s, a, s')` tuples
2. **Train VAE**: Learn to encode states into latent space
3. **Train dynamics model**: Learn to predict `z_{t+1}` from `(z_t, a_t)`
4. **Mental rollouts**: Generate imagined trajectories without running the environment

## Troubleshooting

### Issue: Environment doesn't render

**Solution**: Make sure you set `render_mode="human"` and have a display available. In headless environments, use `render_mode="rgb_array"` to get image arrays.

### Issue: Objects not being grasped

**Solution**: The end-effector needs to be within 0.15 units of an object. Try moving the arm closer to the object.

### Issue: Observations seem wrong

**Solution**: Remember observations are normalized. Multiply by `workspace_size` to get actual positions.

### Issue: Episode ends too quickly

**Solution**: Increase `max_episode_steps` or adjust reward thresholds.

## Architecture Summary

```
Environment (arm_env.py)
├── State: joint angles, velocities, object/goal positions
├── Action: joint velocity commands
├── Reward: based on object-goal distances
└── Renderer (renderer.py)
    ├── Human visualization (matplotlib)
    └── Image generation (for image-based VAE)
```

The environment is now ready for world model learning!
