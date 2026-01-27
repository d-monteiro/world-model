# it's a solid bock, doesn't work as I expected

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

from src.environment.arm import Arm3

# ----- Joint limits (match Arm3 env) -----
env = Arm3()

# Use the env's joint_low / joint_high directly
q1_min, q2_min, q3_min = env.joint_low
q1_max, q2_max, q3_max = env.joint_high

step = 0.05  # rad

# ----- Create grids for q1, q2, q3 -----
q1_vals = np.arange(q1_min, q1_max + step / 2.0, step)
q2_vals = np.arange(q2_min, q2_max + step / 2.0, step)
q3_vals = np.arange(q3_min, q3_max + step / 2.0, step)

# Meshgrid: every combination of angles
Q1, Q2, Q3 = np.meshgrid(q1_vals, q2_vals, q3_vals, indexing="ij")

# Flatten to point cloud (N, 3)
q1_flat = Q1.ravel()
q2_flat = Q2.ravel()
q3_flat = Q3.ravel()

# ----- Feasibility check via Arm3.is_valid_q -----
def is_valid(q1, q2, q3):
    q = np.array([q1, q2, q3], dtype=np.float32)
    return env.is_valid_q(q)

mask = np.array(
    [is_valid(a, b, c) for a, b, c in zip(q1_flat, q2_flat, q3_flat)],
    dtype=bool,
)

q1_valid = q1_flat[mask]
q2_valid = q2_flat[mask]
q3_valid = q3_flat[mask]

print(f"Total combinations: {q1_flat.size}")
print(f"Valid combinations (Arm3.is_valid_q): {q1_valid.size}")

# ----- 3D scatter plot of (q1, q2, q3) -----
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    q1_valid,
    q2_valid,
    q3_valid,
    s=1,          # tiny points for dense cloud
    alpha=0.5,
    c="tab:blue",
)

ax.set_xlabel("q1 (rad)")
ax.set_ylabel("q2 (rad)")
ax.set_zlabel("q3 (rad)")
ax.set_title("3D Joint Angle State Space (q1, q2, q3)")

# Set axis limits with padding for better visualization
padding = 2.3  # radians
ax.set_xlim(q1_min - padding, q1_max + padding)
ax.set_ylim(q2_min - padding, q2_max + padding)
ax.set_zlim(q3_min - padding, q3_max + padding)

plt.tight_layout()
plt.show()