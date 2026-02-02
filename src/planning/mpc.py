"""
MPC Planner using short-horizon CEM with damped Jacobian initialization.

Fixes over previous versions:
- Short horizon (8 steps) to prevent dynamics model error compounding
- During grasping, cost uses analytical FK (ee position) not dynamics-predicted obj
- Damped pseudoinverse Jacobian handles near-singular configurations
- Pure Jacobian sequence always included as a candidate
"""

import torch
import numpy as np
from collections import deque


class MPCPlanner:

    def __init__(
        self,
        dynamics_model,
        env,
        action_dim: int = 3,
        horizon: int = 12,
        n_candidates: int = 500,
        n_elites: int = 50,
        cem_iterations: int = 5,
        max_dq: float = 0.05,
        grasp_threshold: float = 0.15,
    ):
        self.dynamics = dynamics_model
        self.env = env
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.n_elites = n_elites
        self.cem_iterations = cem_iterations
        self.max_dq = max_dq
        self.grasp_threshold = grasp_threshold

        self.prev_mean = None
        self.was_grasping = False

        # Loop detection
        self.recent_states = deque(maxlen=10)
        self.stuck_count = 0

    def _ee_from_q(self, q_batch: torch.Tensor) -> torch.Tensor:
        """Vectorized forward kinematics."""
        lengths = torch.tensor(self.env.link_lengths, dtype=torch.float32)
        angle_sum = torch.zeros(q_batch.shape[0])
        x = torch.zeros(q_batch.shape[0])
        y = torch.zeros(q_batch.shape[0])

        for i in range(3):
            angle_sum = angle_sum + q_batch[:, i]
            x = x + lengths[i] * torch.cos(angle_sum)
            y = y + lengths[i] * torch.sin(angle_sum)

        return torch.stack([x, y], dim=1)

    def _jacobian_action(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute a goal-directed action using the damped pseudoinverse Jacobian.

        Uses J^T (J J^T + lambda I)^{-1} instead of plain J^T to handle
        near-singular configurations (arm near full extension).
        """
        lengths = self.env.link_lengths
        n_joints = len(lengths)

        # Compute Jacobian
        J = np.zeros((2, n_joints))
        angle_sums = np.cumsum(q)

        for i in range(n_joints):
            for k in range(i, n_joints):
                J[0, i] += -lengths[k] * np.sin(angle_sums[k])
                J[1, i] += lengths[k] * np.cos(angle_sums[k])

        # Current end-effector position
        ee = self.env._joint_positions(q)[-1]

        # Desired EE velocity (toward target)
        direction = target - ee
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return np.zeros(n_joints, dtype=np.float32)

        # Damped pseudoinverse: dq = J^T (J J^T + λI)^{-1} direction
        # λ provides regularization near singularities
        damping = 0.05
        JJT = J @ J.T
        dq = J.T @ np.linalg.solve(JJT + damping * np.eye(2), direction)

        # Scale to max_dq
        max_abs = np.abs(dq).max()
        if max_abs > 1e-8:
            dq = dq / max_abs * self.max_dq

        return dq.astype(np.float32)

    def _rollout_cost(self, state: torch.Tensor, goal: torch.Tensor,
                      action_sequences: torch.Tensor, grasping: bool) -> torch.Tensor:
        """
        Simulate all candidates and return costs.

        Key: when grasping, use analytical FK position (ee) for delivery cost,
        NOT the dynamics model's predicted object position which drifts over time.
        """
        n = action_sequences.shape[0]
        s = state.unsqueeze(0).repeat(n, 1)
        is_grasping = torch.full((n,), grasping, dtype=torch.bool)
        total_cost = torch.zeros(n)

        for t in range(self.horizon):
            actions = action_sequences[:, t, :]
            s = self.dynamics(s, actions)

            ee = self._ee_from_q(s[:, :3])
            obj = s[:, 3:5]

            ee_to_obj = torch.norm(ee - obj, dim=1)
            is_grasping = is_grasping | (ee_to_obj < self.grasp_threshold)

            # Reaching cost: move EE to object
            cost_reach = ee_to_obj
            # Delivery cost: use EE position (= true object position when grasping)
            cost_deliver = torch.norm(ee - goal, dim=1)
            cost = torch.where(is_grasping, cost_deliver, cost_reach)

            weight = (t + 1) / self.horizon
            total_cost += cost * weight

        return total_cost

    def _is_stuck(self, state: np.ndarray) -> bool:
        """Check if we're oscillating between similar states."""
        state_round = np.round(state, 2)

        for prev in self.recent_states:
            if np.allclose(state_round, prev, atol=0.03):
                self.stuck_count += 1
                return self.stuck_count > 3

        self.stuck_count = max(0, self.stuck_count - 1)
        self.recent_states.append(state_round)
        return False

    def plan(self, state: np.ndarray, goal: np.ndarray, grasping: bool) -> np.ndarray:
        """Pick the best action using CEM with Jacobian-directed initialization."""
        self.dynamics.eval()

        # Reset on grasp change
        if grasping != self.was_grasping:
            self.prev_mean = None
            self.was_grasping = grasping
            self.recent_states.clear()
            self.stuck_count = 0

        # If stuck, reset and widen exploration
        stuck = self._is_stuck(state)
        if stuck:
            self.prev_mean = None
            self.stuck_count = 0
            self.recent_states.clear()

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32)
            goal_t = torch.tensor(goal, dtype=torch.float32)

            # Compute Jacobian-directed action
            q = state[:3]
            if grasping:
                jac_action = self._jacobian_action(q, goal)
            else:
                jac_action = self._jacobian_action(q, state[3:5])

            jac_action_t = torch.tensor(jac_action, dtype=torch.float32)

            # Initialize CEM distribution centered on Jacobian direction
            if self.prev_mean is not None and not stuck:
                mean = torch.cat([self.prev_mean[1:], jac_action_t.unsqueeze(0)])
            else:
                mean = jac_action_t.unsqueeze(0).repeat(self.horizon, 1)

            init_std = self.max_dq * (1.0 if stuck else 0.7)
            std = torch.ones(self.horizon, self.action_dim) * init_std

            best_sequence = None
            best_cost = float("inf")

            for iteration in range(self.cem_iterations):
                # Sample candidates around mean
                noise = torch.randn(self.n_candidates - 1, self.horizon, self.action_dim)
                sampled = mean.unsqueeze(0) + noise * std.unsqueeze(0)

                # Always include pure Jacobian sequence as a candidate
                jac_sequence = jac_action_t.unsqueeze(0).repeat(self.horizon, 1).unsqueeze(0)
                action_sequences = torch.cat([jac_sequence, sampled], dim=0)
                action_sequences = action_sequences.clamp(-self.max_dq, self.max_dq)

                costs = self._rollout_cost(state_t, goal_t, action_sequences, grasping)

                elite_idx = torch.argsort(costs)[:self.n_elites]
                elites = action_sequences[elite_idx]

                new_mean = elites.mean(dim=0)
                new_std = elites.std(dim=0)
                alpha = 0.3
                mean = alpha * mean + (1 - alpha) * new_mean
                std = alpha * std + (1 - alpha) * new_std
                std = std.clamp(min=self.max_dq * 0.1)

                if costs[elite_idx[0]] < best_cost:
                    best_cost = costs[elite_idx[0]]
                    best_sequence = elites[0]

            self.prev_mean = mean.clone()

            return best_sequence[0].numpy()
