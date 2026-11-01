"""Normalized residual state model using SuperMimic-Reproduced's dynamics MLP."""
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn

FIELDS = ("particle_q", "particle_qd", "cut_spring_ke", "knife_pose", "knife_velocity", "force")
COUPLED_FIELDS = ("observation", "joint_q", "joint_qd", "ee_pose", "wrench_history",
                  "motion_command", "previous_ee_pose", "previous_velocity",
                  "previous_acceleration", "target_pose", "success_counter", "step_count")


def pack_trajectory(path):
    with np.load(path) as data:
        count = len(data["time"])
        fields = FIELDS + COUPLED_FIELDS if "observation" in data else FIELDS
        arrays = [data[name].reshape(count, -1) for name in fields]
        state = np.concatenate(arrays, axis=1).astype(np.float32)
        actions = data["actions"].astype(np.float32)
        if actions.shape != (count - 1, 6):
            raise ValueError("Each trajectory must have T actions and T+1 states")
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError("Trajectory contains non-finite values")
        layout = {}
        offset = 0
        for name, array in zip(fields, arrays):
            layout[name] = (offset, offset + array.shape[1])
            offset += array.shape[1]
        return state, actions, layout


def load_upstream(root):
    path = Path(root) / "src/supermimic/world_model/dynamics.py"
    spec = importlib.util.spec_from_file_location("supermimic_dynamics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DynamicsMLP


class CuttingDynamics(nn.Module):
    def __init__(self, upstream_root, mean, scale, hidden=(2048, 2048, 2048)):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        cls = load_upstream(upstream_root)
        self.dynamics = cls(feat_dim=len(mean), act_dim=6, xi_dim=0,
                            hidden=hidden, out_dim=len(mean))

    def forward(self, state, action):
        normalized = (state - self.mean) / self.scale
        context = state.new_empty((*state.shape[:-1], 0))
        delta = self.dynamics(normalized, action, context)
        return state + self.scale * delta

    def rollout(self, initial, actions):
        state = initial
        states = []
        for action in actions.unbind(0):
            state = self(state, action)
            states.append(state)
        return torch.stack(states)
