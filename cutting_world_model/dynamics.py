"""Normalized residual state model using SuperMimic-Reproduced's dynamics MLP."""
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn

FIELDS = ("particle_q", "particle_qd", "cut_spring_ke", "knife_pose", "knife_velocity", "force")


def pack_trajectory(path):
    with np.load(path) as data:
        count = len(data["time"])
        arrays = [data[name].reshape(count, -1) for name in FIELDS]
        state = np.concatenate(arrays, axis=1).astype(np.float32)
        layout = {}
        offset = 0
        for name, array in zip(FIELDS, arrays):
            layout[name] = (offset, offset + array.shape[1])
            offset += array.shape[1]
        return state, data["actions"].astype(np.float32), layout


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
