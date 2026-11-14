"""SuperMimic encoder/decoder architecture with SliceIt observation, goal and action dimensions."""
import torch
from torch import nn

from third_party.supermimic.dynamics import MLP


class SliceItPolicy(nn.Module):
    history = 10
    observation_terms = (3, 3, 3, 6, 9)

    def __init__(self, goal_dim=7, observation_dim=24):
        super().__init__()
        if observation_dim not in (24, 25, 33, 34):
            raise ValueError('Policy supports 3- or 6-axis wrench observations, optionally with time')
        force_width = 18 if observation_dim in (33, 34) else 9
        self.observation_terms = (3, 3, 3, 6, force_width) + (
            (1,) if observation_dim in (25, 34) else ())
        self.observation_dim = observation_dim
        hidden = (2048, 1024, 512, 256)
        self.encoder = MLP(goal_dim, hidden, 32)
        self.decoder = MLP(32 + self.history * observation_dim, hidden, 6, last_gain=.01)

    def forward(self, history, target_pose):
        # History is (time, batch, observation_dim), source-compatible term-major layout.
        if history.shape[0] != self.history or history.shape[-1] != self.observation_dim:
            raise ValueError("Policy observation history shape does not match its interface")
        parts = []
        begin = 0
        for size in self.observation_terms:
            parts.append(history[..., begin:begin + size].permute(1, 0, 2).flatten(1))
            begin += size
        latent = torch.tanh(self.encoder(target_pose))
        return self.decoder(torch.cat((latent, *parts), dim=-1))


class SliceItCritic(nn.Module):
    def __init__(self, state_dim, goal_dim=7):
        super().__init__()
        self.mlp = MLP(state_dim + goal_dim + 6, (256, 128), 1)

    def forward(self, features, goal, error):
        return self.mlp(torch.cat((features, goal, error), -1)).squeeze(-1)
