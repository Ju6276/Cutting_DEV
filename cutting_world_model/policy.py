"""SuperMimic encoder/decoder architecture with SliceIt observation, goal and action dimensions."""
import torch
from torch import nn

from third_party.supermimic.dynamics import MLP


class SliceItPolicy(nn.Module):
    history = 10
    observation_terms = (3, 3, 3, 6, 9)

    def __init__(self, goal_dim=7):
        super().__init__()
        hidden = (2048, 1024, 512, 256)
        self.encoder = MLP(goal_dim, hidden, 32)
        self.decoder = MLP(32 + self.history * 24, hidden, 6, last_gain=.01)

    def forward(self, history, target_pose):
        # history is (time, batch, 24), oldest first, matching the source's term-major layout.
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
