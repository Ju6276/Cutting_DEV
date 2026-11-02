"""Streaming writes to the unchanged upstream replay, with state-aligned terminal labels."""
import torch

from third_party.supermimic.replay_buffer import ReplayBuffer


class OnlineReplay:
    def __init__(self, state_dim, device):
        self.buffer = ReplayBuffer(1, 1000, state_dim, 6, 7, device)
        self.device = device
        self.env_id = torch.tensor([0], device=device)
        self.frame = 0
        self.episode = 0

    def add(self, state, clean, executed, goal, reward, next_state, terminated, truncated):
        b, device = self.buffer, self.device

        def row(s, a, ae, r, done, terminal, initial):
            b.add(self.env_id, s[None].detach(), a[None].detach(), ae[None].detach(), goal[None].detach(),
                  torch.tensor([r], dtype=torch.float32, device=device),
                  torch.tensor([done], dtype=torch.float32, device=device),
                  torch.tensor([self.frame], device=device), torch.tensor([self.episode], device=device),
                  torch.tensor([terminal], device=device), torch.tensor([initial], device=device))

        row(state, clean, executed, reward, False, False, self.frame == 0)
        self.frame += 1
        if terminated or truncated:
            zero = torch.zeros_like(clean)
            row(next_state, zero, zero, 0., terminated, True, False)
            b.end_episode(self.env_id)
            self.frame = 0
            self.episode += 1
