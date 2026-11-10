"""Streaming replay writes with state-aligned terminal labels."""
from pathlib import Path

import torch

from .sequence_replay import SequenceReplay


class OnlineReplay:
    def __init__(self, state_dim, device, capacity=1000):
        self.buffer = SequenceReplay(1, capacity, state_dim, 6, 7, device)
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


class BatchedOnlineReplay:
    def __init__(self, worlds, state_dim, device, capacity=1000, goal_dim=7):
        self.buffer = SequenceReplay(worlds, capacity, state_dim, 6, goal_dim, device)
        self.ids = torch.arange(worlds, device=device)
        self.frame = torch.zeros(worlds, dtype=torch.long, device=device)
        self.episode = torch.zeros_like(self.frame)

    def state_dict(self):
        return {'buffer': self.buffer.state_dict(), 'frame': self.frame.cpu(),
                'episode': self.episode.cpu()}

    def load_state_dict(self, saved):
        self.buffer.load_state_dict(saved['buffer'])
        self.frame.copy_(saved['frame'].to(self.frame.device))
        self.episode.copy_(saved['episode'].to(self.episode.device))

    def save(self, path, **metadata):
        """Atomic replay snapshot; online data otherwise exists only in device memory."""
        path = Path(path)
        temp = path.with_name(path.name + '.tmp')
        torch.save({**self.state_dict(), **metadata}, temp)
        temp.replace(path)

    @torch.no_grad()
    def add(self, state, clean, executed, goal, reward, next_state, terminated, truncated):
        false = torch.zeros_like(terminated)
        self.buffer.add(self.ids, state, clean, executed, goal, reward, false.float(),
                        self.frame, self.episode, false, self.frame == 0)
        self.frame += 1
        ended = terminated | truncated
        ids = self.ids[ended]
        if ids.numel():
            zero_action = torch.zeros_like(clean[ended])
            self.buffer.add(ids, next_state[ended], zero_action, zero_action, goal[ended],
                            torch.zeros_like(reward[ended]), terminated[ended].float(),
                            self.frame[ended], self.episode[ended], torch.ones_like(terminated[ended]), false[ended])
            self.buffer.end_episode(ids)
            self.frame[ended] = 0
            self.episode[ended] += 1
