"""Episode-bounded training windows, including recorded short terminal sequences."""
from dataclasses import dataclass

import torch

from third_party.supermimic.replay_buffer import ReplayBuffer, WindowBatch


@dataclass
class MaskedWindowBatch(WindowBatch):
    valid_steps: torch.Tensor  # (k, n), true only for recorded transitions


class SequenceReplay(ReplayBuffer):
    @torch.no_grad()
    def window_lengths(self, k):
        if not 1 <= k < self.T:
            raise ValueError('Window length must be positive and smaller than replay capacity')
        positions = torch.arange(self.T, device=self.device)[None, :]
        distance = (self.ptr[:, None] - positions - 1) % self.T
        alive = self.valid & ~self.terminal
        lengths = torch.zeros_like(self.ep_id)
        ended = torch.zeros_like(alive)
        for j in range(1, k + 1):
            alive = (alive & (distance >= j) & self._rolled(self.valid, j)
                     & (self._rolled(self.ep_id, j) == self.ep_id))
            lengths += alive.long()
            terminal = alive & self._rolled(self.terminal, j)
            ended |= terminal
            alive &= ~terminal
        return torch.where(ended | (lengths == k), lengths, 0)

    @torch.no_grad()
    def window_mask(self, k):
        return self.window_lengths(k) > 0

    @torch.no_grad()
    def sample_windows(self, n, k):
        lengths = self.window_lengths(k)
        e, t = self._sample_from_mask(lengths > 0, n)
        length = lengths[e, t]
        offsets = torch.arange(k + 1, device=self.device)[:, None]
        # Padding repeats the final recorded state, never a state after reset.
        times = (t[None, :] + torch.minimum(offsets, length[None, :])) % self.T
        envs = e[None, :].expand(k + 1, n)
        valid = offsets[:-1] < length[None, :]
        return MaskedWindowBatch(
            states=self.s[envs, times],
            a_exec=torch.where(valid[..., None], self.a_exec[envs[:-1], times[:-1]], 0.),
            a_clean=torch.where(valid[..., None], self.a[envs[:-1], times[:-1]], 0.),
            xi=self.xi[e, t],
            reward=torch.where(valid, self.r[envs[:-1], times[:-1]], 0.),
            done=torch.where(valid, self.done[envs[1:], times[1:]], 0.),
            frame=self.frame[envs[:-1], times[:-1]],
            valid_steps=valid,
        )
