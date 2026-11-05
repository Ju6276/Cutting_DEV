"""Episode-bounded training windows, including recorded short terminal sequences."""
from dataclasses import dataclass

import torch

from third_party.supermimic.replay_buffer import ReplayBuffer, WindowBatch


@dataclass
class MaskedWindowBatch(WindowBatch):
    valid_steps: torch.Tensor  # (k, n), true only for recorded transitions


class SequenceReplay(ReplayBuffer):
    tensors = ('s', 'a', 'a_exec', 'xi', 'r', 'done', 'frame', 'clip', 'ep_id',
               'terminal', 'ep_start', 'valid', 'ptr', 'ep_counter')

    @torch.no_grad()
    def state_dict(self):
        """Complete ring contents on CPU; rows never written are omitted when no world wrapped."""
        rows = self.T if bool((self.valid[:, -1]).any()) else max(int(self.ptr.max()), 1)
        data = {name: getattr(self, name).cpu() for name in self.tensors}
        for name in self.tensors[:12]:
            data[name] = data[name][:, :rows].clone()
        return {'format': 'sequence_replay_v1', 'capacity': self.T, 'rows': rows,
                'total_rows': self.total_rows, 'tensors': data}

    @torch.no_grad()
    def load_state_dict(self, saved):
        if saved.get('format') != 'sequence_replay_v1' or saved['capacity'] != self.T:
            raise ValueError('Replay format or capacity differs')
        rows = saved['rows']
        for name in self.tensors:
            value = saved['tensors'][name].to(self.device)
            target = getattr(self, name)
            if name in self.tensors[:12]:
                if value.shape[0] != self.E or value.shape[2:] != target.shape[2:]:
                    raise ValueError('Replay tensor shape differs: ' + name)
                target.zero_() if name != 'ep_id' else target.fill_(-1)
                target[:, :rows] = value
            else:
                target.copy_(value)
        self.total_rows = saved['total_rows']

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
