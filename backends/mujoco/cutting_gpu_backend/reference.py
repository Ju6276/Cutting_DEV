"""Prescribed knife trajectories for the reference control branch.

A trajectory is piecewise-constant DiSECt-frame knife linear velocity: interval
``[times[i], times[i + 1])`` moves with ``linear_velocity[i]`` (m/s). After the
last time the displacement is held. Orientation is not prescribed; the
controller target keeps the reset tip orientation, as in the constant reference.
"""
import hashlib
import json
from pathlib import Path

import torch

from .controller.fdcc import quat_apply

FORMAT = 'cutting-knife-velocity-v1'


class KnifeTrajectory:
    def __init__(self, times, linear_velocity, metadata=None):
        self.times = torch.as_tensor(times, dtype=torch.float64).flatten()
        self.linear_velocity = torch.as_tensor(linear_velocity, dtype=torch.float64)
        if self.times.ndim != 1 or self.times.numel() < 2:
            raise ValueError('Trajectory needs at least two times')
        if self.linear_velocity.shape != (self.times.numel() - 1, 3):
            raise ValueError('Trajectory needs one 3-D velocity per time interval')
        if float(self.times[0]) != 0. or not bool((self.times[1:] > self.times[:-1]).all()):
            raise ValueError('Trajectory times must start at zero and increase')
        if not bool(torch.isfinite(self.times).all() & torch.isfinite(self.linear_velocity).all()):
            raise ValueError('Trajectory must be finite')
        self.metadata = dict(metadata or {})

    @classmethod
    def constant(cls, velocity, duration, **metadata):
        return cls([0., duration], torch.as_tensor(velocity, dtype=torch.float64).reshape(1, 3), metadata)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get('format') != FORMAT or data.get('frame') != 'disect':
            raise ValueError(f'Expected {FORMAT} trajectory in the DiSECt frame')
        return cls(data['times'], data['linear_velocity'], data.get('source'))

    def to_dict(self):
        return {'format': FORMAT, 'frame': 'disect', 'times': self.times.tolist(),
                'linear_velocity': self.linear_velocity.tolist(), 'source': self.metadata}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), allow_nan=False) + '\n')

    def digest(self):
        return hashlib.sha256(json.dumps(
            [self.times.tolist(), self.linear_velocity.tolist()]).encode()).hexdigest()

    @property
    def duration(self):
        return float(self.times[-1])

    def displacement(self, time, rotation=None):
        """Knife displacement at ``time`` [worlds], optionally rotated into another frame.

        Velocities are rotated before integration so a one-interval trajectory
        evaluates exactly ``R v * min(time, T)``, the original constant reference.
        """
        times = self.times.to(time)
        velocity = self.linear_velocity.to(time)
        if rotation is not None:
            velocity = quat_apply(rotation.to(velocity).expand(len(velocity), 4), velocity)
        elapsed = torch.minimum(torch.maximum(time[:, None], times[:-1]), times[1:]) - times[:-1]
        return (elapsed[..., None] * velocity).sum(-2)
