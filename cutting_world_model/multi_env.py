"""Multi-condition cutting: several object groups, one batch, conditions sampled per world.

Each group is one prepared DiSECt scene (one object geometry/topology) simulated in its
own homogeneous ``GPUEnvironment``. Groups are concatenated along the world dimension so
one learner sees one batch with the training loop's existing interface (``current``,
``metadata``, ``heads``, ``step``, ``reset``). Fixed-size frame fields are concatenated;
per-object arrays (particles, springs) stay inside their group.

Conditions:
- object and its force-limit scale: fixed per group;
- start clearance (knife edge above the object top): fixed per world, set at construction
  by inverse kinematics (``CuttingBatch.initial_lift``);
- material scales (Young's modulus, density, cut-spring stiffness): log-uniform per
  episode, applied at reset;
- target cutting speed: uniform per episode, carried in the frame for the task reward.
"""
from dataclasses import dataclass, field
import math
from pathlib import Path

import torch

from .gpu_env import GPUEnvironment

MATERIAL_KEYS = ('youngs_modulus', 'density', 'cut_spring_stiffness')


@dataclass
class ObjectGroup:
    name: str
    scene: str                      # exported scene, relative to the backend root
    worlds: int
    force_limit_scale: float = 1.
    clearance: tuple = (.005, .03)  # knife edge above the object top (m), per world
    material: dict = field(default_factory=lambda: {k: (1., 1.) for k in MATERIAL_KEYS})
    target_speed: tuple = (.035, .065)  # m/s, per episode


def knife_clearance(backend_root, scene):
    """Vertical gap (m) between the scene's initial knife edge and the object top."""
    import sys
    disect_root = str(Path(backend_root).resolve() / 'third_party/disect')
    if disect_root not in sys.path:
        sys.path.insert(0, disect_root)
    saved = torch.load(Path(backend_root) / scene, map_location='cpu', weights_only=False)
    top = float(saved['state'].particle_q.reshape(-1, 3)[:, 1].max())
    edge = float(saved['initial_pos'][1] + saved['model'].knife_tri_vertices.reshape(-1, 3)[:, 1].min())
    return edge - top


class MultiConditionEnvironment:
    def __init__(self, backend_root, groups, reward=None, seed=0):
        if not groups:
            raise ValueError('At least one object group is required')
        self.groups = list(groups)
        self.generator = torch.Generator().manual_seed(seed)
        self.envs, self.slices, start = [], [], 0
        self.lift = []
        for group in self.groups:
            for key, (low, high) in group.material.items():
                if key not in MATERIAL_KEYS or not 0 < low <= high:
                    raise ValueError(f'{group.name}: bad material range {key}={low, high}')
            gap = knife_clearance(backend_root, group.scene)
            clearance = self._uniform(group.clearance, group.worlds)
            lift = clearance - gap
            env = GPUEnvironment(backend_root, group.worlds, reward=reward, scene=group.scene,
                                 initial_lift=lift,
                                 force_limit_scale=torch.full((group.worlds,), float(group.force_limit_scale)))
            self.envs.append(env)
            self.lift.append(lift)
            self.slices.append(slice(start, start + group.worlds))
            start += group.worlds
        self.worlds = start
        first = self.envs[0]
        self.device, self.heads, self.metadata = first.device, first.heads, first.metadata
        for env, group in zip(self.envs[1:], self.groups[1:]):
            if env.metadata != first.metadata:
                raise ValueError(f'{group.name}: task metadata differs from {self.groups[0].name}')
        self.group_index = torch.cat([torch.full((g.worlds,), i, dtype=torch.long)
                                      for i, g in enumerate(self.groups)]).to(self.device)
        self.material_scales = {k: torch.ones(self.worlds, device=self.device) for k in MATERIAL_KEYS}
        self.target_speed = torch.zeros(self.worlds, device=self.device)
        self.current = None
        self.reset()

    def _uniform(self, bounds, n, log=False):
        low, high = bounds
        u = torch.rand(n, generator=self.generator, dtype=torch.float64)
        if log:
            return torch.exp(math.log(low) + u * (math.log(high) - math.log(low)))
        return low + u * (high - low)

    def _merge(self, parts):
        keys = [k for k in parts[0] if all(k in p and p[k].shape[1:] == parts[0][k].shape[1:] for p in parts)]
        return {k: torch.cat([p[k] for p in parts]) for k in keys}

    def _with_conditions(self, frame):
        frame['target_speed'] = self.target_speed[:, None].to(frame['observation']).clone()
        frame['object_group'] = self.group_index[:, None].clone()
        return frame

    @torch.no_grad()
    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.worlds, dtype=torch.bool, device=self.device)
        mask = mask.to(self.device, torch.bool)
        parts = []
        for group, env, part in zip(self.groups, self.envs, self.slices):
            local = mask[part]
            if bool(local.any()):
                n = int(local.sum())
                scales = {k: self._uniform(group.material.get(k, (1., 1.)), n, log=True) for k in MATERIAL_KEYS}
                full = {k: env.physics.material.material_scales[k].clone() for k in MATERIAL_KEYS}
                for k, v in scales.items():
                    full[k][local] = v.to(full[k])
                    self.material_scales[k][part][local] = v.to(self.device, torch.float32)
                env.physics.material.set_material_scales(local, **full)
                speed = self._uniform(group.target_speed, n).to(self.device, torch.float32)
                self.target_speed[part.start:part.stop][local] = speed
                parts.append(env.reset(local))
            else:
                parts.append(env.current)
        self.current = self._with_conditions(self._merge(parts))
        return self.current

    @torch.no_grad()
    def step(self, action, reference=False):
        outputs = [env.step(action[part], reference=reference) for env, part in zip(self.envs, self.slices)]
        frame = self._with_conditions(self._merge([o[0] for o in outputs]))
        reward, terminated, truncated = (torch.cat([o[i] for o in outputs]) for i in (1, 2, 3))
        info = self._merge([o[4] for o in outputs])
        self.current = frame
        return frame, reward, terminated, truncated, info

    def conditions(self, worlds):
        """Per-world condition record for episode logs."""
        rows = []
        for w in worlds:
            g = int(self.group_index[w])
            part = self.slices[g]
            rows.append({'group': self.groups[g].name, 'scene': self.groups[g].scene,
                         'force_limit_scale': float(self.groups[g].force_limit_scale),
                         'start_lift_m': float(self.lift[g][w - part.start]),
                         'target_speed_mps': float(self.target_speed[w]),
                         **{k: float(self.material_scales[k][w]) for k in MATERIAL_KEYS}})
        return rows
