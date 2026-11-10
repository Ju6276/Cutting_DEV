"""Multi-condition cutting: several object groups, one batch, conditions sampled per world.

Each group is one prepared DiSECt scene (one object geometry/topology) simulated in its
own homogeneous ``GPUEnvironment``. Groups are concatenated along the world dimension so
one learner sees one batch with the training loop's existing interface (``current``,
``metadata``, ``heads``, ``step``, ``reset``). Fixed-size frame fields are concatenated;
per-object arrays (particles, springs) stay inside their group.

Conditions:
- object: fixed per group; every object shares one force limit (the hardware limit), so
  the learner has to infer how hard an object is from its own force feedback;
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


# Per-object arrays stay inside their group; only fixed-size fields are concatenated.
PER_OBJECT_KEYS = ('particle_q', 'particle_qd', 'cut_spring_ke', 'spring_reference_ke')


def _fixed(frame):
    return {k: v for k, v in frame.items() if k not in PER_OBJECT_KEYS}


class LocalGroup:
    """One object group simulated in this process."""

    def __init__(self, backend_root, scene, worlds, lift, reward, force_limits):
        self.env = GPUEnvironment(backend_root, worlds, reward=reward, scene=scene,
                                  initial_lift=lift, force_limits=force_limits)
        self.env.target_speed = torch.zeros(worlds, device=self.env.device)
        self.metadata = self.env.metadata
        self.current = _fixed(self.env.current)

    @torch.no_grad()
    def reset(self, local, scales, speeds):
        material = self.env.physics.material
        local = local.to(self.env.device)
        full = {k: material.material_scales[k].clone() for k in MATERIAL_KEYS}
        for k, v in scales.items():
            full[k][local] = v.to(full[k])
        material.set_material_scales(local, **full)
        self.env.target_speed[local] = speeds.to(self.env.target_speed)
        self.current = _fixed(self.env.reset(local))
        return self.current

    @torch.no_grad()
    def step(self, action, reference=False):
        frame, reward, terminated, truncated, info = self.env.step(action.to(self.env.device), reference=reference)
        self.current = _fixed(frame)
        return self.current, reward, terminated, truncated, info


def _cpu(value):
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(v) for v in value)
    return value.cpu() if isinstance(value, torch.Tensor) else value


def _group_worker(connection, arguments):
    import traceback
    try:
        torch.set_num_threads(2)
        group = LocalGroup(*arguments)
        connection.send(('ready', group.metadata, _cpu(group.current)))
        while True:
            command, payload = connection.recv()
            if command == 'close':
                break
            result = group.step(*payload) if command == 'step' else group.reset(*payload)
            connection.send(('ok', _cpu(result)))
    except BaseException:
        connection.send(('error', traceback.format_exc()))
    finally:
        connection.close()


class ProcessGroup:
    """One object group simulated in its own persistent process on the same GPU.
    Commands are sent to every group before any result is awaited, so groups step
    concurrently; tensors cross the process boundary on the CPU."""

    def __init__(self, backend_root, scene, worlds, lift, reward, force_limits):
        import torch.multiprocessing as mp
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_group_worker, daemon=True, args=(
            child, (str(backend_root), scene, worlds, lift.cpu(), reward, force_limits)))
        self.process.start()
        self.pending = None

    def finish_start(self, device):
        status, *payload = self.connection.recv()
        if status != 'ready':
            raise RuntimeError('Object group process failed:\n' + payload[0])
        self.device = device
        self.metadata, current = payload
        self.current = {k: v.to(device) for k, v in current.items()}

    def send(self, command, *payload):
        self.connection.send((command, _cpu(payload)))

    def receive(self):
        status, payload = self.connection.recv()
        if status != 'ok':
            raise RuntimeError('Object group process failed:\n' + payload)
        move = lambda v: {k: x.to(self.device) for k, x in v.items()} if isinstance(v, dict) else v.to(self.device)
        if isinstance(payload, tuple):
            payload = tuple(move(v) for v in payload)
            self.current = payload[0]
        else:
            payload = move(payload)
            self.current = payload
        return payload

    def close(self):
        try:
            self.connection.send(('close', None))
        except (BrokenPipeError, OSError):
            pass
        self.process.join(timeout=30)


class MultiConditionEnvironment:
    def __init__(self, backend_root, groups, reward=None, seed=0, force_limits=None, parallel=False):
        """``parallel``: one persistent process per object group (``ProcessGroup``);
        otherwise groups are stepped one after another in this process (reference path)."""
        if not groups:
            raise ValueError('At least one object group is required')
        self.groups = list(groups)
        self.parallel = parallel
        self.generator = torch.Generator().manual_seed(seed)
        self.members, self.slices, self.lift, start = [], [], [], 0
        for group in self.groups:
            for key, (low, high) in group.material.items():
                if key not in MATERIAL_KEYS or not 0 < low <= high:
                    raise ValueError(f'{group.name}: bad material range {key}={low, high}')
            lift = self._uniform(group.clearance, group.worlds) - knife_clearance(backend_root, group.scene)
            maker = ProcessGroup if parallel else LocalGroup
            self.members.append(maker(backend_root, group.scene, group.worlds, lift, reward, force_limits))
            self.lift.append(lift)
            self.slices.append(slice(start, start + group.worlds))
            start += group.worlds
        self.worlds = start
        self.device = torch.device('cuda:0')
        if parallel:
            for member in self.members:
                member.finish_start(self.device)
        self.metadata = self.members[0].metadata
        for member, group in zip(self.members[1:], self.groups[1:]):
            if member.metadata != self.metadata:
                raise ValueError(f'{group.name}: task metadata differs from {self.groups[0].name}')
        from .task_heads import SliceItHeads
        self.heads = SliceItHeads(self.metadata['environment'], self.metadata['controller'])
        self.group_index = torch.cat([torch.full((g.worlds,), i, dtype=torch.long)
                                      for i, g in enumerate(self.groups)]).to(self.device)
        self.material_scales = {k: torch.ones(self.worlds, device=self.device) for k in MATERIAL_KEYS}
        self.target_speed = torch.zeros(self.worlds, device=self.device)
        self.current = None
        self.reset()

    def close(self):
        for member in self.members:
            if hasattr(member, 'close'):
                member.close()

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

    @staticmethod
    def goal(frame):
        """Task goal of each world: target pose and target cutting speed."""
        return torch.cat((frame['target_pose'], frame['target_speed']), -1).float()

    @classmethod
    def from_config(cls, backend_root, config, reward=None):
        groups = [ObjectGroup(name=g['name'], scene=g['scene'], worlds=int(g['worlds']),
                              clearance=tuple(g.get('clearance', (.005, .03))),
                              material={k: tuple(v) for k, v in g.get('material', {}).items()},
                              target_speed=tuple(g.get('target_speed', (.035, .065))))
                  for g in config['groups']]
        return cls(backend_root, groups, reward=reward, seed=int(config.get('seed', 0)),
                   force_limits=config.get('force_limits'), parallel=bool(config.get('parallel', False)))

    @torch.no_grad()
    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.worlds, dtype=torch.bool, device=self.device)
        mask = mask.to(self.device, torch.bool)
        active = []
        for i, (group, member, part) in enumerate(zip(self.groups, self.members, self.slices)):
            local = mask[part]
            if not bool(local.any()):
                continue
            n = int(local.sum())
            scales = {k: self._uniform(group.material.get(k, (1., 1.)), n, log=True).float() for k in MATERIAL_KEYS}
            speeds = self._uniform(group.target_speed, n).float()
            for k, v in scales.items():
                self.material_scales[k][part][local] = v.to(self.device)
            self.target_speed[part][local] = speeds.to(self.device)
            if self.parallel:
                member.send('reset', local, scales, speeds)
                active.append(member)
            else:
                member.reset(local, scales, speeds)
        for member in active:
            member.receive()
        self.current = self._with_conditions(self._merge([m.current for m in self.members]))
        return self.current

    @torch.no_grad()
    def step(self, action, reference=False):
        if self.parallel:
            for member, part in zip(self.members, self.slices):
                member.send('step', action[part], reference)
            outputs = [member.receive() for member in self.members]
        else:
            outputs = [member.step(action[part], reference) for member, part in zip(self.members, self.slices)]
        frame = self._with_conditions(self._merge([o[0] for o in outputs]))
        reward, terminated, truncated = (torch.cat([o[i].to(self.device) for o in outputs]) for i in (1, 2, 3))
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
                         'start_lift_m': float(self.lift[g][w - part.start]),
                         'target_speed_mps': float(self.target_speed[w]),
                         **{k: float(self.material_scales[k][w]) for k in MATERIAL_KEYS}})
        return rows
