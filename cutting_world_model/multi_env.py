"""Multi-condition cutting: several object groups, one batch, conditions sampled per world.

Each group is one prepared DiSECt scene (one object geometry/topology) simulated in its
own homogeneous ``GPUEnvironment``. Groups are concatenated along the world dimension so
one learner sees one batch with the training loop's existing interface (``current``,
``metadata``, ``heads``, ``step``, ``reset``). Fixed-size frame fields are concatenated;
per-object arrays (particles, springs) stay inside their group.

Conditions:
- object: fixed per group; every object shares one force limit (a shared simulation limit), so
  the learner has to infer how hard an object is from its own force feedback;
- start clearance (knife edge above the object top): fixed per world, set at construction
  by inverse kinematics (``CuttingBatch.initial_lift``);
- optional paired start_clearance_m: re-align to settled top after every reset;
- material scales (Young's modulus, density, cut-spring stiffness): log-uniform per
  episode, applied at reset;
- target cutting speed: uniform per episode, carried in the frame for the task reward.
"""
from dataclasses import dataclass, field
import math
from pathlib import Path

import torch

from .gpu_env import GPUEnvironment
from .task_heads import validate_force_limits, FORCE_OBSERVATION_REVISION

MATERIAL_KEYS = ('youngs_modulus', 'density', 'cut_spring_stiffness')


@dataclass
class ObjectGroup:
    name: str
    scene: str                      # exported scene, relative to the backend root
    worlds: int
    clearance: tuple = (.005, .03)  # knife edge above the object top (m), per world
    material: dict = field(default_factory=lambda: {k: (1., 1.) for k in MATERIAL_KEYS})
    target_speed: tuple = (.035, .065)  # m/s, per episode
    sim_dt: float = None            # DiSECt material substep (s); None keeps the scene's
    force_model: str = 'scalar'     # 'scalar' (source proxy) or 'net' (contact wrench at the sensor)
    friction_velocity: float = None  # knife friction transition velocity (m/s); None keeps the scene's
    settle_seconds: float = 0.      # material time advanced before the reset state is saved (free objects)
    start_clearance_m: list = None  # explicit paired cases, fixed per world


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
PER_OBJECT_KEYS = ('particle_q', 'particle_qd', 'cut_spring_ke', 'spring_reference_ke', 'object_reference_q')


def _fixed(frame):
    return {k: v for k, v in frame.items() if k not in PER_OBJECT_KEYS}


class LocalGroup:
    """One object group simulated in this process."""

    def __init__(self, backend_root, scene, worlds, lift, reward, force_limits, sim_dt=None,
                 force_model='scalar', friction_velocity=None, settle_seconds=0., task=None, force_observation=None, start_clearance_m=None, robot_platform=None):
        if not math.isfinite(settle_seconds) or settle_seconds < 0:
            raise ValueError('settle_seconds must be finite and nonnegative')
        self.settle_seconds = settle_seconds
        self.env = GPUEnvironment(backend_root, worlds, reward=reward, scene=scene,
                                  initial_lift=lift, force_limits=force_limits, sim_dt=sim_dt,
                                  force_model=force_model, friction_velocity=friction_velocity,
                                  settle_seconds=0., start_clearance_m=start_clearance_m, robot_platform=robot_platform, **({'task': task} if task is not None else {}),
                                  **({'force_observation': force_observation} if force_observation is not None else {}))
        self.env.target_speed = torch.zeros(worlds, device=self.env.device)
        if settle_seconds > 0:
            self.env.metadata['settle_seconds'] = float(settle_seconds)
            self.env.metadata['reset_material_protocol'] = 'export_per_material_settle_v1'
        self.metadata = self.env.metadata
        self.springs = torch.full((worlds, 1), float(self.env.current['cut_spring_ke'].shape[-1]),
                                  device=self.env.device)
        self.current = self._out(self.env.current)

    def _out(self, frame):
        frame = _fixed(frame)
        frame['total_springs'] = self.springs.clone()
        return frame

    @torch.no_grad()
    def reset(self, local, scales, speeds):
        material = self.env.physics.material
        local = local.to(self.env.device)
        full = {k: material.material_scales[k].clone() for k in MATERIAL_KEYS}
        for k, v in scales.items():
            full[k][local] = v.to(full[k])
        material.set_material_scales(local, **full)
        self.env.target_speed[local] = speeds.to(self.env.target_speed)
        frame = (self.env.settle(self.settle_seconds, local) if self.settle_seconds > 0
                 else self.env.reset(local))
        self.current = self._out(frame)
        return self.current

    @torch.no_grad()
    def step(self, action, reference=False):
        frame, reward, terminated, truncated, info = self.env.step(action.to(self.env.device), reference=reference)
        self.current = self._out(frame)
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
            try:
                command, payload = connection.recv()
            except EOFError:      # the parent exited without close(): shut down quietly
                break
            if command == 'close':
                break
            if command == 'capture':
                from .training_snapshot import capture_group
                capture_group(group.env, arguments[0], payload[0])
                connection.send(('ok', None))
                continue
            if command not in ('step', 'reset'):
                raise ValueError('Unknown object-group command: ' + command)
            result = group.step(*payload) if command == 'step' else group.reset(*payload)
            connection.send(('ok', _cpu(result)))
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class ProcessGroup:
    """One object group simulated in its own persistent process on the same GPU.
    Commands are sent to every group before any result is awaited, so groups step
    concurrently; tensors cross the process boundary on the CPU."""

    def __init__(self, backend_root, scene, worlds, lift, reward, force_limits, sim_dt=None,
                 force_model='scalar', friction_velocity=None, settle_seconds=0., task=None, force_observation=None, start_clearance_m=None, robot_platform=None):
        import torch.multiprocessing as mp
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_group_worker, daemon=True, args=(
            child, (str(backend_root), scene, worlds, lift.cpu(), reward, force_limits, sim_dt,
                    force_model, friction_velocity, settle_seconds, task, force_observation, start_clearance_m, robot_platform)))
        self._closed = False
        try:
            self.process.start()
        except BaseException:
            self.connection.close()
            raise
        finally:
            child.close()  # only the worker owns the other end of the pipe

    def _receive_message(self):
        while not self.connection.poll(0.1):
            if not self.process.is_alive():
                raise RuntimeError(f'Object group exited without a reply (exit code {self.process.exitcode})')
        try:
            return self.connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError('Object group connection closed before a reply') from exc

    def finish_start(self, device):
        status, *payload = self._receive_message()
        if status != 'ready':
            raise RuntimeError('Object group process failed:\n' + payload[0])
        self.device = device
        self.metadata, current = payload
        self.current = {k: v.to(device) for k, v in current.items()}

    def send(self, command, *payload):
        self.connection.send((command, _cpu(payload)))

    def receive(self):
        status, payload = self._receive_message()
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

    def close(self, timeout=30):
        if self._closed:
            return
        try:
            if self.process.is_alive():
                try:
                    self.connection.send(('close', None))
                except (BrokenPipeError, OSError):
                    pass
            self.process.join(timeout=timeout)
            if self.process.is_alive():
                import warnings
                warnings.warn('Object group did not shut down; terminating the worker', RuntimeWarning)
                self.process.terminate()
                self.process.join(timeout=timeout)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=timeout)
            if self.process.is_alive():
                raise RuntimeError('Object group could not be stopped')
        finally:
            self.connection.close()
            self._closed = True


class MultiConditionEnvironment:
    def __init__(self, backend_root, groups, reward=None, seed=0, force_limits=None, parallel=False, task=None, force_observation=None, episode_conditions=None, robot_platform=None):
        """``parallel``: one persistent process per object group (``ProcessGroup``);
        otherwise groups are stepped one after another in this process (reference path)."""
        force_limits = validate_force_limits(force_limits, required=True)
        if force_observation is not None and force_observation != FORCE_OBSERVATION_REVISION:
            raise ValueError("Unknown force_observation revision: " + str(force_observation))
        if not groups:
            raise ValueError('At least one object group is required')
        self.groups = list(groups)
        self.condition_sampler = None
        if episode_conditions is not None:
            from .episode_conditions import EpisodeConditions
            self.condition_sampler = EpisodeConditions(episode_conditions,
                [dict(name=g.name, worlds=g.worlds) for g in self.groups])
        self.parallel = parallel
        self.generator = torch.Generator().manual_seed(seed)
        self.members, self.slices, self.lift, start = [], [], [], 0
        try:
            for group in self.groups:
                for key, (low, high) in group.material.items():
                    if key not in MATERIAL_KEYS or not 0 < low <= high:
                        raise ValueError(f'{group.name}: bad material range {key}={low, high}')
                clearances = (self._uniform(group.clearance, group.worlds)
                              if group.start_clearance_m is None
                              else torch.as_tensor(group.start_clearance_m, dtype=torch.float64))
                if clearances.shape != (group.worlds,):
                    raise ValueError('One explicit clearance per world required')
                lift = clearances - knife_clearance(backend_root, group.scene)
                maker = ProcessGroup if parallel else LocalGroup
                self.members.append(maker(backend_root, group.scene, group.worlds, lift, reward, force_limits,
                                          group.sim_dt, group.force_model, group.friction_velocity,
                                          group.settle_seconds, **({'robot_platform': robot_platform} if robot_platform is not None else {}), **({'start_clearance_m': group.start_clearance_m} if group.start_clearance_m is not None else {}), **({'task': task} if task is not None else {}),
                                          **({'force_observation': force_observation} if force_observation is not None else {})))
                self.lift.append(lift)
                self.slices.append(slice(start, start + group.worlds))
                start += group.worlds
            self.worlds = start
            self.device = torch.device('cuda:0')
            if parallel:
                for member in self.members:
                    member.finish_start(self.device)
            # Friction can differ by object. Observation/limit frames must still agree.
            self.metadata = dict(self.members[0].metadata)
            # Reset preparation can differ by object without changing learner
            # state/reward semantics. Preserve it explicitly at group scope.
            per_group = {'material_contact', 'settle_seconds', 'reset_material_protocol'}
            shared = lambda m: {k: v for k, v in m.items() if k not in per_group}
            for member, group in zip(self.members[1:], self.groups[1:]):
                if shared(member.metadata) != shared(self.metadata):
                    raise ValueError(f'{group.name}: task metadata differs from {self.groups[0].name}')
            reset_settings = [
                {'settle_seconds': m.metadata.get('settle_seconds', 0.),
                 'reset_material_protocol': m.metadata.get('reset_material_protocol')}
                for m in self.members]
            if any(settings != reset_settings[0] for settings in reset_settings[1:]):
                self.metadata.pop('settle_seconds', None)
                self.metadata.pop('reset_material_protocol', None)
                self.metadata['group_reset_settings'] = [
                    {'name': g.name, 'scene': g.scene, **settings}
                    for g, settings in zip(self.groups, reset_settings)]
            if any('material_contact' in m.metadata for m in self.members):
                self.metadata.pop('material_contact', None)
                self.metadata['group_material_contact'] = [
                    {'name': g.name, 'scene': g.scene, **m.metadata.get('material_contact', {
                        'force_model': 'scalar', 'friction_velocity_override_mps': None})}
                    for g, m in zip(self.groups, self.members)]
            from .task_heads import SliceItHeads
            self.heads = SliceItHeads(self.metadata['environment'], self.metadata['controller'])
            self.group_index = torch.cat([torch.full((g.worlds,), i, dtype=torch.long)
                                          for i, g in enumerate(self.groups)]).to(self.device)
            self.material_scales = {k: torch.ones(self.worlds, device=self.device) for k in MATERIAL_KEYS}
            self.target_speed = torch.zeros(self.worlds, device=self.device)
            if self.condition_sampler is not None:
                from .episode_conditions import metadata
                self.metadata['episode_conditions'] = metadata(self.condition_sampler.config)
            self.current = None
            self.reset()
        except BaseException:
            self.close()
            raise

    def close(self):
        errors = []
        for member in self.members:
            if hasattr(member, 'close'):
                try:
                    member.close()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise RuntimeError('Failed to close object group(s): ' + repr(errors)) from errors[0]

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
        if self.condition_sampler is not None:
            frame['condition_episode'] = torch.as_tensor(
                self.condition_sampler.next_episode-1,device=self.device)[:,None].clone()
        return frame

    @staticmethod
    def goal(frame):
        """Task goal of each world: target pose and target cutting speed."""
        return torch.cat((frame['target_pose'], frame['target_speed']), -1).float()

    @classmethod
    def from_config(cls, backend_root, config, reward=None):
        groups = [ObjectGroup(name=g['name'], scene=g['scene'], worlds=int(g['worlds']),
                              clearance=tuple(g.get('clearance', (.005, .03))),
                              start_clearance_m=g.get('start_clearance_m'),
                              material={k: tuple(v) for k, v in g.get('material', {}).items()},
                              target_speed=tuple(g.get('target_speed', (.035, .065))),
                              sim_dt=None if g.get('sim_dt') is None else float(g['sim_dt']),
                              force_model=g.get('force_model', 'scalar'),
                              friction_velocity=(None if g.get('friction_velocity') is None
                                                 else float(g['friction_velocity'])),
                              settle_seconds=float(g.get('settle_seconds', 0.)))
                  for g in config['groups']]
        return cls(backend_root, groups, reward=reward, seed=int(config.get('seed', 0)),
                   force_limits=config.get('force_limits'), parallel=bool(config.get('parallel', False)), task=config.get('task'),
                   force_observation=config.get('force_observation'), episode_conditions=config.get('episode_conditions'), robot_platform=config.get('robot_platform'))

    @torch.no_grad()
    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.worlds, dtype=torch.bool, device=self.device)
        mask = mask.to(self.device, torch.bool)
        conditions = (None if self.condition_sampler is None else
                      self.condition_sampler.sample(mask.cpu().numpy()))
        active = []
        for i, (group, member, part) in enumerate(zip(self.groups, self.members, self.slices)):
            local = mask[part]
            if not bool(local.any()):
                continue
            n = int(local.sum())
            if conditions is None:
                scales = {k: self._uniform(group.material.get(k, (1., 1.)), n, log=True).float() for k in MATERIAL_KEYS}
                speeds = self._uniform(group.target_speed, n).float()
            else:
                indices = torch.arange(part.start, part.stop)[local.cpu()].numpy()
                scales = {k:torch.tensor(conditions[source][indices], dtype=torch.float32)
                          for k,source in (('youngs_modulus','youngs_modulus'),
                                           ('density','density'),
                                           ('cut_spring_stiffness','cut_strength'))}
                speeds = torch.tensor(conditions['target_speed'][indices],dtype=torch.float32)
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
                         **({'start_clearance_m': float(self.current['start_clearance_m'][w, 0])}
                            if 'start_clearance_m' in self.current else {}),
                         **({'condition_episode': int(self.condition_sampler.next_episode[w]-1)}
                            if self.condition_sampler is not None else {}),
                         'target_speed_mps': float(self.target_speed[w]),
                         **{k: float(self.material_scales[k][w]) for k in MATERIAL_KEYS}})
        return rows
