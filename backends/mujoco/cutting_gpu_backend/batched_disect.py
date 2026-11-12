"""Packed independent copies of one exported DiSECt scene, in local world coordinates."""
import copy
import math
import torch

from .disect import DiSECt
from .controller.fdcc import quat_mul


def repeat_tensor(value, worlds):
    return value.detach().repeat((worlds,) + (1,) * (value.ndim - 1)).clone()


def pack_model(source, worlds):
    if source.enable_tri_collisions:
        raise ValueError('Triangle self-collision needs an environment-aware pair list')
    result = copy.copy(source)
    shared = {'gravity', 'knife_params', 'knife_tri_indices', 'knife_tri_vertices',
              'com', 'com_above_cut', 'com_below_cut'}
    offsets = {
        'tet_indices': source.particle_count, 'tri_indices': source.particle_count,
        'spring_indices': source.particle_count, 'edge_indices': source.particle_count,
        'cut_edge_indices': source.particle_count,
        'cut_spring_indices': source.cut_edge_count,
        'shape_body': source.link_count, 'joint_parent': source.link_count,
        'cut_tri_indices': source.particle_count,
        'cut_virtual_tri_indices': source.cut_edge_count,
        'cut_virtual_tri_indices_above_cut': source.cut_edge_count,
        'cut_virtual_tri_indices_below_cut': source.cut_edge_count,
        'contact_body0': source.link_count,
        'articulation_J_start': source.J_size, 'articulation_M_start': source.M_size,
        'articulation_H_start': source.H_size,
        'articulation_dof_start': source.joint_dof_count,
        'articulation_coord_start': source.joint_coord_count,
    }
    sentinels = {'joint_q_start': source.joint_coord_count,
                 'joint_qd_start': source.joint_dof_count,
                 'articulation_joint_start': source.link_count}
    for name, value in vars(source).items():
        if isinstance(value, torch.Tensor):
            if name in shared or value.ndim == 0:
                value = value.detach().clone()
            elif name in sentinels:
                stride = sentinels[name]
                value = torch.cat([value[:-1] + i * stride for i in range(worlds)] + [value[-1:] * worlds])
            elif name in offsets:
                stride = offsets[name]
                value = torch.cat([torch.where(value >= 0, value + i * stride, value)
                                   for i in range(worlds)])
            else:
                value = repeat_tensor(value, worlds)
            setattr(result, name, value)
        elif isinstance(value, int) and (name.endswith('_count') or name in ('J_size', 'M_size', 'H_size')):
            setattr(result, name, value * worlds)
    result.knife_tri_count = source.knife_tri_count
    result.particles_per_world = source.particle_count
    result.springs_per_world = source.cut_spring_count
    result.links_per_world = source.link_count
    result.shapes_per_world = source.shape_count
    return result


# Per-world material scales relative to the exported scene. Each scale multiplies the listed
# model arrays (and divides the inverse arrays); Poisson's ratio and spring softness stay fixed.
MATERIAL_SCALES = {
    'youngs_modulus': {'scaled': ('tet_mu', 'tet_lambda'), 'inverse': ()},
    'density': {'scaled': ('particle_mass',), 'inverse': ('particle_inv_mass',)},
    'cut_spring_stiffness': {'scaled': ('cut_spring_stiffness',), 'inverse': ()},
}


class BatchedDiSECt(DiSECt):
    def __init__(self, scene_path, worlds, sim_dt=None, friction_velocity=None):
        """``friction_velocity`` (m/s) replaces the scene's knife-contact ``sdf_kf``, which the
        contact kernel uses as the tangential transition velocity of its friction law; None keeps
        the exported value."""
        if friction_velocity is not None and (
                isinstance(friction_velocity, bool) or not math.isfinite(float(friction_velocity))
                or float(friction_velocity) <= 0):
            raise ValueError('friction_velocity must be positive and finite')
        if worlds < 1:
            raise ValueError('worlds must be positive')
        super().__init__(scene_path, sim_dt=sim_dt)
        self._scene_path = scene_path
        self._friction_velocity = friction_velocity
        self._settle_batches = {}
        self.worlds = worlds
        self.particles = self.model.particle_count
        self.springs = self.model.cut_spring_count
        self.model = pack_model(self.model, worlds)
        for name, value in list(vars(self.state).items()):
            if isinstance(value, torch.Tensor):
                setattr(self.state, name, repeat_tensor(value, worlds))
            elif name.endswith('_count'):
                setattr(self.state, name, value * worlds)
        for name in ('position', 'rotation', 'linear_velocity', 'angular_velocity'):
            setattr(self, name, getattr(self, name)[None].repeat(worlds, 1))
        for name in ('body_ft_s', 'body_f_ext_s'):
            setattr(self.state, name, torch.zeros((self.model.link_count, 6), device=self.position.device))
        self.base_material = {name: getattr(self.model, name).reshape(worlds, -1).clone()
                              for spec in MATERIAL_SCALES.values()
                              for name in spec['scaled'] + spec['inverse']}
        self.material_scales = {key: torch.ones(worlds, device=self.position.device) for key in MATERIAL_SCALES}
        if friction_velocity is not None:
            self.model.sdf_kf.fill_(float(friction_velocity))
        springs = self.model.cut_spring_indices.reshape(-1, 2).long()   # cut edges (left, right) per spring
        edges = self.model.cut_edge_indices.reshape(-1, 2).long()       # particles of each cut edge
        # Same order as state.knife_f (spring * 2 + side); the right edge is traversed reversed,
        # as in eval_spring_sdf_contacts.
        self._contact_edge = springs.reshape(-1)
        self._contact_first = torch.stack((edges[springs[:, 0], 0], edges[springs[:, 1], 1]), 1).reshape(-1)
        self._contact_second = torch.stack((edges[springs[:, 0], 1], edges[springs[:, 1], 0]), 1).reshape(-1)
        # no_grad integration overwrites particle_q after evaluating knife_f.
        # Keep the positions from that force evaluation for its moment arm.
        self.state.knife_contact_q = self.state.particle_q.clone()
        self.save_reset_state()
        # Immutable export snapshot: re-settle from here after material randomization,
        # never from the last episode's deformed/settled geometry.
        self.export_state = {k: v.clone() for k, v in self.initial_state.items()}
        self.export_contact = {k: v.clone() for k, v in self.initial_contact.items()}

    @torch.no_grad()
    def restore_export(self, mask):
        for name, initial in self.export_state.items():
            self.state_field(name)[mask] = initial.reshape_as(self.state_field(name))[mask]
        for name, initial in self.export_contact.items():
            target = getattr(self.model, name).reshape(self.worlds, -1)
            target[mask] = initial.reshape_as(target)[mask]
        self.state_field('cut_spring_ke')[mask] *= self.material_scales['cut_spring_stiffness'][mask, None]

    @torch.no_grad()
    def settle_selected(self, seconds, mask):
        """Settle only resetting rows in cached packed scratch batches.

        Each row uses its actual material, pose and export state. Padding duplicates a row
        but never participates in feedback or contact tests. No live world is advanced.
        """
        indices = mask.nonzero(as_tuple=False).flatten()
        n = len(indices)
        if not n:
            return
        capacity = 1 << (n - 1).bit_length()
        if capacity not in self._settle_batches:
            # LRU world-capacity budget: at most one rounded-up full material batch
            # in scratch storage, rather than retaining 1+2+...+N forever.
            budget = 1 << (self.worlds - 1).bit_length()
            while self._settle_batches and sum(self._settle_batches) + capacity > budget:
                oldest = next(iter(self._settle_batches))
                evicted = self._settle_batches.pop(oldest)
                # The cached Python step closes over its material. Break that cycle so
                # graph/tensor allocations release now, not at an arbitrary GC pass.
                if hasattr(evicted, '_stationary_step'):
                    del evicted._stationary_step
                del evicted
            self._settle_batches[capacity] = BatchedDiSECt(
                self._scene_path, capacity, sim_dt=self.dt,
                friction_velocity=self._friction_velocity)
        scratch = self._settle_batches.pop(capacity)
        self._settle_batches[capacity] = scratch
        all_rows = torch.ones(capacity, dtype=torch.bool, device=self.position.device)
        padded = torch.cat((indices, indices[-1:].expand(capacity - n)))
        scales = {k: v[padded] for k, v in self.material_scales.items()}
        scratch.set_material_scales(all_rows, **scales)
        state, contact, motion = self.capture_rows(mask)
        def pad(v):
            return torch.cat((v, v[-1:].expand(capacity - n, *v.shape[1:])), 0)
        scratch.restore_rows(all_rows, tuple({k: pad(v) for k, v in group.items()}
                                            for group in (state, contact, motion)))
        peak = scratch.advance_stationary(seconds)
        if bool((peak[:n] > 0).any()):
            raise ValueError('The knife touches the object while it settles; settling needs a free start')
        saved = scratch.capture_rows(all_rows)
        self.restore_rows(mask, tuple({k: v[:n] for k, v in group.items()} for group in saved))

    @torch.no_grad()
    def advance_stationary(self, duration):
        """Same integrator, CUDA graph of 32 substeps, contact checked at every substep.

        Intended only for reset settling with a stationary knife. Does not shorten time,
        clear velocities, alter damage or skip contacts.
        """
        count = round(duration / self.dt)
        if not math.isfinite(duration) or count < 1 or abs(count * self.dt - duration) > 1e-12:
            raise ValueError('Settling duration must contain whole material substeps')
        if bool((self.linear_velocity != 0).any() | (self.angular_velocity != 0).any()):
            raise ValueError('Stationary settling requires zero knife velocity')
        if not hasattr(self, '_settle_graph'):
            # The pybind kernel signature takes a host float. Cache this immutable scalar
            # before capture instead of triggering a GPU-to-host conversion each substep.
            self.model.sdf_radius = float(self.model.sdf_radius)
            self._settle_peak = torch.zeros(self.worlds, device=self.position.device)
            def step():
                q = self.state.joint_q.reshape(self.worlds, -1)
                qd = self.state.joint_qd.reshape(self.worlds, -1)
                q[:, :3] = self.position
                q[:, 3:7] = self.rotation
                qd[:, :6] = 0
                self.state.knife_contact_q.copy_(self.state.particle_q)
                self.state = self.integrator.forward(self.model, self.state, self.dt,
                                                     update_mass_matrix=False)
                now = torch.linalg.vector_norm(self.state_field('knife_f'), dim=-1).sum(-1)
                torch.maximum(self._settle_peak, now, out=self._settle_peak)
            self._stationary_step = step
            mask = torch.ones(self.worlds, dtype=torch.bool, device=self.position.device)
            saved = self.capture_rows(mask)
            # Compile/warm up kernels before capture. Wrappers honor the PyTorch stream.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                step()
            torch.cuda.current_stream().wait_stream(stream)
            self.restore_rows(mask, saved)
            self._settle_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._settle_graph):
                for _ in range(32):
                    step()
            self.restore_rows(mask, saved)
        self._settle_peak.zero_()
        for _ in range(count // 32):
            self._settle_graph.replay()
        for _ in range(count % 32):
            self._stationary_step()
        self.steps += count
        if not bool(torch.isfinite(self.state.particle_q).all() & torch.isfinite(self.state.cut_spring_ke).all()):
            raise FloatingPointError('Nonfinite stationary settling state')
        return self._settle_peak.clone()

    @torch.no_grad()
    def knife_wrench(self):
        """Net contact wrench on the knife in the DiSECt frame, [worlds, 6]: force and moment
        about the DiSECt origin. ``state.knife_f`` holds the force each contact applies to the
        material; the knife receives the opposite force at the contact point on the cut edge."""
        force = self.state.knife_f
        coord = self.model.cut_edge_contact_coord[self._contact_edge][:, None]
        q = self.state.knife_contact_q
        point = q[self._contact_first] * (1. - coord) + q[self._contact_second] * coord
        moment = torch.linalg.cross(point, force)
        return -torch.cat((force.reshape(self.worlds, -1, 3).sum(1),
                           moment.reshape(self.worlds, -1, 3).sum(1)), -1)

    def state_field(self, name):
        value = getattr(self.state, name)
        return value.reshape(self.worlds, -1, *value.shape[1:])

    def save_reset_state(self, mask=None):
        if mask is not None:
            for name, initial in self.initial_state.items():
                initial.reshape(self.worlds, -1)[mask] = getattr(self.state, name).reshape(self.worlds, -1)[mask]
            for name, initial in self.initial_contact.items():
                initial.reshape(self.worlds, -1)[mask] = getattr(self.model, name).reshape(self.worlds, -1)[mask]
            for name, initial in self.initial_motion.items():
                initial[mask] = getattr(self, name)[mask]
            return
        self.initial_state = {k: v.detach().clone() for k, v in vars(self.state).items()
                              if isinstance(v, torch.Tensor)}
        self.initial_contact = {k: getattr(self.model, k).clone() for k in
                                ('cut_edge_contact_coord', 'cut_edge_contact_dist', 'cut_edge_contact_normal')}
        self.initial_motion = {k: getattr(self, k).clone() for k in
                               ('position', 'rotation', 'linear_velocity', 'angular_velocity')}

    @torch.no_grad()
    def set_material_scales(self, mask, **scales):
        """Scale the exported material of the worlds in ``mask``; takes effect at their next reset.

        Each value is a float or a [worlds] tensor relative to the exported scene. The cut springs'
        reset stiffness (the state the integrator weakens) is rescaled by the same factor.
        """
        unknown = set(scales) - set(MATERIAL_SCALES)
        if unknown:
            raise ValueError('Unknown material scales: ' + ', '.join(sorted(unknown)))
        device = self.position.device
        mask = torch.as_tensor(mask, device=device, dtype=torch.bool).expand(self.worlds)
        for key, value in scales.items():
            value = torch.as_tensor(value, device=device, dtype=torch.float32).expand(self.worlds)
            if not bool((torch.isfinite(value) & (value > 0))[mask].all()):
                raise ValueError(f'{key} scale must be positive and finite')
            spec = MATERIAL_SCALES[key]
            if key == 'cut_spring_stiffness':
                reset_ke = self.initial_state['cut_spring_ke'].reshape(self.worlds, -1)
                reset_ke[mask] *= (value / self.material_scales[key])[mask, None]
            for name in spec['scaled']:
                getattr(self.model, name).reshape(self.worlds, -1)[mask] = (
                    self.base_material[name][mask] * value[mask, None])
            for name in spec['inverse']:
                getattr(self.model, name).reshape(self.worlds, -1)[mask] = (
                    self.base_material[name][mask] / value[mask, None])
            self.material_scales[key][mask] = value[mask]

    @torch.no_grad()
    def reset(self, mask):
        for name, initial in self.initial_state.items():
            target = self.state_field(name)
            target[mask] = initial.reshape_as(target)[mask]
        for name, initial in self.initial_contact.items():
            value = getattr(self.model, name).reshape(self.worlds, -1)
            value[mask] = initial.reshape_as(value)[mask]
        for name, initial in self.initial_motion.items():
            getattr(self, name)[mask] = initial[mask]


    def capture_rows(self, mask):
        state = {k: self.state_field(k)[mask].clone() for k, v in vars(self.state).items()
                 if isinstance(v, torch.Tensor)}
        contact = {k: getattr(self.model, k).reshape(self.worlds, -1)[mask].clone()
                   for k in self.initial_contact}
        motion = {k: getattr(self, k)[mask].clone() for k in self.initial_motion}
        return state, contact, motion

    @torch.no_grad()
    def restore_rows(self, mask, saved):
        state, contact, motion = saved
        for k, value in state.items():
            self.state_field(k)[mask] = value
        for k, value in contact.items():
            getattr(self.model, k).reshape(self.worlds, -1)[mask] = value
        for k, value in motion.items():
            getattr(self, k)[mask] = value

    @torch.no_grad()
    def set_motion(self, twist, pose=None, active=None):
        if active is None:
            active = slice(None)
        self.linear_velocity[active] = twist[active, :3]
        self.angular_velocity[active] = twist[active, 3:]
        if pose is not None:
            self.position[active] = pose[active, :3]
            self.rotation[active] = pose[active, 3:]

    @torch.no_grad()
    def advance(self, duration, active=None, track_peak=False):
        """``track_peak``: also return the largest summed contact-force magnitude over every
        substep (not only the last one), as (last, peak)."""
        paused = None
        if active is not None and not bool(active.all()):
            paused = self.capture_rows(~active)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Material advance duration must be finite and positive')
        count = round(duration / self.dt)
        if abs(count * self.dt - duration) > 1e-12:
            raise ValueError('Coupling duration must contain whole DiSECt substeps')
        if count < 1:
            raise ValueError('Material advance needs at least one substep')
        norm = torch.linalg.vector_norm(self.angular_velocity, dim=-1, keepdim=True)
        half = norm * self.dt / 2
        factor = torch.sinc(half / torch.pi) * self.dt / 2
        increment = torch.cat((factor * self.angular_velocity, torch.cos(half)), -1)
        q = self.state.joint_q.reshape(self.worlds, -1)
        qd = self.state.joint_qd.reshape(self.worlds, -1)
        for substep in range(count):
            self.position.add_(self.dt * self.linear_velocity)
            self.rotation.copy_(quat_mul(increment, self.rotation))
            q[:, :3] = self.position
            q[:, 3:7] = self.rotation
            qd[:, :3] = self.linear_velocity
            qd[:, 3:6] = self.angular_velocity
            if substep == count - 1:
                self.state.knife_contact_q.copy_(self.state.particle_q)
            self.state = self.integrator.forward(self.model, self.state, self.dt, update_mass_matrix=False)
            if track_peak:
                now = torch.linalg.vector_norm(self.state_field('knife_f'), dim=-1).sum(-1)
                peak = now if substep == 0 else torch.maximum(peak, now)
        self.steps += count
        if paused is not None:
            self.restore_rows(~active, paused)
        if not torch.isfinite(self.state.particle_q).all() or not torch.isfinite(self.state.cut_spring_ke).all():
            raise FloatingPointError('Nonfinite DiSECt material state')
        last = torch.linalg.vector_norm(self.state_field('knife_f'), dim=-1).sum(-1)
        return (last, peak) if track_peak else last
