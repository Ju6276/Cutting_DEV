"""Read-only, opt-in instrumentation of DiSECt's exact material damage launch.

Installed only inside the platform-check process, after settling. Inputs are
observed immediately before eval_cut_springs and stiffness immediately after it,
before particle integration. No tensors used by physics are modified.
"""
import numpy as np
import torch


def damage_inputs(inputs):
    x, _, edges, coords, springs, rest, stiffness, _, _, _, softness, knife_f = inputs
    points = edges.long()[springs.long().reshape(-1, 2)]
    u = coords[springs.long().reshape(-1, 2)][..., None]
    endpoints = (1 - u) * x[points[..., 0]] + u * x[points[..., 1]]
    stretch = (torch.linalg.vector_norm(endpoints[:, 0] - endpoints[:, 1], dim=-1) - rest).clamp_min(0)
    force = torch.linalg.vector_norm(knife_f.reshape(-1, 2, 3), dim=-1).sum(-1)
    contact = force != 0
    before = stiffness.clone()
    expected = (before - contact * stretch * softness).clamp_min(0)
    return before, expected, contact, stretch, force


class DamageTrace:
    def __init__(self, material):
        self.material = material
        self.worlds = material.worlds
        self.dt = material.dt
        self.phase = 'rest'
        self.stats = {}
        self.steps = 0
        self.snapshots = []
        self.initial = material.state.cut_spring_ke.clone()
        self.final = self.initial.clone()

    def __enter__(self):
        import disect.dflex.adjoint as df
        self.tape_class = df.Tape
        self.original = df.Tape.launch
        tracer = self

        def observed_launch(tape, func, dim, inputs, outputs, adapter, *args, **kwargs):
            is_damage = getattr(getattr(func, 'func', None), '__name__', '') == 'eval_cut_springs'
            # Only this model's material launches; no global/background collectors.
            is_damage = is_damage and inputs[6].data_ptr() == tracer.material.state.cut_spring_ke.data_ptr()
            values = damage_inputs(inputs) if is_damage else None
            result = tracer.original(tape, func, dim, inputs, outputs, adapter, *args, **kwargs)
            if is_damage:
                tracer.record(*values, outputs[-1])
            return result

        self.tape_class.launch = observed_launch
        return self

    def __exit__(self, exc_type, exc, tb):
        self.tape_class.launch = self.original

    @torch.no_grad()
    def record(self, before, expected, contact, stretch, force, after):
        self.steps += 1
        if self.phase not in self.stats:
            zeros = torch.zeros_like(before, dtype=torch.float64)
            counts = torch.zeros_like(before, dtype=torch.int64)
            self.stats[self.phase] = {
                **{k: counts.clone() for k in ('contact_steps', 'stretch_steps', 'eligible_steps', 'damage_steps',
                                               'last_contact_step', 'last_damage_step')},
                **{k: zeros.clone() for k in ('actual_loss', 'expected_loss', 'max_formula_error',
                                              'max_positive_stretch_m', 'max_stretch_in_contact_m',
                                              'max_contact_force_n')}}
        s = self.stats[self.phase]
        change = before.double() - after.double()
        damaged = change > 0
        eligible = contact & (stretch > 0) & (before > 0)
        for key, mask in [('contact_steps', contact), ('stretch_steps', stretch > 0),
                          ('eligible_steps', eligible), ('damage_steps', damaged)]:
            s[key].add_(mask)
        for key, mask in [('last_contact_step', contact), ('last_damage_step', damaged)]:
            s[key].copy_(torch.where(mask, self.steps, s[key]))
        s['actual_loss'].add_(change)
        s['expected_loss'].add_(before.double() - expected.double())
        for key, value in [('max_formula_error', (expected.double() - after.double()).abs()),
                           ('max_positive_stretch_m', stretch), ('max_stretch_in_contact_m', stretch * contact),
                           ('max_contact_force_n', force)]:
            s[key].copy_(torch.maximum(s[key], value.double()))
        self.final.copy_(after)

    def snapshot(self, sim_time):
        """One control-step record for locating when material damage stopped."""
        self.snapshots.append((sim_time.detach().cpu().numpy().copy(), self.phase,
                               self.final.detach().cpu().numpy().copy()))

    def save(self, path):
        if self.steps == 0:
            raise RuntimeError('Damage tracer observed no material kernel launches')
        arrays = {'initial_stiffness': self.initial.cpu().numpy().reshape(self.worlds, -1),
                  'final_stiffness': self.final.cpu().numpy().reshape(self.worlds, -1)}
        for phase, stats in self.stats.items():
            arrays.update({phase + '__' + k: v.cpu().numpy().reshape(self.worlds, -1) for k, v in stats.items()})
        arrays.update(snapshot_time_s=np.asarray([v[0] for v in self.snapshots]),
                      snapshot_phase=np.asarray([v[1] for v in self.snapshots]),
                      snapshot_stiffness=np.asarray([v[2] for v in self.snapshots]).reshape(
                          len(self.snapshots), self.worlds, -1))
        np.savez_compressed(path, **arrays)
        return {'file': path.name, 'material_substeps': self.steps, 'material_dt_s': self.dt,
                'scope': 'Every eval_cut_springs launch after settling; no physics writes',
                'max_formula_error': max(float(s['max_formula_error'].max()) for s in self.stats.values()),
                'phases': list(self.stats)}
