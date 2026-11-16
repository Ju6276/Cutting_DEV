"""Optional horizontal material-motion constraint, shared by both backends.

RMS relative to settled reset prevents cancellation between separating fragments.
It also includes rotation/deformation: it is not a rigid-body slip estimator.
"""
import math
import torch

REVISION = 'planar_mass_rms_v1'

def validate(config):
    if not isinstance(config, dict) or set(config) != {'revision', 'free_m', 'failure_m', 'weight'}:
        raise ValueError('Specify revision, free_m, failure_m and weight for object_slip')
    if config['revision'] != REVISION:
        raise ValueError('Unsupported object motion metric')
    for key in ('free_m', 'failure_m', 'weight'):
        v = config[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError('Object motion settings must be finite numbers')
    if not 0 <= config['free_m'] < config['failure_m'] or config['weight'] < 0:
        raise ValueError('Require 0 <= free_m < failure_m and nonnegative weight')
    return dict(config)

def planar_rms(position, reference, mass, axes):
    displacement = (position - reference)[..., list(axes)]
    weights = mass / mass.sum(-1, keepdim=True)
    return (displacement.square().sum(-1) * weights).sum(-1).clamp_min(0).sqrt()

def penalty(distance, config, horizon):
    excess = ((distance - config['free_m']) /
              (config['failure_m'] - config['free_m'])).clamp(0, 1)
    return config['weight'] / horizon * excess.square()

@torch.no_grad()
def observe(env, frame):
    """Read particles without changing physics; reset only selected reference clouds."""
    if 'object_slip' not in env.metadata.get('free_cut_task', {}):
        return
    reset = env.step_count == 0
    if env.metadata['physical_backend'] == 'mujoco_warp_disect':
        position = frame['particle_q'].reshape(env.worlds, -1, 3)
        mass = env.physics.material.model.particle_mass.reshape(env.worlds, -1).to(position)
        reference = getattr(env, '_object_reference', None)
        if reference is None:
            reference = position.clone()
        reference[reset] = position[reset]
        env._object_reference = reference
        frame['object_reference_q'] = reference.clone()
        distance = planar_rms(position, reference, mass, (0, 2))
    else:
        import warp as wp
        groups = env.physics.batch_groups
        references = getattr(env, '_object_references', {})
        distance = frame['joint_q'].new_zeros(env.worlds)
        if groups:
            clouds = [(indices, wp.to_torch(solver.x.data)) for indices, solver in groups]
        else:
            clouds = [(slice(i, i+1), wp.to_torch(solver.x)[None])
                      for i, solver in enumerate(env.physics.solvers)]
        for indices, position in clouds:
            key = indices.start
            reference = references.get(key)
            if reference is None:
                reference = position.clone()
            reference[reset[indices]] = position[reset[indices]]
            references[key] = reference
            # BladeMaster uses equal-volume, equal-mass particles within each object.
            mass = position.new_ones(position.shape[:-1])
            distance[indices] = planar_rms(position, reference, mass, (0, 1)).to(distance)
        env._object_references = references
    if not bool(torch.isfinite(distance).all()):
        raise FloatingPointError('Nonfinite material displacement')
    frame['object_slip'] = distance[:, None]
