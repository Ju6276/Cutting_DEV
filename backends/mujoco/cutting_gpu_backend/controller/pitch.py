"""Opt-in bounded target pitch; independent of the material solver."""
import math
import torch

REVISION = 'bounded_pitch_v1'

def validate_pitch_control(config):
    keys = {'revision', 'enabled', 'max_angle_deg', 'max_rate_deg_s'}
    if not isinstance(config, dict) or set(config) != keys or config['revision'] != REVISION:
        raise ValueError('Specify the bounded_pitch_v1 settings explicitly')
    if type(config['enabled']) is not bool:
        raise ValueError('Pitch enabled must be a boolean')
    for name in ('max_angle_deg', 'max_rate_deg_s'):
        v = config[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError('Pitch angle/rate bounds must be positive finite numbers')
    if config['max_angle_deg'] >= 90:
        raise ValueError('Pitch range must be smaller than 90 degrees')
    return dict(config)

def advance_pitch(previous, action, config, dt):
    """Pitch about base Y, relative to the fixed task target, in radians.

    Positive sixth action reduces pitch, matching the inherited mapper sign.
    dt is the *agent* interval, not the robot's inner integration interval.
    """
    if not config['enabled']:
        return torch.zeros_like(previous)
    limit = math.radians(config['max_angle_deg'])
    increment = -action.clamp(-1, 1) * math.radians(config['max_rate_deg_s']) * dt
    return (previous + increment).clamp(-limit, limit)
