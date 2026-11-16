"""Finite-episode displacement costs, independent of the measured moving entity."""
import math
import torch


def validate_limits(config):
    for key in ('free_m', 'failure_m', 'weight'):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Motion settings must be finite numbers')
    if not 0 <= config['free_m'] < config['failure_m'] or config['weight'] < 0:
        raise ValueError('Require 0 <= free_m < failure_m and nonnegative weight')


def penalty(distance, config, horizon):
    excess = ((distance - config['free_m']) /
              (config['failure_m'] - config['free_m'])).clamp(0, 1)
    return config['weight'] / horizon * excess.square()


def boundary_tolerance(distance, limit):
    # Positions are subtracted in metre-scale robot frames before taking a norm.
    return 8 * torch.finfo(distance.dtype).eps * max(1., abs(limit))


def inside(distance, limit):
    return distance < limit - boundary_tolerance(distance, limit)


def outside(distance, limit):
    return distance > limit + boundary_tolerance(distance, limit)
