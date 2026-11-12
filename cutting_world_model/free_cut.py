"""Versioned free-object single-cut task, shared by physics and imagination.

Weights are explicit experiment inputs. No legacy pose-success reward is reused.
"""
import math
import numpy as np
import torch

REVISION = 'free_cut_v1'
WEIGHTS = ('progress', 'speed', 'load', 'jerk', 'success', 'failure')


def validate_task(config):
    if not isinstance(config, dict) or config.get('version') != REVISION:
        raise ValueError('free_cut task requires version free_cut_v1')
    if set(config) != {'version', 'horizon_steps', 'weights'}:
        raise ValueError('free_cut requires only version, horizon_steps and weights')
    if type(config['horizon_steps']) is not int or config['horizon_steps'] < 1:
        raise ValueError('horizon_steps must be a positive integer')
    weights = config['weights']
    if set(weights) != set(WEIGHTS):
        raise ValueError('Specify all six free_cut weights explicitly')
    if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError('Free-cut weights must be finite and nonnegative')
    if weights['success'] <= 0 or weights['failure'] <= 0:
        raise ValueError('Success and failure magnitudes must be positive')
    return {'version': REVISION, 'horizon_steps': config['horizon_steps'], 'weights': dict(weights)}


def validate_topology(material):
    """Supported assets: two FEM components joined exclusively by undamped cut springs.
    This is a structural check, not a claim of calibrated contact/fracture physics.
    """
    model = material.model
    n = model.particles_per_world
    tets = model.tet_indices.reshape(-1, 4).detach().cpu().numpy()
    tets = tets[(tets < n).all(1)]
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for tet in tets:
        a = root(int(tet[0]))
        for b in tet[1:]:
            parent[root(int(b))] = a
    labels = np.array([root(i) for i in range(n)])
    edges = model.cut_edge_indices.reshape(-1, 2).detach().cpu().numpy()
    springs = model.cut_spring_indices.reshape(material.worlds, -1, 2)[0].detach().cpu().numpy()
    if len(np.unique(labels)) != 2 or not len(springs):
        raise ValueError('free_cut needs exactly two FEM components and nonempty cut springs')
    sides = labels[edges[springs]]
    if not np.all(sides[:, :, 0] == sides[:, :, 1]) or not np.all(sides[:, 0, 0] != sides[:, 1, 0]):
        raise ValueError('Every cut spring must join the two FEM components')
    if bool((model.particle_inv_mass == 0).any()):
        raise ValueError('free_cut requires free objects (no fixed particles)')
    if bool((material.state.cut_spring_kd != 0).any()):
        raise ValueError('Zero spring stiffness is not separation with residual cut-spring damping')


class FreeCutTask:
    def __init__(self, config, heads):
        self.config = validate_task(config)
        self.heads = heads
        self.horizon = self.config['horizon_steps']
        self.weights = self.config['weights']
        self.dt = heads.env['agent_control_dt']

    def workspace(self, observation):
        error = (observation[..., :len(self.heads.dims)] *
                 observation.new_tensor(self.heads.max_distance)).abs()
        return (error > error.new_tensor(self.heads.env['workspace_limit'])).any(-1)

    def events(self, observation, peak, progress, step_count, collision=None):
        limits = peak.new_tensor(self.heads.force_limits)
        force = (peak.abs() > limits).any(-1)
        if collision is not None:
            force = force | collision
        outside = self.workspace(observation)
        separated = progress >= 1
        success = separated & ~force & ~outside
        # Precedence: force/workspace failure > completion > timeout.
        timeout = (step_count >= self.horizon) & ~force & ~outside & ~success
        return dict(separated=separated, collision=force, outside=outside,
                    success=success, timeout=timeout,
                    terminated=force | outside | success | timeout)

    def step_reward(self, previous_progress, progress, observation, peak, target_speed):
        if target_speed is None:
            raise ValueError('free_cut requires a commanded feed speed')
        w = self.weights
        speed = self.heads.cutting_speed(observation)
        # The uncoupled candidate: no relief for slowing down under high load.
        mismatch = torch.tanh((speed - target_speed).abs() / target_speed)
        utilization = (peak.abs() / peak.new_tensor(self.heads.force_limits)).amax(-1)
        jerk = self.heads.jerkiness(observation).clamp(max=1)
        return (w['progress'] * (progress - previous_progress)
                - self.dt * (w['speed'] * mismatch + w['load'] * utilization.clamp(max=1)
                             + w['jerk'] * jerk))

    def reward(self, previous_progress, progress, observation, peak, target_speed, events):
        result = self.step_reward(previous_progress, progress, observation, peak, target_speed)
        failure = events['collision'] | events['outside']
        return (result + self.weights['success'] * events['success'].to(result.dtype)
                - self.weights['failure'] * failure.to(result.dtype))

