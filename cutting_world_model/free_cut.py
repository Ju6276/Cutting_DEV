"""Versioned free-object single-cut tasks (v4 currently physical evaluation only).

Weights are explicit experiment inputs. The pose variant reuses arrival semantics
while preserving v3 running costs; historical separation variants remain unchanged.
"""
import math
import numpy as np
import torch

REVISION = 'free_cut_v3'
SUPPORTED_REVISIONS = ('free_cut_v1', 'free_cut_v2', REVISION, 'free_cut_v4', 'free_cut_pose_v1')
WEIGHTS = ('progress', 'speed', 'load', 'jerk', 'success', 'failure')


def validate_task(config):
    if not isinstance(config, dict) or config.get('version') not in SUPPORTED_REVISIONS:
        raise ValueError('Unsupported free-cut task version')
    required = {'version', 'horizon_steps', 'weights'}
    if config['version'] != 'free_cut_v1':
        required |= {'gamma', 'initial_progress_fraction', 'workspace'}
        if config.get('gamma') != 1.0:
            raise ValueError('This free-cut version is an undiscounted finite-horizon task (gamma=1)')
        fraction = config.get('initial_progress_fraction', 0)
        if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError('Initial progress fraction must be strictly between 0 and 1')
        box = config.get('workspace', {})
        if set(box) != {'frame', 'point', 'lower', 'upper'} or box['frame'] != 'robot_base' or box['point'] != 'ee':
            raise ValueError('Specify workspace for the EE in the robot_base frame')
        lo, hi = np.asarray(box['lower']), np.asarray(box['upper'])
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or not (lo < hi).all():
            raise ValueError('Workspace needs finite ordered XYZ bounds')
    if config['version'] in ('free_cut_v4', 'free_cut_pose_v1'):
        required.add('board')
        board = config.get('board', {})
        expected = {'height_m', 'stop_gap_m'}
        if config['version'] == 'free_cut_v4':
            expected.add('coverage_tolerance_m')
        if set(board) != expected:
            raise ValueError('Board fields do not match task version')
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in board.values()):
            raise ValueError('Board settings must be finite')
        tolerance = board.get('coverage_tolerance_m', .001)
        if board['height_m'] != 0 or not 0 < board['stop_gap_m'] < tolerance <= .001:
            raise ValueError('Current material ground is y=0; require 0 < stop gap < coverage tolerance <= 1 mm')
    if set(config) != required:
        raise ValueError('Free-cut task fields do not match its version')
    if type(config['horizon_steps']) is not int or config['horizon_steps'] < 1:
        raise ValueError('horizon_steps must be a positive integer')
    weights = config['weights']
    if set(weights) != set(WEIGHTS):
        raise ValueError('Specify all six free_cut weights explicitly')
    if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError('Free-cut weights must be finite and nonnegative')
    if weights['success'] <= 0 or weights['failure'] <= 0:
        raise ValueError('Success and failure magnitudes must be positive')
    return {**config, 'weights': dict(weights)}


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
        if self.config['version'] in ('free_cut_v3', 'free_cut_v4', 'free_cut_pose_v1'):
            # Over one finite episode: sum positive delta-rho <= 1 and sum dt/T <= 1.
            # Enforce settlement dominance for every accepted configuration, not one probe.
            penalty_bound = (2 * self.weights['speed'] + self.horizon * self.dt *
                             (self.weights['load'] + self.weights['jerk']))
            if min(self.weights['success'], self.weights['failure']) <= self.weights['progress'] + penalty_bound:
                raise ValueError('Terminal rewards must exceed progress plus the finite running-penalty bound')

    @property
    def observes_time(self):
        return self.config['version'] != 'free_cut_v1'

    def observation(self, observation, step_count):
        if not self.observes_time:
            return observation
        remaining = (1 - step_count.to(observation.dtype) / self.horizon).clamp(0, 1)
        return torch.cat((observation, remaining[..., None]), -1)

    def workspace_features(self, observation, pose=None):
        if self.observes_time:
            if pose is None:
                raise ValueError('Free-cut workspace requires the robot-base EE pose')
            box = self.config['workspace']
            lo, hi = pose.new_tensor(box['lower']), pose.new_tensor(box['upper'])
            return ((pose[..., :3] - (lo + hi) / 2).abs() / ((hi - lo) / 2) - 1)
        return ((observation[..., :3] * observation.new_tensor(self.heads.max_distance)).abs()
                / observation.new_tensor(self.heads.env['workspace_limit']) - 1)

    def workspace(self, observation, pose=None):
        if self.observes_time:
            return (self.workspace_features(observation, pose) > 0).any(-1)
        error = (observation[..., :len(self.heads.dims)] *
                 observation.new_tensor(self.heads.max_distance)).abs()
        return (error > error.new_tensor(self.heads.env['workspace_limit'])).any(-1)

    def events(self, observation, peak, progress, step_count, collision=None, pose=None, board=None, success_counter=None):
        limits = peak.new_tensor(self.heads.force_limits)
        force = (peak.abs() > limits).any(-1)
        if collision is not None:
            force = force | collision
        outside = self.workspace(observation, pose)
        separated = progress >= 1
        if self.config['version'] == 'free_cut_pose_v1':
            if success_counter is None or board is None:
                raise ValueError('Pose task requires arrival counter and board measurements')
            if bool((board['knife_min_board_gap'][..., 0] < -2e-6).any()):
                raise FloatingPointError('Invalid knife-board penetration, not an RL failure')
            _, reached, _, counter = self.heads.termination(observation, success_counter, force)
            success = reached & ~outside & ~force
            timeout = (step_count >= self.horizon) & ~force & ~outside & ~success
            return dict(separated=separated, collision=force, outside=outside,
                        success=success, timeout=timeout, counter=counter,
                        board_contact=board['knife_board_contact'][..., 0].bool(),
                        terminated=force | outside | success | timeout)
        if self.config['version'] == 'free_cut_v4':
            if board is None:
                raise ValueError('v4 requires physical knife-board and coverage measurements')
            contact = board['knife_board_contact'][:, 0].bool()
            covered = board['knife_path_complete'][:, 0].bool()
            penetrating = board['knife_min_board_gap'][:, 0] < -2e-6
            success = contact & covered & ~force & ~outside & ~penetrating
            incomplete = contact & ~covered & ~force & ~outside & ~penetrating
            timeout = (step_count >= self.horizon) & ~force & ~outside & ~contact & ~penetrating
            return dict(separated=separated, collision=force, outside=outside,
                        success=success, timeout=timeout, board_contact=contact,
                        board_incomplete=incomplete, board_penetration=penetrating,
                        terminated=force | outside | contact | penetrating | timeout)
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
        increment = progress - previous_progress
        if self.config['version'] in ('free_cut_v3', 'free_cut_v4', 'free_cut_pose_v1'):
            # Symmetric relative mismatch: reciprocal speeds v/v* and v*/v have
            # equal cost per released fraction. Reverse motion has maximum cost.
            mismatch = ((speed - target_speed) / (speed.abs() + target_speed)).square()
            # Progress prevents fast completion evading cost; finite-time cost penalizes
            # waiting/retraction without letting a long horizon defeat failure settlement.
            speed_cost = w['speed'] * mismatch * (increment.clamp_min(0) + 1 / self.horizon)
        else:
            speed_cost = self.dt * w['speed'] * mismatch
        return (w['progress'] * increment - speed_cost
                - self.dt * (w['load'] * utilization.clamp(max=1) + w['jerk'] * jerk))

    def reward(self, previous_progress, progress, observation, peak, target_speed, events):
        if self.config['version'] == 'free_cut_v4' and bool(events['board_penetration'].any()):
            raise FloatingPointError('Invalid knife-board penetration: exclude transition, not an RL failure penalty')
        result = self.step_reward(previous_progress, progress, observation, peak, target_speed)
        failure = events['collision'] | events['outside']
        return (result + self.weights['success'] * events['success'].to(result.dtype)
                - self.weights['failure'] * failure.to(result.dtype))

