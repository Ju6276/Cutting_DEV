"""Versioned free-object single-cut tasks (v4 currently physical evaluation only).

Weights are explicit experiment inputs. Pose variants reuse arrival semantics; pose_depth_v2/v3 use geometric progress.
Historical pose and separation variants retain their recorded running rewards.
"""
import math
import numpy as np
import torch

REVISION = 'free_cut_v3'
SUPPORTED_REVISIONS = ('free_cut_v1', 'free_cut_v2', REVISION, 'free_cut_v4', 'free_cut_pose_v1', 'blademaster_pose_v1', 'pose_depth_v2', 'pose_depth_v3')
# pose_depth_v3 inherits every pose_depth_v2 rule and adds only a software board-stop
# boundary consistency fix (latched-state fields, post-latch peak rule, loss masking,
# analytic imagined survival). Checks of the shared semantics use this family.
POSE_DEPTH_VERSIONS = ('pose_depth_v2', 'pose_depth_v3')
# The single current task version. pose_depth_v2 remains readable/evaluable only, for
# traceability of the 2026-11-16 runs; it is not a separate task choice.
CURRENT_POSE_DEPTH = 'pose_depth_v3'
RETIRED_TRAINING = {'pose_depth_v2': (
    'pose_depth_v2 is kept only to read/evaluate the 2026-11-16 runs. New training uses '
    'pose_depth_v3 (same task rules plus the software board-stop consistency fix and '
    'analytic imagined survival): configs/shared/scene_v1.json. Continue the 2026-11-16 '
    'runs only from their frozen checkout 5ec9f85.')}


def refuse_retired_training(task):
    """Raise before any training launch or resume on a read/evaluate-only task version."""
    version = (task or {}).get('version')
    if version in RETIRED_TRAINING:
        raise ValueError(RETIRED_TRAINING[version])
WEIGHTS = ('progress', 'speed', 'load', 'jerk', 'success', 'failure')


def validate_task(config):
    if not isinstance(config, dict) or config.get('version') not in SUPPORTED_REVISIONS:
        raise ValueError('Unsupported free-cut task version')
    required = {'version', 'horizon_steps', 'weights'}
    if 'scene_alignment' in config:
        from .scene_alignment import validate_alignment
        validate_alignment(config['scene_alignment'])
        if config['version'] not in POSE_DEPTH_VERSIONS or config.get('tool_revision')!='disect_knife_rigid_mount_v1':
            raise ValueError('Aligned reset requires the shared rigid knife task')
        required.add('scene_alignment')
    if 'pitch_control' in config:
        from cutting_gpu_backend.controller.pitch import validate_pitch_control
        validate_pitch_control(config['pitch_control'])
        if config['version'] not in POSE_DEPTH_VERSIONS:
            raise ValueError('Bounded pitch requires pose_depth_v2/v3')
        required.add('pitch_control')
    if 'completion_position' in config:
        position = config['completion_position']
        if (config['version'] not in POSE_DEPTH_VERSIONS or not isinstance(position, dict)
                or set(position) != {'revision', 'radius_m'}
                or position['revision'] != 'robot_base_xy_goal_v1'
                or isinstance(position['radius_m'], bool)
                or not isinstance(position['radius_m'], (int, float))
                or not math.isfinite(position['radius_m']) or position['radius_m'] <= 0):
            raise ValueError('Explicit positive final XY radius required for pose_depth_v2/v3')
        required.add('completion_position')
    if 'tool_motion' in config:
        from .motion_limits import validate_limits
        motion = config['tool_motion']
        if (config['version'] not in POSE_DEPTH_VERSIONS or 'completion_position' not in config
                or not isinstance(motion, dict)
                or set(motion) != {'revision', 'enabled', 'free_m', 'failure_m', 'weight'}
                or motion['revision'] != 'robot_base_xy_goal_v1'
                or type(motion['enabled']) is not bool):
            raise ValueError('Explicit optional tool-motion constraint required')
        validate_limits(motion)
        required.add('tool_motion')
    if 'object_slip' in config:
        from .object_slip import validate
        validate(config['object_slip'])
        if config['version'] not in POSE_DEPTH_VERSIONS:
            raise ValueError('Object motion constraint requires pose_depth_v2/v3')
        required.add('object_slip')
    if 'tool_revision' in config:
        if config['tool_revision'] != 'disect_knife_rigid_mount_v1' or config['version'] not in POSE_DEPTH_VERSIONS:
            raise ValueError('Shared knife requires pose_depth_v2/v3')
        required.add('tool_revision')
    if config['version'] != 'free_cut_v1':
        required |= {'gamma', 'workspace'}
        if config['version'] not in POSE_DEPTH_VERSIONS:
            required.add('initial_progress_fraction')
        if config.get('gamma') != 1.0:
            raise ValueError('This free-cut version is an undiscounted finite-horizon task (gamma=1)')
        fraction = config.get('initial_progress_fraction', .001 if config['version'] in POSE_DEPTH_VERSIONS else 0)
        if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError('Initial progress fraction must be strictly between 0 and 1')
        box = config.get('workspace', {})
        if set(box) != {'frame', 'point', 'lower', 'upper'} or box['frame'] != 'robot_base' or box['point'] != 'ee':
            raise ValueError('Specify workspace for the EE in the robot_base frame')
        lo, hi = np.asarray(box['lower']), np.asarray(box['upper'])
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or not (lo < hi).all():
            raise ValueError('Workspace needs finite ordered XYZ bounds')
    if config['version'] in ('free_cut_v4', 'free_cut_pose_v1', 'blademaster_pose_v1') + POSE_DEPTH_VERSIONS:
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
        if self.config['version'] in ('free_cut_v3', 'free_cut_v4', 'free_cut_pose_v1', 'blademaster_pose_v1') + POSE_DEPTH_VERSIONS:
            # Over one finite episode: sum positive delta-rho <= 1 and sum dt/T <= 1.
            # Enforce settlement dominance for every accepted configuration, not one probe.
            penalty_bound = (2 * self.weights['speed'] + self.horizon * self.dt *
                             (self.weights['load'] + self.weights['jerk'])
                             + self.config.get('object_slip', {}).get('weight', 0)
                             + (self.config.get('tool_motion', {}).get('weight', 0)
                                if self.config.get('tool_motion', {}).get('enabled') else 0))
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

    def completion_distance(self, pose, target_pose):
        if pose is None or target_pose is None:
            raise ValueError('Final position criterion requires EE and target poses')
        return torch.linalg.vector_norm(pose[..., :2] - target_pose[..., :2], dim=-1)

    def arrival_mask(self, pose, target_pose):
        config = self.config.get('completion_position')
        if config is None:
            return None
        return self.completion_inside(self.completion_distance(pose, target_pose))

    def completion_inside(self, distance):
        from .motion_limits import inside
        return inside(distance, self.config['completion_position']['radius_m'])

    def tool_motion_failure(self, distance, like):
        config = self.config.get('tool_motion')
        if config is None or not config['enabled']:
            return torch.zeros_like(like, dtype=torch.bool)
        if distance is None:
            raise ValueError('Tool-motion constraint requires robot-derived distance')
        from .motion_limits import outside
        return outside(distance, config['failure_m'])

    def slip_failure(self, distance, like):
        config = self.config.get('object_slip')
        if config is None:
            return torch.zeros_like(like, dtype=torch.bool)
        if distance is None:
            raise ValueError('Object motion task requires measured/predicted object_slip')
        return distance > config['failure_m']

    def events(self, observation, peak, progress, step_count, collision=None, pose=None, board=None, success_counter=None, object_slip=None, target_pose=None):
        limits = peak.new_tensor(self.heads.force_limits)
        force = (peak.abs() > limits).any(-1)
        if collision is not None:
            force = force | collision
        outside = self.workspace(observation, pose)
        separated = None if progress is None else progress >= 1
        # pose_depth_v2/v3 receive physical separation only as a diagnostic in the
        # DiSECt adapter; geometric progress never determines this label.
        if self.config['version'] in ('free_cut_pose_v1', 'blademaster_pose_v1') + POSE_DEPTH_VERSIONS:
            if success_counter is None or board is None:
                raise ValueError('Pose task requires arrival counter and board measurements')
            if bool((board['knife_min_board_gap'][..., 0] < -2e-6).any()):
                raise FloatingPointError('Invalid knife-board penetration, not an RL failure')
            _, reached, _, counter = self.heads.termination(observation, success_counter, force, self.arrival_mask(pose, target_pose))
            slipped = self.slip_failure(object_slip, force)
            tool_error = self.completion_distance(pose,target_pose) if 'completion_position' in self.config else None
            tool_outside = self.tool_motion_failure(tool_error,force)
            success = reached & ~outside & ~force & ~slipped & ~tool_outside
            timeout = (step_count >= self.horizon) & ~force & ~outside & ~slipped & ~tool_outside & ~success
            return dict(**({} if separated is None else {'separated': separated}),
                        **({'object_slip_failure': slipped, 'object_slip_m': object_slip}
                           if 'object_slip' in self.config else {}),
                        **({'completion_xy_error_m': tool_error, 'tool_motion_failure': tool_outside}
                           if 'completion_position' in self.config else {}),
                        collision=force, outside=outside,
                        success=success, timeout=timeout, counter=counter,
                        board_contact=board['knife_board_contact'][..., 0].bool(),
                        terminated=force | outside | slipped | tool_outside | success | timeout)
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

    def step_reward(self, previous_progress, progress, observation, peak, target_speed, object_slip=None, tool_distance=None):
        if target_speed is None:
            raise ValueError('free_cut requires a commanded feed speed')
        w = self.weights
        speed = self.heads.cutting_speed(observation)
        # The uncoupled candidate: no relief for slowing down under high load.
        mismatch = torch.tanh((speed - target_speed).abs() / target_speed)
        utilization = (peak.abs() / peak.new_tensor(self.heads.force_limits)).amax(-1)
        jerk = self.heads.jerkiness(observation).clamp(max=1)
        increment = progress - previous_progress
        if self.config['version'] in ('free_cut_v3', 'free_cut_v4', 'free_cut_pose_v1', 'blademaster_pose_v1') + POSE_DEPTH_VERSIONS:
            # Symmetric relative mismatch: reciprocal speeds v/v* and v*/v have
            # equal cost per progress increment. Reverse motion has maximum cost.
            mismatch = ((speed - target_speed) / (speed.abs() + target_speed)).square()
            # Progress prevents fast completion evading cost; finite-time cost penalizes
            # waiting/retraction without letting a long horizon defeat failure settlement.
            speed_cost = w['speed'] * mismatch * (increment.clamp_min(0) + 1 / self.horizon)
        else:
            speed_cost = self.dt * w['speed'] * mismatch
        slip_cost = 0.
        if 'object_slip' in self.config:
            if object_slip is None:
                raise ValueError('Object motion reward requires object_slip')
            from .object_slip import penalty
            slip_cost = penalty(object_slip, self.config['object_slip'], self.horizon)
        tool_cost = 0.
        motion = self.config.get('tool_motion')
        if motion is not None and motion['enabled']:
            if tool_distance is None:
                raise ValueError('Tool-motion reward requires robot-derived distance')
            from .motion_limits import penalty
            tool_cost = penalty(tool_distance, motion, self.horizon)
        return (w['progress'] * increment - speed_cost - slip_cost - tool_cost
                - self.dt * (w['load'] * utilization.clamp(max=1) + w['jerk'] * jerk))

    def reward(self, previous_progress, progress, observation, peak, target_speed, events):
        if self.config['version'] == 'free_cut_v4' and bool(events['board_penetration'].any()):
            raise FloatingPointError('Invalid knife-board penetration: exclude transition, not an RL failure penalty')
        result = self.step_reward(previous_progress, progress, observation, peak, target_speed, events.get('object_slip_m'), events.get('completion_xy_error_m'))
        failure = events['collision'] | events['outside'] | events.get('object_slip_failure', False) | events.get('tool_motion_failure', False)
        return (result + self.weights['success'] * events['success'].to(result.dtype)
                - self.weights['failure'] * failure.to(result.dtype))

