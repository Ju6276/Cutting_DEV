"""Task-specific StateSpec for the source residual dynamics and analytic heads."""
import numpy as np
import torch
from torch import nn

from .kinematics import (SerialKinematics, quat_exp, quat_mul, pose_difference,
                         observation_velocity)
from .task_heads import SliceItHeads, pose_error


class CuttingStateSpec(nn.Module):
    physical_fields = ('particle_q', 'particle_qd', 'cut_spring_ke', 'joint_q', 'joint_qd',
                       'knife_pose', 'knife_velocity', 'wrench_history', 'peak_wrench')
    history_fields = ('motion_command', 'previous_ee_pose', 'previous_velocity',
                      'previous_acceleration', 'observation', 'success_counter')

    recurrent_fields = ('controller_q', 'controller_velocity', 'controller_previous_error',
                        'controller_previous_position_command', 'controller_cached_tip_pose',
                        'filter_history', 'filter_zi', 'wrench_queue', 'robot_warmstart',
                        'coupling_previous_pose', 'coupling_initialized')
    recurrent_constants = ('controller_tare_compensation', 'filter_offset')
    native_recurrent_fields = (
        'controller_measured_q', 'controller_measured_qd', 'controller_q', 'controller_velocity',
        'controller_previous_error', 'controller_previous_position_command',
        'controller_cached_tip_pose', 'controller_target_pose',
        'controller_sensor_wrench', 'controller_target_wrench',
        'filter_history', 'filter_zi', 'filter_history_count', 'filter_initialized',
        'filter_added_wrench', 'filter_flags', 'filter_added_age', 'filter_raw_age',
        'controller_sample_age', 'filter_sample_age',
        'wrench_queue', 'wrench_queue_count', 'current_ft_value',
        'coupling_previous_pose', 'coupling_initialized')
    pose_fields = ('knife_pose', 'controller_cached_tip_pose', 'coupling_previous_pose',
                   'controller_target_pose')

    @classmethod
    def fields_for(cls, metadata):
        layout = metadata.get('state_layout')
        if layout == 'native_ros_recurrent_v1':
            recurrent = cls.native_recurrent_fields
        elif layout == 'coupled_recurrent_v1':
            recurrent = cls.recurrent_fields
        else:
            recurrent = ()
        return (cls.physical_fields + recurrent,
                cls.history_fields + (cls.recurrent_constants if recurrent else ()))

    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata
        self.physical_fields, self.history_fields = self.fields_for(metadata)
        self.heads = SliceItHeads(metadata['environment'], metadata['controller'])
        self.dt = metadata['environment']['agent_control_dt']
        self.fk = SerialKinematics(metadata['robot_xml'], metadata['robot_base'],
                                   metadata['robot_tip'], metadata['joint_names'])
        self.layout, self.shapes = {}, metadata['state_shapes']
        offset = 0
        for name in self.physical_fields + self.history_fields:
            size = int(np.prod(self.shapes[name]))
            self.layout[name] = slice(offset, offset + size)
            offset += size
        self.state_dim = offset
        self.delta_slices = {}
        scales_in, scales_error, scales_output = [], [], []
        delta_offset = 0
        force = np.asarray(self.heads.force_limits)
        spring = np.asarray(metadata.get('spring_reference_ke', []))
        if 'cut_spring_ke' in self.physical_fields and (not spring.size or np.any(spring <= 0)):
            raise ValueError('Reference spring stiffness must be positive')
        source = {
            'particle_q': (1., .05, .01), 'particle_qd': (1., .5, .3),
            'cut_spring_ke': (spring, spring, spring),
            'joint_q': (1., .1, .02), 'joint_qd': (5., 2., 2.),
            'knife_pose': (np.ones(7), np.array([.05] * 3 + [.1] * 3),
                           np.array([.01] * 3 + [.02] * 3)),
            'knife_velocity': (np.array([1.] * 3 + [2.] * 3),
                               np.array([.5] * 3 + [1.] * 3),
                               np.array([.3] * 3 + [.5] * 3)),
            'wrench_history': (np.tile(force, self.heads.history),) * 3,
            'peak_wrench': (force,) * 3,
            'cut_progress': (1., 1., 1.),
        }
        for name in ('controller_q', 'controller_previous_position_command', 'controller_measured_q'):
            source[name] = source['joint_q']
        source['controller_velocity'] = source['joint_qd']
        source['controller_measured_qd'] = source['joint_qd']
        for name in ('controller_previous_error', 'filter_history', 'filter_zi', 'wrench_queue',
                     'controller_sensor_wrench', 'controller_target_wrench',
                     'filter_added_wrench', 'current_ft_value'):
            if name in self.shapes:
                scale = np.tile(force, int(np.prod(self.shapes[name])) // 6)
                source[name] = (scale, scale, scale)
        source['robot_warmstart'] = (1., 1., 1.)
        source['coupling_initialized'] = (1., 1., 1.)
        source['controller_cached_tip_pose'] = source['knife_pose']
        source['coupling_previous_pose'] = source['knife_pose']
        source['controller_target_pose'] = source['knife_pose']
        for name in ('filter_initialized', 'filter_flags'):
            source[name] = (1., 1., 1.)
        for name, queue in (('filter_history_count', 'filter_history'),
                            ('wrench_queue_count', 'wrench_queue')):
            if queue in self.shapes:
                source[name] = (float(self.shapes[queue][0]),) * 3
        for name in ('filter_added_age', 'filter_raw_age', 'controller_sample_age', 'filter_sample_age'):
            source[name] = (self.dt, self.dt, self.dt)
        for name in self.physical_fields:
            width = self.layout[name].stop - self.layout[name].start
            delta_width = 6 if name in self.pose_fields else width
            ins, err, out = source[name]
            scales_in.extend(np.broadcast_to(ins, (width,)).tolist())
            scales_error.extend(np.broadcast_to(err, (delta_width,)).tolist())
            scales_output.extend(np.broadcast_to(out, (delta_width,)).tolist())
            self.delta_slices[name] = slice(delta_offset, delta_offset + delta_width)
            delta_offset += delta_width
        history_scale = {
            'motion_command': [1.] * 6, 'previous_ee_pose': [1.] * 7,
            'previous_velocity': [1.] * 3 + [2.] * 3,
            'previous_acceleration': [1.] * 6, 'observation': [1.] * int(np.prod(self.shapes['observation'])),
            'success_counter': [metadata['environment']['successes_threshold'] + 1.],
        }
        history_scale['step_count'] = [metadata.get('free_cut_task', {}).get('horizon_steps', 1)]
        history_scale.update(controller_tare_compensation=force.tolist(), filter_offset=force.tolist())
        for name in self.history_fields:
            scales_in.extend(history_scale[name])
        self.delta_dim = delta_offset
        self.register_buffer('in_scale', torch.tensor(scales_in, dtype=torch.float32))
        self.register_buffer('err_scale', torch.tensor(scales_error, dtype=torch.float32))
        self.register_buffer('out_scale', torch.tensor(scales_output, dtype=torch.float32))
        bounds = torch.tensor(metadata['action_bounds'])
        self.register_buffer('action_low', bounds[0])
        self.register_buffer('action_high', bounds[1])
        self.register_buffer('force_limits', torch.tensor(force, dtype=torch.float32))
        # Task goal: target pose (7), optionally followed by the target cutting speed (m/s).
        # Every object shares the one force limit; object properties are not task inputs.
        self.goal_dim = int(metadata.get('goal_dim', 7))
        if self.goal_dim not in (7, 8):
            raise ValueError('goal_dim must be 7 (pose) or 8 (pose, target cutting speed)')

    def goal_parts(self, goal):
        """(target pose, target speed or None)."""
        if goal.shape[-1] != self.goal_dim:
            raise ValueError(f'Expected a {self.goal_dim}-dimensional goal, got {goal.shape[-1]}')
        return (goal, None) if self.goal_dim == 7 else (goal[..., :7], goal[..., 7])

    def limits(self, goal=None):
        """The task's force/torque limits (the same for every object and goal)."""
        return self.force_limits

    @classmethod
    def metadata_from_frame(cls, task, frame):
        result = dict(task)
        physical, history = cls.fields_for(task)
        result['state_shapes'] = {key: list(np.asarray(frame[key]).shape)
                                  for key in physical + history}
        result['spring_reference_ke'] = np.asarray(frame['spring_reference_ke']).tolist()
        return result

    def pack(self, frame, device=None):
        arrays = [np.asarray(frame[key]).reshape(-1)
                  for key in self.physical_fields + self.history_fields]
        return torch.as_tensor(np.concatenate(arrays), dtype=torch.float32, device=device)

    def pack_transition(self, frame, previous_state):
        return self.pack(frame, previous_state.device)

    def field(self, state, name):
        value = state[..., self.layout[name]]
        if name == 'success_counter':
            return value[..., 0]
        return value.reshape(*state.shape[:-1], *self.shapes[name]) if self.shapes[name] else value[..., 0]

    def assemble(self, fields):
        return torch.cat([fields[name].reshape(*fields['joint_q'].shape[:-1], -1)
                          for name in self.physical_fields + self.history_fields], -1)

    def canonical(self, state):
        return state / self.in_scale

    @property
    def dynamics_dim(self):
        return self.state_dim

    def dynamics_features(self, state):
        return self.canonical(state)

    def execute(self, action):
        return torch.maximum(torch.minimum(action, self.action_high), self.action_low)

    def integrate_physical(self, state, delta):
        fields = {name: self.field(state, name) for name in self.physical_fields + self.history_fields}
        d = delta * self.out_scale
        for name in self.physical_fields:
            if name in self.pose_fields:
                continue
            fields[name] = fields[name] + d[..., self.delta_slices[name]].reshape_as(fields[name])
        if 'particle_q' in fields:
            fields['particle_q'] = fields['particle_q'] + self.dt * fields['particle_qd']
        fields['joint_q'] = fields['joint_q'] + self.dt * fields['joint_qd']
        if 'knife_pose' not in fields:
            return fields
        pose = self.field(state, 'knife_pose')
        correction = d[..., self.delta_slices['knife_pose']]
        twist = fields['knife_velocity']
        pos = pose[..., :3] + self.dt * twist[..., :3] + correction[..., :3]
        rotation = quat_mul(pose[..., 3:], quat_exp(self.dt * twist[..., 3:] + correction[..., 3:]))
        fields['knife_pose'] = torch.cat((pos, torch.nn.functional.normalize(rotation, dim=-1)), -1)
        for name in self.pose_fields[1:]:
            if name not in fields:
                continue
            old_pose = self.field(state, name)
            correction = d[..., self.delta_slices[name]]
            rotation = quat_mul(old_pose[..., 3:], quat_exp(correction[..., 3:]))
            fields[name] = torch.cat((old_pose[..., :3] + correction[..., :3],
                                      torch.nn.functional.normalize(rotation, dim=-1)), -1)
        return fields

    def derive_motion(self, old, fields, action, goal):
        pose = self.fk(fields['joint_q'])
        velocity = observation_velocity(pose, self.field(old, 'previous_ee_pose'), self.dt)
        acceleration = (velocity - self.field(old, 'previous_velocity')) / self.dt
        jerk = (acceleration - self.field(old, 'previous_acceleration')) / self.dt
        target, target_speed = self.goal_parts(goal)
        obs = self.heads.observation(pose_error(target, pose), velocity, jerk, action, fields['wrench_history'])
        direction = action.new_tensor([1., 0., 0., 0., 0., 0.])
        rotation_direction = action.new_tensor([0., 0., 0., 0., -1., 0.])
        limits = self.metadata['controller']['max_twist']
        dt_robot = self.metadata['environment']['robot_control_dt']
        motion = self.field(old, 'motion_command') + dt_robot * (
            limits[0] * action[..., 4:5] * direction + limits[1] * action[..., 5:6] * rotation_direction)
        fields.update(motion_command=motion, previous_ee_pose=pose, previous_velocity=velocity,
                      previous_acceleration=acceleration, observation=obs)
        return fields, target_speed


    def derive(self, old, fields, action, goal):
        fields, target_speed = self.derive_motion(old, fields, action, goal)
        obs = fields['observation']
        collision = (fields['peak_wrench'] > self.force_limits).any(-1)
        done, success, outside, counter = self.heads.termination(
            obs, self.field(old, 'success_counter'), collision)
        fields['success_counter'] = counter
        reward = self.heads.reward(obs, collision, success, outside, target_speed)
        return self.assemble(fields), reward, done

    def error_vector(self, pred, true):
        errors = []
        for name in self.physical_fields:
            a, b = self.field(pred, name), self.field(true, name)
            err = pose_difference(a, b) if name in self.pose_fields else a - b
            errors.append(err.reshape(*pred.shape[:-1], -1))
        return torch.cat(errors, -1) / self.err_scale

    def terminal_features(self, state, goal=None):
        obs = self.field(state, 'observation')
        error = (obs[..., :3] * obs.new_tensor(self.heads.max_distance)).abs()
        workspace = error / obs.new_tensor(self.heads.env['workspace_limit']) - 1
        target = error / obs.new_tensor(self.heads.env['goal_threshold']) - 1
        force = self.field(state, 'peak_wrench') / self.limits(goal) - 1
        count = self.field(state, 'success_counter')[..., None] / (self.heads.env['successes_threshold'] + 1) - 1
        return torch.cat((force, workspace, target, count), -1)
