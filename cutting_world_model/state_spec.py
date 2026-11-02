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

    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata
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
        spring = np.asarray(metadata['spring_reference_ke'])
        if np.any(spring <= 0):
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
        }
        for name in self.physical_fields:
            width = self.layout[name].stop - self.layout[name].start
            delta_width = 6 if name == 'knife_pose' else width
            ins, err, out = source[name]
            scales_in.extend(np.broadcast_to(ins, (width,)).tolist())
            scales_error.extend(np.broadcast_to(err, (delta_width,)).tolist())
            scales_output.extend(np.broadcast_to(out, (delta_width,)).tolist())
            self.delta_slices[name] = slice(delta_offset, delta_offset + delta_width)
            delta_offset += delta_width
        history_scale = {
            'motion_command': [1.] * 6, 'previous_ee_pose': [1.] * 7,
            'previous_velocity': [1.] * 3 + [2.] * 3,
            'previous_acceleration': [1.] * 6, 'observation': [1.] * 24,
            'success_counter': [metadata['environment']['successes_threshold'] + 1.],
        }
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

    @classmethod
    def metadata_from_frame(cls, task, frame):
        result = dict(task)
        result['state_shapes'] = {key: list(np.asarray(frame[key]).shape)
                                  for key in cls.physical_fields + cls.history_fields}
        result['spring_reference_ke'] = np.asarray(frame['spring_reference_ke']).tolist()
        return result

    def pack(self, frame, device=None):
        arrays = [np.asarray(frame[key]).reshape(-1)
                  for key in self.physical_fields + self.history_fields]
        return torch.as_tensor(np.concatenate(arrays), dtype=torch.float32, device=device)

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

    def execute(self, action):
        return torch.maximum(torch.minimum(action, self.action_high), self.action_low)

    def integrate_physical(self, state, delta):
        fields = {name: self.field(state, name) for name in self.physical_fields + self.history_fields}
        d = delta * self.out_scale
        for name in self.physical_fields:
            if name == 'knife_pose':
                continue
            fields[name] = fields[name] + d[..., self.delta_slices[name]].reshape_as(fields[name])
        fields['particle_q'] = fields['particle_q'] + self.dt * fields['particle_qd']
        fields['joint_q'] = fields['joint_q'] + self.dt * fields['joint_qd']
        pose = self.field(state, 'knife_pose')
        correction = d[..., self.delta_slices['knife_pose']]
        twist = fields['knife_velocity']
        pos = pose[..., :3] + self.dt * twist[..., :3] + correction[..., :3]
        rotation = quat_mul(pose[..., 3:], quat_exp(self.dt * twist[..., 3:] + correction[..., 3:]))
        fields['knife_pose'] = torch.cat((pos, torch.nn.functional.normalize(rotation, dim=-1)), -1)
        return fields

    def derive(self, old, fields, action, goal):
        pose = self.fk(fields['joint_q'])
        velocity = observation_velocity(pose, self.field(old, 'previous_ee_pose'), self.dt)
        acceleration = (velocity - self.field(old, 'previous_velocity')) / self.dt
        jerk = (acceleration - self.field(old, 'previous_acceleration')) / self.dt
        obs = self.heads.observation(pose_error(goal, pose), velocity, jerk, action, fields['wrench_history'])
        collision = (fields['peak_wrench'] > self.force_limits).any(-1)
        done, success, outside, counter = self.heads.termination(
            obs, self.field(old, 'success_counter'), collision)
        reward = self.heads.reward(obs, collision, success, outside)
        direction = action.new_tensor([1., 0., 0., 0., 0., 0.])
        rotation_direction = action.new_tensor([0., 0., 0., 0., -1., 0.])
        limits = self.metadata['controller']['max_twist']
        dt_robot = self.metadata['environment']['robot_control_dt']
        motion = self.field(old, 'motion_command') + dt_robot * (
            limits[0] * action[..., 4:5] * direction + limits[1] * action[..., 5:6] * rotation_direction)
        fields.update(motion_command=motion, previous_ee_pose=pose, previous_velocity=velocity,
                      previous_acceleration=acceleration, observation=obs, success_counter=counter)
        return self.assemble(fields), reward, done

    def error_vector(self, pred, true):
        errors = []
        for name in self.physical_fields:
            a, b = self.field(pred, name), self.field(true, name)
            err = pose_difference(a, b) if name == 'knife_pose' else a - b
            errors.append(err.reshape(*pred.shape[:-1], -1))
        return torch.cat(errors, -1) / self.err_scale

    def terminal_features(self, state):
        obs = self.field(state, 'observation')
        error = (obs[..., :3] * obs.new_tensor(self.heads.max_distance)).abs()
        workspace = error / obs.new_tensor(self.heads.env['workspace_limit']) - 1
        target = error / obs.new_tensor(self.heads.env['goal_threshold']) - 1
        force = self.field(state, 'peak_wrench') / self.force_limits - 1
        count = self.field(state, 'success_counter')[..., None] / (self.heads.env['successes_threshold'] + 1) - 1
        return torch.cat((force, workspace, target, count), -1)
