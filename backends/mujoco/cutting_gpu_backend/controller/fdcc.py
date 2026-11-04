"""ROS-free tensor port of the SliceIt FZI FDCC position controller.

C++ source: cambel/cartesian_controllers f4359b966bb3b8d8a07d829df463da814c2bee33.
Copyright 2019 FZI Research Center for Information Technology (BSD-3-Clause).
Action mapping: Copyright 2018-2022 Cristian C Beltran-Hernandez (MIT).
License texts are in licenses/. Poses use meters and ROS xyzw quaternions;
wrenches use [force, torque], and joint position commands use radians.
"""
from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import torch
from torch import nn

from .kdl_math import rotation_error


def quat_mul(a, b):
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return torch.cat((aw*bv + bw*av + torch.linalg.cross(av, bv),
                      aw*bw - (av*bv).sum(-1, keepdim=True)), -1)


def quat_inverse(q):
    return torch.cat((-q[..., :3], q[..., 3:]), -1) / q.square().sum(-1, keepdim=True)


def quat_apply(q, vector):
    cross = 2 * torch.linalg.cross(q[..., :3], vector)
    return vector + q[..., 3:] * cross + torch.linalg.cross(q[..., :3], cross)


def quat_exp(vector):
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return torch.cat((0.5 * torch.sinc(angle / (2*math.pi)) * vector, torch.cos(angle/2)), -1)


def quat_matrix(q):
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def pose_error(current, target):
    delta = target[..., :3] - current[..., :3]
    distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    translation = delta / distance.clamp_min(1.0)
    translation = torch.where(distance < 1e-6, torch.zeros_like(translation), translation)
    relative = quat_matrix(target[..., 3:]) @ quat_matrix(current[..., 3:]).transpose(-1,-2)
    rotation = rotation_error(relative)
    return torch.cat((translation, rotation), -1)


class SerialChain(nn.Module):
    def __init__(self, xml, base, tip, joint_names):
        super().__init__()
        root = ET.fromstring(xml)
        by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
        chain, cursor = [], tip
        while cursor != base:
            joint = by_child[cursor]
            chain.append(joint)
            cursor = joint.find('parent').get('link')
        self.names, self.kinds, self.indices = [], [], []
        positions, quaternions, axes = [], [], []
        lower, upper, velocity = [None]*len(joint_names), [None]*len(joint_names), [None]*len(joint_names)
        for joint in reversed(chain):
            name = joint.get('name')
            kind = joint.get('type')
            if kind not in ('fixed', 'revolute', 'continuous', 'prismatic'):
                raise ValueError('Unsupported joint type: ' + kind)
            if joint.find('mimic') is not None:
                raise ValueError('Mimic joint requires an explicit mapping')
            index = -1 if kind == 'fixed' else joint_names.index(name)
            origin = joint.find('origin')
            xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()] if origin is not None else [0., 0., 0.]
            rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()] if origin is not None else [0., 0., 0.]
            r, p, y = [v/2 for v in rpy]
            cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
            quaternion = [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                          cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]
            axis = joint.find('axis')
            axis = [float(x) for x in axis.get('xyz', '1 0 0').split()] if axis is not None else [1., 0., 0.]
            self.names.append(joint.find('child').get('link'))
            self.kinds.append(kind)
            self.indices.append(index)
            positions.append(xyz)
            quaternions.append(quaternion)
            axes.append(axis)
            if index >= 0:
                limit = joint.find('limit')
                lower[index] = -float('inf') if kind == 'continuous' else float(limit.get('lower', '0'))
                upper[index] = float('inf') if kind == 'continuous' else float(limit.get('upper', '0'))
                velocity[index] = float(limit.get('velocity'))
        self.base, self.tip = base, tip
        self.joint_names = tuple(joint_names)
        for name, values in [('origin_position', positions), ('origin_quaternion', quaternions),
                             ('axes', axes), ('lower', lower), ('upper', upper), ('velocity_limit', velocity)]:
            self.register_buffer(name, torch.tensor(values, dtype=torch.float64))

    def forward(self, q, link_mass=0.1, mass_matrix=True):
        batch, count = q.shape[0], q.shape[-1]
        position = torch.zeros((batch, 3), dtype=q.dtype, device=q.device)
        quaternion = torch.zeros((batch, 4), dtype=q.dtype, device=q.device)
        quaternion[:, 3] = 1
        frames = {self.base: torch.cat((position, quaternion), -1)}
        origins, directions, kinds = [], [], []
        mass = q.new_zeros((batch, count, count))
        tip_jacobian = None
        for segment, (name, kind, index) in enumerate(zip(self.names, self.kinds, self.indices)):
            offset = self.origin_position[segment].expand(batch, -1)
            position = position + quat_apply(quaternion, offset)
            quaternion = quat_mul(quaternion, self.origin_quaternion[segment].expand(batch, -1))
            if index >= 0:
                axis = self.axes[segment].expand(batch, -1)
                origins.append(position)
                directions.append(quat_apply(quaternion, axis))
                kinds.append((index, kind))
                if kind == 'prismatic':
                    position = position + directions[-1] * q[:, index:index+1]
                else:
                    quaternion = quat_mul(quaternion, quat_exp(axis * q[:, index:index+1]))
            frames[name] = torch.cat((position, quaternion), -1)
            linear = [q.new_zeros((batch, 3)) for _ in range(count)]
            angular = [q.new_zeros((batch, 3)) for _ in range(count)]
            for joint_origin, direction, (joint_index, joint_kind) in zip(origins, directions, kinds):
                if joint_kind == 'prismatic':
                    linear[joint_index] = direction
                else:
                    linear[joint_index] = torch.linalg.cross(direction, position-joint_origin)
                    angular[joint_index] = direction
            jv, jw = torch.stack(linear, -1), torch.stack(angular, -1)
            tip_jacobian = torch.cat((jv, jw), -2)
            if mass_matrix:
                if segment == len(self.names)-1:
                    m, inertia = 1.0, 1.0
                elif index >= 0:
                    m, inertia = link_mass, 1e-6
                else:
                    continue
                mass = mass + m*(jv.transpose(-1,-2) @ jv) + inertia*(jw.transpose(-1,-2) @ jw)
        return frames, tip_jacobian, mass


@dataclass
class ControllerConfig:
    compliance_ref_link: str
    sensor_ref_link: str
    iterations: int = 1
    link_mass: float = 0.1
    stiffness: tuple = (1000., 2000., 2000., 80., 40., 80.)
    p_gains: tuple = (0.2, 0.2, 0.02, 2., 0.2, 2.)
    d_gains: tuple = (0.,)*6
    selection: tuple = (1., 1., 0.5, 1., 0.5, 1.)
    error_scale: float = 0.015
    parallel: bool = True
    hand_frame_control: bool = False
    gravity: tuple = (0., 0., 0.)
    tool_mass: float = 0.
    tool_com: tuple = (0., 0., 0.)

    @classmethod
    def from_parameters(cls, parameters):
        labels = ('trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z')
        selectors = ('sel_x', 'sel_y', 'sel_z', 'sel_ax', 'sel_ay', 'sel_az')
        stiffness, solver = parameters['stiffness'], parameters['solver']
        return cls(compliance_ref_link=parameters['compliance_ref_link'],
                   sensor_ref_link=parameters['ft_sensor_ref_link'],
                   iterations=solver['iterations'],
                   link_mass=solver['forward_dynamics']['link_mass'],
                   stiffness=tuple(stiffness[n] for n in labels),
                   p_gains=tuple(parameters['pd_gains'][n]['p'] for n in labels),
                   d_gains=tuple(parameters['pd_gains'][n]['d'] for n in labels),
                   selection=tuple(stiffness[n] for n in selectors),
                   error_scale=solver['error_scale'],
                   parallel=stiffness['use_parallel_force_position_control'],
                   hand_frame_control=parameters['force']['hand_frame_control'],
                   gravity=tuple(parameters.get('gravity', {}).get(n, 0.) for n in ('x','y','z')),
                   tool_mass=parameters.get('tool', {}).get('mass', 0.),
                   tool_com=tuple(parameters.get('tool', {}).get('com_'+n, 0.) for n in ('x','y','z')))


class FDCCController:
    internal_dt = 0.02

    def __init__(self, chain, config):
        self.chain, self.config = chain, config

    def reset(self, q, qd=None):
        self.q = q.clone()
        self.velocity = torch.zeros_like(q) if qd is None else qd.clone() * 0.9
        self.previous_error = q.new_zeros((q.shape[0], 6))
        self.previous_position_command = q.clone()
        frames, _, _ = self.chain(q, mass_matrix=False)
        self.cached_tip_pose = frames[self.chain.tip].clone()
        self.tare_compensation = q.new_tensor((*[self.config.tool_mass*g for g in self.config.gravity], 0., 0., 0.)).expand(q.shape[0], -1).clone()

    def _gravity_sensor_wrench(self, frames):
        orientation = frames[self.config.sensor_ref_link][:, 3:]
        force_base = orientation.new_tensor([self.config.tool_mass*g for g in self.config.gravity]).expand(orientation.shape[0], -1)
        force = quat_apply(quat_inverse(orientation), force_base)
        com = force.new_tensor(self.config.tool_com).expand_as(force)
        return torch.cat((force, torch.linalg.cross(com, force)), -1)

    def tare(self, q):
        frames, _, _ = self.chain(q, mass_matrix=False)
        self.tare_compensation = self._gravity_sensor_wrench(frames)

    def force_error(self, frames, sensor_wrench, target_wrench):
        sensor = frames[self.config.sensor_ref_link]
        reference = frames[self.config.compliance_ref_link]
        sensor_wrench = sensor_wrench - self._gravity_sensor_wrench(frames) + self.tare_compensation
        force = quat_apply(sensor[:, 3:], sensor_wrench[:, :3])
        torque = quat_apply(sensor[:, 3:], sensor_wrench[:, 3:])
        torque = torque + torch.linalg.cross(sensor[:, :3]-reference[:, :3], force)
        if self.config.hand_frame_control:
            tip_quaternion = frames[self.chain.tip][:, 3:]
            target_wrench = torch.cat((quat_apply(tip_quaternion, target_wrench[:, :3]),
                                       quat_apply(tip_quaternion, target_wrench[:, 3:])), -1)
        return torch.cat((force, torque), -1) + target_wrench

    def update(self, measured_q, target_pose, sensor_wrench, target_wrench=None,
               stiffness=None, p_gains=None, d_gains=None, error_scale=None, selection=None):
        """One source C++ update; returns unsaturated hardware position commands.

        Internal velocity persists while measured position is resynchronized.
        The source caches the previous internal FK pose across this synchronization.
        """
        q = measured_q.clone()
        target_wrench = torch.zeros_like(sensor_wrench) if target_wrench is None else target_wrench
        def value(argument, default):
            return q.new_tensor(default).expand(q.shape[0], -1) if argument is None else argument
        stiffness = value(stiffness, self.config.stiffness)
        p_gains = value(p_gains, self.config.p_gains)
        d_gains = value(d_gains, self.config.d_gains)
        selection = value(selection, self.config.selection)
        if error_scale is None:
            error_scale = self.config.error_scale
        if isinstance(error_scale, torch.Tensor) and error_scale.ndim == 1:
            error_scale = error_scale[:, None]
        for _ in range(self.config.iterations):
            frames, jacobian, mass = self.chain(q, self.config.link_mass)
            motion = pose_error(self.cached_tip_pose, target_pose)
            rotation = quat_matrix(frames[self.config.compliance_ref_link][:, 3:])
            spring_parts = []
            for offset in (0, 3):
                rotated_error = rotation.transpose(-1,-2) @ motion[:, offset:offset+3, None]
                spring_parts.append((rotation @ (stiffness[:, offset:offset+3, None]*rotated_error)).squeeze(-1))
            spring = torch.cat(spring_parts, -1)
            force = self.force_error(frames, sensor_wrench, target_wrench)
            error = selection*spring + (1-selection)*force if self.config.parallel else spring+force
            cartesian_input = error_scale*(p_gains*error + d_gains*(error-self.previous_error)/self.internal_dt)
            acceleration = torch.linalg.solve(mass, jacobian.transpose(-1,-2) @ cartesian_input[..., None]).squeeze(-1)
            q = torch.maximum(torch.minimum(q+self.velocity*self.internal_dt, self.chain.upper), self.chain.lower)
            self.velocity = 0.9*(self.velocity+acceleration*self.internal_dt)
            self.previous_error = error
            self.cached_tip_pose = self.chain(q, mass_matrix=False)[0][self.chain.tip]
        self.q = q
        return q

    def enforce_position_limits(self, command, hardware_period):
        """Gazebo PositionJointSaturationHandle uses its previous command."""
        delta = self.chain.velocity_limit * hardware_period
        lower = torch.maximum(self.previous_position_command-delta, self.chain.lower)
        upper = torch.minimum(self.previous_position_command+delta, self.chain.upper)
        command = torch.maximum(torch.minimum(command, upper), lower)
        self.previous_position_command = command.clone()
        return command


class SlicingActionMapper:
    def __init__(self, max_twist=(1., 0.), robot_control_dt=0.002,
                 min_scale_error=1.5, max_scale_error=2.5, simulator_scale=0.01):
        self.max_twist, self.robot_control_dt = max_twist, robot_control_dt
        self.min_scale_error, self.max_scale_error = min_scale_error, max_scale_error
        self.simulator_scale = simulator_scale

    def reset(self, batch, device='cpu', dtype=torch.float64):
        self.added_motion_command = torch.zeros((batch, 6), device=device, dtype=dtype)

    def __call__(self, actions, target_pose, current_pose):
        normalized = actions[:, :4].clamp(0, 1)
        stiffness = actions.new_tensor([0., 1000., 1000., 40., 0., 40.]).expand(actions.shape[0], -1).clone()
        stiffness[:, 0] = torch.trunc(500+4500*normalized[:, 0])
        stiffness[:, 4] = torch.trunc(20+80*normalized[:, 1])
        stiffness *= 2
        gains = actions.new_tensor([0.1, 0.1, 0., 1., 0., 1.]).expand(actions.shape[0], -1).clone()
        gains[:, 2] = torch.round((0.01+0.04*normalized[:, 2])*1000)/1000
        gains[:, 4] = torch.round((0.1+1.4*normalized[:, 3])*100)/100
        gains *= 2
        self.added_motion_command[:, 0] += actions[:, 4].clamp(-1,1)*self.max_twist[0]*self.robot_control_dt
        self.added_motion_command[:, 4] -= actions[:, 5].clamp(-1,1)*self.max_twist[1]*self.robot_control_dt
        pose = target_pose.clone()
        pose[:, :3] += self.added_motion_command[:, :3]
        pose[:, 3:] = quat_mul(quat_exp(self.added_motion_command[:, 3:]), pose[:, 3:])
        pose[:, :3] = current_pose[:, :3]+0.8*(pose[:, :3]-current_pose[:, :3])
        distance = torch.linalg.vector_norm(pose[:, :3]-current_pose[:, :3], dim=-1)
        factor = 1-torch.tanh(100*distance)
        scale = self.min_scale_error+(self.max_scale_error-self.min_scale_error)*factor
        scale = torch.round(scale*1000)/1000
        scale = torch.round(scale*self.simulator_scale*10000)/10000
        return {'target_pose': pose, 'stiffness': stiffness, 'p_gains': gains,
                'd_gains': torch.zeros_like(gains), 'error_scale': scale,
                'selection': actions.new_tensor([1.,1.,0.5,1.,0.5,1.]).expand(actions.shape[0], -1)}
