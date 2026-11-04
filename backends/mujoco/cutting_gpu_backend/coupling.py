"""SliceIt knife-motion and force-signal coupling without ROS.

Copyright (c) 2023 OMRON SINIC X (BSD-3-Clause), and (c) 2018-2022
Cristian C Beltran-Hernandez (MIT). See licenses/ for the source licenses.
Poses use [xyz, quaternion xyzw]; twists and wrenches use [linear, angular]
and [force, torque]. Runtime tensor dimensions begin with environment.
"""
import math
import xml.etree.ElementTree as ET

import torch

from .controller.fdcc import quat_apply, quat_exp, quat_inverse, quat_matrix, quat_mul


def compose_pose(target_from_source, pose_source):
    transform = target_from_source.expand_as(pose_source)
    position = transform[..., :3] + quat_apply(transform[..., 3:], pose_source[..., :3])
    quaternion = quat_mul(transform[..., 3:], pose_source[..., 3:])
    return torch.cat((position, quaternion), -1)


def inverse_pose(pose):
    quaternion = quat_inverse(pose[..., 3:])
    return torch.cat((-quat_apply(quaternion, pose[..., :3]), quaternion), -1)


def fixed_pose_from_urdf(xml, target, source, device='cpu', dtype=torch.float64):
    root = ET.fromstring(xml)
    parents = {j.find('child').get('link'): j for j in root.findall('joint')}
    def ancestors(link):
        names = [link]
        while link in parents:
            link = parents[link].find('parent').get('link')
            names.append(link)
        return names
    target_ancestors, source_ancestors = ancestors(target), ancestors(source)
    common = next(name for name in target_ancestors if name in source_ancestors)
    def from_common(link):
        joints = []
        while link != common:
            joint = parents[link]
            if joint.get('type') != 'fixed':
                raise ValueError('Requested transform traverses a moving joint: '+joint.get('name'))
            joints.append(joint)
            link = joint.find('parent').get('link')
        pose = torch.tensor([0.,0.,0.,0.,0.,0.,1.],device=device,dtype=dtype)
        for joint in reversed(joints):
            origin = joint.find('origin')
            xyz = [float(v) for v in origin.get('xyz','0 0 0').split()] if origin is not None else [0.]*3
            rpy = [float(v) for v in origin.get('rpy','0 0 0').split()] if origin is not None else [0.]*3
            r,p,y = [v/2 for v in rpy]
            cr,sr,cp,sp,cy,sy = math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
            q = [sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy]
            pose = compose_pose(pose,torch.tensor(xyz+q,device=device,dtype=dtype))
        return pose
    return compose_pose(inverse_pose(from_common(target)),from_common(source))


def source_wrench_transform(wrench, target_from_source):
    """The exact block ordering used by ur_control.spalg.convert_wrench."""
    pose = target_from_source.expand(*wrench.shape[:-1],7)
    rotated_force = quat_apply(pose[...,3:],wrench[...,:3])
    rotated_torque = quat_apply(pose[...,3:],wrench[...,3:])
    force = rotated_force + torch.linalg.cross(pose[...,:3],rotated_torque)
    return torch.cat((force,rotated_torque),-1)


def source_angular_velocity(current_quaternion, previous_quaternion, dt=0.002):
    return (2*quat_mul(previous_quaternion-current_quaternion,
                       quat_inverse(current_quaternion))/dt)[...,:3]


def constrain_knife(pose_disect, twist_disect, spine_height):
    pose = pose_disect.clone()
    twist = twist_disect.clone()
    pose[...,0] = 0.
    pose[...,1] += spine_height/2
    rotation = quat_matrix(pose[...,3:])
    cy = torch.sqrt(rotation[...,0,0].square()+rotation[...,1,0].square())
    roll = torch.where(cy > 4*torch.finfo(torch.float64).eps,
                       torch.atan2(rotation[...,2,1],rotation[...,2,2]),
                       torch.atan2(-rotation[...,1,2],rotation[...,1,1]))
    roll_vector = torch.zeros_like(pose[...,:3])
    roll_vector[...,0] = roll
    pose[...,3:] = quat_exp(roll_vector)
    twist[...,0] = 0.
    twist[...,4:] = 0.
    return pose,twist


class SourceWrenchFilter:
    """Order-2, 50/500 Hz Butterworth; each call refilters the 200-sample window.

    The source carries lfilter's zi between full-window calls. A prefilled
    sensor baseline represents the startup sampling before the task begins.
    """
    def __init__(self, cutoff=50., sample_rate=500., window=200):
        self.window = window
        k = math.tan(math.pi*cutoff/sample_rate)
        scale = 1/(1+math.sqrt(2)*k+k*k)
        self.b = (k*k*scale,2*k*k*scale,k*k*scale)
        self.a = (1.,2*(k*k-1)*scale,(1-math.sqrt(2)*k+k*k)*scale)

    def reset(self, baseline):
        self.history = baseline[:,None,:].expand(-1,self.window,-1).clone()
        self.zi = baseline.new_zeros((baseline.shape[0],2,6))
        self.offset = torch.zeros_like(baseline)
        b0,b1,b2 = self.b
        _,a1,a2 = self.a
        transition = baseline.new_tensor([[-a1,1.],[-a2,0.]])
        input_map = baseline.new_tensor([b1-a1*b0,b2-a2*b0])
        state_weights = baseline.new_zeros((2,self.window))
        output_weights = baseline.new_zeros(self.window)
        power = torch.eye(2,device=baseline.device,dtype=baseline.dtype)
        for lag in range(self.window):
            state_weights[:,self.window-1-lag] = power @ input_map
            if lag < self.window-1:
                output_weights[self.window-2-lag] = (power @ input_map)[0]
            power = transition @ power
        output_weights[-1] = b0
        self.state_map = power
        self.output_state_map = torch.linalg.matrix_power(transition,self.window-1)[0]
        self.state_weights,self.output_weights = state_weights,output_weights
        self.tare()

    def filter_window(self):
        result = torch.einsum('t,btc->bc',self.output_weights,self.history)
        result += torch.einsum('s,bsc->bc',self.output_state_map,self.zi)
        self.zi = (torch.einsum('st,btc->bsc',self.state_weights,self.history)
                   + torch.einsum('st,btc->bsc',self.state_map,self.zi))
        return result

    def tare(self):
        self.offset = self.filter_window().clone()

    def update(self, sensor_wrench):
        self.history = torch.roll(self.history,-1,1)
        self.history[:,-1] = sensor_wrench
        return self.filter_window()-self.offset


class SliceItCoupling:
    def __init__(self, base_from_disect, spine_height):
        self.base_from_disect = base_from_disect
        self.disect_from_base = inverse_pose(base_from_disect)
        self.spine_height = spine_height
        self.previous_pose_base = None
        self.wrench_filter = SourceWrenchFilter()
        self.filter_initialized = False

    @classmethod
    def from_urdf(cls, xml, spine_height, base_link='b_bot_base_link',
                  disect_link='cutting_board_disect', device='cpu', dtype=torch.float64):
        transform = fixed_pose_from_urdf(xml,base_link,disect_link,device,dtype)
        return cls(transform,spine_height)

    def reset_pose(self, knife_pose_base):
        pose = compose_pose(self.disect_from_base,knife_pose_base)
        pose[...,1] -= 0.0043
        twist = pose.new_zeros((*pose.shape[:-1],6))
        pose,twist = constrain_knife(pose,twist,self.spine_height)
        return pose

    def motion(self, knife_pose_base):
        previous = knife_pose_base if self.previous_pose_base is None else self.previous_pose_base
        linear = (knife_pose_base[...,:3]-previous[...,:3])/0.002
        jump = (linear.abs()>1.).any(-1,keepdim=True)
        angular = source_angular_velocity(knife_pose_base[...,3:],previous[...,3:])
        transform = self.disect_from_base.expand_as(knife_pose_base)
        twist = torch.cat((quat_apply(transform[...,3:],linear),
                           quat_apply(transform[...,3:],angular)),-1)
        twist = torch.where(jump,torch.zeros_like(twist),twist)
        pose = compose_pose(self.disect_from_base,knife_pose_base)
        pose,twist = constrain_knife(pose,twist,self.spine_height)
        self.previous_pose_base = knife_pose_base.clone()
        return {'pose':pose,'linear_velocity':twist[...,:3],
                'angular_velocity':twist[...,3:],'update_pose':False}

    def reset_filter(self, raw_sensor_baseline):
        self.wrench_filter.reset(raw_sensor_baseline)
        self.filter_initialized = True

    def feedback(self, knife_f, sensor_pose_base, ee_pose_base, raw_sensor_wrench):
        magnitude = torch.linalg.vector_norm(knife_f,dim=-1).sum(-1)
        wrench_disect = knife_f.new_zeros((*magnitude.shape,6))
        wrench_disect[...,1] = magnitude
        wrench_base = source_wrench_transform(wrench_disect,self.base_from_disect)
        external_sensor = source_wrench_transform(wrench_base,inverse_pose(sensor_pose_base))
        raw = raw_sensor_wrench
        if not self.filter_initialized:
            self.reset_filter(raw)
        filtered = self.wrench_filter.update(raw+external_sensor)
        observation = source_wrench_transform(filtered,ee_pose_base)
        return {'force_magnitude':magnitude,'wrench_base':wrench_base,
                'external_sensor_wrench':external_sensor,'sensor_wrench':filtered,
                'observation_wrench':observation}
