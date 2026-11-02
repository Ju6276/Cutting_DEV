"""URDF serial-chain kinematics; poses and quaternions use ROS xyzw ordering."""
import math
import xml.etree.ElementTree as ET

import torch
from torch import nn


def quat_mul(a, b):
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return torch.cat((aw * bv + bw * av + torch.linalg.cross(av, bv),
                      aw * bw - (av * bv).sum(-1, keepdim=True)), -1)


def quat_inverse(q):
    return torch.cat((-q[..., :3], q[..., 3:]), -1) / q.square().sum(-1, keepdim=True)


def quat_apply(q, v):
    t = 2 * torch.linalg.cross(q[..., :3].expand_as(v), v)
    return v + q[..., 3:] * t + torch.linalg.cross(q[..., :3].expand_as(v), t)


def quat_exp(v):
    # sinc supplies the analytic limit at zero rotation.
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    return torch.cat((v * .5 * torch.sinc(angle / (2 * math.pi)), torch.cos(angle / 2)), -1)


def quat_log(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    q = torch.where(q[..., 3:] < 0, -q, q)
    s = torch.linalg.vector_norm(q[..., :3], dim=-1, keepdim=True)
    angle = 2 * torch.atan2(s, q[..., 3:])
    # Same small-angle branch as upstream rotations.rotvec_from_quat.
    small = s < 1e-6
    k = torch.where(small, 2 + angle.square() / 12, angle / torch.where(small, torch.ones_like(s), s))
    return q[..., :3] * k


def pose_compose(a, b):
    return torch.cat((a[..., :3] + quat_apply(a[..., 3:], b[..., :3]),
                      quat_mul(a[..., 3:], b[..., 3:])), -1)


def pose_difference(pred, true):
    return torch.cat((pred[..., :3] - true[..., :3],
                      quat_log(quat_mul(quat_inverse(true[..., 3:]), pred[..., 3:]))), -1)


def observation_velocity(current, previous, dt):
    # ur_control.transformations.angular_velocity_from_quaternions(current, previous, dt).
    angular = 2 * quat_mul(previous[..., 3:] - current[..., 3:],
                           quat_inverse(current[..., 3:]))[..., :3] / dt
    return torch.cat(((current[..., :3] - previous[..., :3]) / dt, angular), -1)


class SerialKinematics(nn.Module):
    def __init__(self, xml, base, tip, joint_names):
        super().__init__()
        joints = {j.find('child').get('link'): j for j in ET.fromstring(xml).findall('joint')}
        chain = []
        cursor = tip
        while cursor != base:
            joint = joints[cursor]
            chain.append(joint)
            cursor = joint.find('parent').get('link')
        self.indices, self.types = [], []
        origins, axes = [], []
        for joint in reversed(chain):
            origin = joint.find('origin')
            xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()] if origin is not None else [0.] * 3
            rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()] if origin is not None else [0.] * 3
            r, p, y = (torch.tensor(a, dtype=torch.float64) for a in rpy)
            qx = torch.stack((torch.sin(r / 2), r * 0, r * 0, torch.cos(r / 2)))
            qy = torch.stack((p * 0, torch.sin(p / 2), p * 0, torch.cos(p / 2)))
            qz = torch.stack((y * 0, y * 0, torch.sin(y / 2), torch.cos(y / 2)))
            origins.append(torch.cat((torch.tensor(xyz, dtype=torch.float64), quat_mul(qz, quat_mul(qy, qx)))))
            kind = joint.get('type')
            if kind not in ('fixed', 'revolute', 'continuous', 'prismatic'):
                raise ValueError('Unsupported URDF joint: ' + kind)
            if joint.find('mimic') is not None:
                raise ValueError('Mimic joint requires an explicit mapping')
            self.types.append(kind)
            self.indices.append(-1 if kind == 'fixed' else joint_names.index(joint.get('name')))
            axis = joint.find('axis')
            axes.append([float(x) for x in axis.get('xyz').split()] if axis is not None else [1., 0., 0.])
        self.register_buffer('origins', torch.stack(origins))
        self.register_buffer('axes', torch.tensor(axes, dtype=torch.float64))

    def forward(self, joints):
        pose = joints.new_zeros((*joints.shape[:-1], 7))
        pose[..., 6] = 1
        for i, (index, kind) in enumerate(zip(self.indices, self.types)):
            origin = self.origins[i].to(joints).expand_as(pose)
            pose = pose_compose(pose, origin)
            if index < 0:
                continue
            displacement = joints[..., index:index + 1] * self.axes[i].to(joints)
            zero = torch.zeros_like(displacement)
            if kind == 'prismatic':
                identity = torch.cat((zero, torch.ones_like(displacement[..., :1])), -1)
                motion = torch.cat((displacement, identity), -1)
            else:
                motion = torch.cat((zero, quat_exp(displacement)), -1)
            pose = pose_compose(pose, motion)
        q = pose[..., 3:]
        square = q.square()
        # KDL GetQuaternion selects positive w for trace > 1e-12, otherwise
        # the component corresponding to the largest matrix diagonal.
        xyz_index = torch.where((square[..., 0] > square[..., 1]) & (square[..., 0] > square[..., 2]),
                                0, torch.where(square[..., 1] > square[..., 2], 1, 2))
        index = torch.where(4 * square[..., 3] - 1 > 1e-12, 3, xyz_index)
        selected = q.gather(-1, index[..., None])
        q = torch.where(selected < 0, -q, q)
        return torch.cat((pose[..., :3], q), -1)
