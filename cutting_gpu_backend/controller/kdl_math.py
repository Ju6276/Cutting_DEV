"""KDL 1.4 rotation-error conventions expressed with batched tensors.

Adapted from Orocos KDL frames.cpp, Copyright (C) 2006 Erwin Aertbelien.
SPDX-License-Identifier: LGPL-2.1-or-later. See licenses/KDL-LGPL-2.1.txt.
https://github.com/orocos/orocos_kinematics_dynamics/blob/v1.4.0/orocos_kdl/src/frames.cpp
"""
import math
import torch


def rotation_error(rotation):
    # These tolerances are KDL::epsilon and 10*KDL::epsilon from version 1.4.
    epsilon = 1e-6
    off_diagonal = torch.stack((rotation[:, 2, 1]-rotation[:, 1, 2],
                                rotation[:, 0, 2]-rotation[:, 2, 0],
                                rotation[:, 1, 0]-rotation[:, 0, 1]), -1)
    symmetric = torch.stack((rotation[:, 2, 1]+rotation[:, 1, 2],
                             rotation[:, 0, 2]+rotation[:, 2, 0],
                             rotation[:, 1, 0]+rotation[:, 0, 1]), -1)
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    singular = (off_diagonal.abs() < epsilon).all(-1)
    identity = (symmetric.abs() < 10*epsilon).all(-1) & ((trace-3).abs() < 10*epsilon)
    angle = torch.acos(((trace-1)/2).clamp(-1,1)).clamp(max=1.0)
    norm = torch.linalg.vector_norm(off_diagonal, dim=-1, keepdim=True)
    unit_x = off_diagonal.new_tensor([1.,0.,0.]).expand_as(off_diagonal)
    axis = torch.where(norm < epsilon, unit_x,
                       off_diagonal / torch.where(norm > 0, norm, torch.ones_like(norm)))
    diagonal = (rotation.diagonal(dim1=-2,dim2=-1)+1)/2
    xx, yy, zz = diagonal.unbind(-1)
    xy, xz, yz = symmetric[:, 2]/4, symmetric[:, 1]/4, symmetric[:, 0]/4
    chosen = torch.where((xx>yy)&(xx>zz),0,torch.where(yy>zz,1,2))
    root = torch.sqrt(diagonal.clamp_min(0))
    safe_root = torch.where(root > 0, root, torch.ones_like(root))
    x_case = torch.stack((root[:,0],xy/safe_root[:,0],xz/safe_root[:,0]),-1)
    y_case = torch.stack((xy/safe_root[:,1],root[:,1],yz/safe_root[:,1]),-1)
    z_case = torch.stack((xz/safe_root[:,2],yz/safe_root[:,2],root[:,2]),-1)
    half_sqrt_two = math.sqrt(2)/2
    x_case = torch.where((xx < epsilon)[:,None], x_case.new_tensor([0.,half_sqrt_two,half_sqrt_two]),x_case)
    y_case = torch.where((yy < epsilon)[:,None], y_case.new_tensor([half_sqrt_two,0.,half_sqrt_two]),y_case)
    z_case = torch.where((zz < epsilon)[:,None], z_case.new_tensor([half_sqrt_two,half_sqrt_two,0.]),z_case)
    pi_axis = torch.where((chosen==0)[:,None],x_case,torch.where((chosen==1)[:,None],y_case,z_case))
    result = angle[:,None]*axis
    result = torch.where((singular & ~identity)[:,None],pi_axis,result)
    return torch.where((singular & identity)[:,None],torch.zeros_like(result),result)
