import json
from pathlib import Path

import numpy as np
import torch

from cutting_gpu_backend.controller.fdcc import quat_exp, quat_matrix
from cutting_gpu_backend.coupling import (
    SliceItCoupling, SourceWrenchFilter, compose_pose, fixed_pose_from_urdf,
    inverse_pose, source_angular_velocity, source_wrench_transform,
)


def test_source_spalg_block_order_with_translation_and_torque():
    pose = torch.tensor([[0.2,-0.3,0.4,0.,0.,0.,1.]],dtype=torch.float64)
    pose[:,3:] = quat_exp(torch.tensor([[0.1,0.2,-0.3]],dtype=torch.float64))
    wrench = torch.tensor([[1.,2.,3.,4.,5.,6.]],dtype=torch.float64)
    transform = np.eye(4)
    transform[:3,:3] = quat_matrix(pose[:,3:])[0].numpy()
    transform[:3,3] = pose[0,:3].numpy()
    inverse = np.linalg.inv(transform)
    x,y,z = inverse[:3,3]
    cross = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    motion = np.zeros((6,6))
    motion[:3,:3] = inverse[:3,:3]
    motion[3:,:3] = cross @ inverse[:3,:3]
    motion[3:,3:] = inverse[:3,:3]
    expected = motion.T @ wrench[0].numpy()
    np.testing.assert_allclose(source_wrench_transform(wrench,pose)[0],expected,atol=1e-14)


def test_fixed_urdf_transform_and_inverse():
    xml = '''<robot><joint name="a" type="fixed"><parent link="world"/><child link="base"/>
      <origin xyz="1 0 0"/></joint><joint name="b" type="fixed"><parent link="world"/>
      <child link="board"/><origin xyz="0 2 0" rpy="0 0 1.5707963267948966"/></joint></robot>'''
    pose = fixed_pose_from_urdf(xml,'base','board')
    torch.testing.assert_close(pose[:3],torch.tensor([-1.,2.,0.],dtype=torch.float64))
    identity = compose_pose(inverse_pose(pose),pose)
    torch.testing.assert_close(identity,torch.tensor([0.,0.,0.,0.,0.,0.,1.],dtype=torch.float64),atol=1e-14,rtol=0)
    task = json.loads(Path('runs/source/task.json').read_text())
    actual = fixed_pose_from_urdf(task['robot_xml'],task['robot_base'],'cutting_board_disect')
    assert torch.isfinite(actual).all()


def test_reset_offset_plane_constraint_and_source_velocity_sign():
    coupling = SliceItCoupling(torch.tensor([0.,0.,0.,0.,0.,0.,1.],dtype=torch.float64),0.04)
    pose = torch.tensor([[0.2,0.3,0.4,0.,0.,0.,1.]],dtype=torch.float64)
    reset = coupling.reset_pose(pose)
    torch.testing.assert_close(reset[0,:3],torch.tensor([0.,0.3157,0.4],dtype=torch.float64))
    coupling.motion(pose)
    moved = pose.clone()
    moved[:,1] += 0.0004
    moved[:,3:] = quat_exp(torch.tensor([[0.002,0.,0.]],dtype=torch.float64))
    motion = coupling.motion(moved)
    torch.testing.assert_close(motion['linear_velocity'],torch.tensor([[0.,0.2,0.]],dtype=torch.float64))
    expected_angular = -2*np.sin(0.001)/0.002
    torch.testing.assert_close(motion['angular_velocity'],torch.tensor([[expected_angular,0.,0.]],dtype=torch.float64))
    assert motion['update_pose'] is False
    jump = moved.clone()
    jump[:,0] += 0.01
    stopped = coupling.motion(jump)
    assert torch.equal(stopped['linear_velocity'],torch.zeros_like(stopped['linear_velocity']))
    assert torch.equal(stopped['angular_velocity'],torch.zeros_like(stopped['angular_velocity']))


def test_whole_window_filter_matches_source_lfilter_recurrence():
    filtering = SourceWrenchFilter()
    baseline = torch.tensor([[0.2,-0.1,1.,0.1,0.2,0.3]],dtype=torch.float64)
    filtering.reset(baseline)
    history = np.repeat(baseline.numpy()[:,None,:],200,axis=1)
    zi = np.zeros((1,2,6))
    b0,b1,b2 = filtering.b
    _,a1,a2 = filtering.a
    def original_window(window, state):
        state = state.copy()
        for i in range(window.shape[1]):
            x = window[:,i]
            y = b0*x+state[:,0]
            first = b1*x-a1*y+state[:,1]
            second = b2*x-a2*y
            state[:,0] = first
            state[:,1] = second
        return y,state
    offset,zi = original_window(history,zi)
    for index in range(10):
        sample = baseline.numpy()+np.sin(index+np.arange(6))*0.4
        history = np.roll(history,-1,axis=1)
        history[:,-1] = sample
        expected,zi = original_window(history,zi)
        actual = filtering.update(torch.tensor(sample))
        np.testing.assert_allclose(actual.numpy(),expected-offset,atol=2e-15,rtol=1e-13)
        np.testing.assert_allclose(filtering.zi.numpy(),zi,atol=2e-15,rtol=1e-13)


def test_feedback_reaches_sensor_and_observation_without_vector_cancellation():
    coupling = SliceItCoupling(torch.tensor([0.,0.,0.,0.,0.,0.,1.],dtype=torch.float64),0.04)
    pose = torch.tensor([[0.,0.,0.,0.,0.,0.,1.]],dtype=torch.float64)
    opposite_forces = torch.tensor([[[0.,3.,0.],[0.,-4.,0.]]],dtype=torch.float64)
    baseline = torch.tensor([[0.,0.,-9.8,0.,0.,0.]],dtype=torch.float64)
    result = coupling.feedback(opposite_forces,pose,pose,raw_sensor_wrench=baseline)
    assert result['force_magnitude'].item() == 7.
    torch.testing.assert_close(result['wrench_base'],torch.tensor([[0.,7.,0.,0.,0.,0.]],dtype=torch.float64))
    expected = torch.tensor([[0.,7*coupling.wrench_filter.b[0],0.,0.,0.,0.]],dtype=torch.float64)
    torch.testing.assert_close(result['sensor_wrench'],expected,atol=1e-14,rtol=1e-13)
    torch.testing.assert_close(result['observation_wrench'],result['sensor_wrench'])
