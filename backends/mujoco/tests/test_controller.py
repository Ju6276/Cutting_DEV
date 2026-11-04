import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cutting_gpu_backend.controller import ControllerConfig, FDCCController, SerialChain, SlicingActionMapper
from cutting_gpu_backend.controller.fdcc import pose_error, quat_exp, quat_mul


def setup_controller(batch=1, iterations=1):
    task = json.loads(Path('runs/source/task.json').read_text())
    chain = SerialChain(task['robot_xml'], task['robot_base'], task['robot_tip'], task['joint_names'])
    config = ControllerConfig('b_bot_gripper_tip_link', 'b_bot_wrist_3_link', iterations=iterations)
    controller = FDCCController(chain, config)
    q = torch.tensor(np.load('runs/source/episode_000.npz')['joint_q'][:batch], dtype=torch.float64)
    controller.reset(q)
    return controller, q


def test_source_action_mapping_rounding_and_history():
    mapper = SlicingActionMapper()
    actions = torch.tensor([[-1,-1,-1,-1,0.5,1], [0.37,0.53,0.71,0.43,-0.4,1]],dtype=torch.float64)
    mapper.reset(2)
    current = torch.tensor([[0.,0.,0.,0.,0.,0.,1.]]*2,dtype=torch.float64)
    target = current.clone()
    target[:,2]=0.08
    actual = mapper(actions,target,current)
    for i, action in enumerate(actions.numpy()):
        stiffness = np.array([np.interp(action[0],[0,1],[500,5000]),1000,1000,40,
                              np.interp(action[1],[0,1],[20,100]),40],dtype=int)*2
        p = np.array([0.1,0.1,round(np.interp(action[2],[0,1],[0.01,0.05]),3),
                      1,round(np.interp(action[3],[0,1],[0.1,1.5]),2),1])*2
        np.testing.assert_array_equal(actual['stiffness'][i].numpy(),stiffness)
        np.testing.assert_allclose(actual['p_gains'][i].numpy(),p,atol=1e-15)
    expected_x=0.8*actions[:,4]*0.002
    torch.testing.assert_close(actual['target_pose'][:,0],expected_x)
    second=mapper(actions,target,current)
    torch.testing.assert_close(second['target_pose'][:,0],2*expected_x)
    assert torch.equal(mapper.added_motion_command[:,4],torch.zeros(2,dtype=torch.float64))


def test_motion_error_caps_are_source_caps():
    current=torch.tensor([[0.,0.,0.,0.,0.,0.,1.]],dtype=torch.float64)
    target=current.clone()
    target[0,:3]=torch.tensor([3.,4.,0.])
    target[:,3:]=quat_exp(torch.tensor([[0.,0.,2.]],dtype=torch.float64))
    torch.testing.assert_close(pose_error(current,target),torch.tensor([[0.6,0.8,0.,0.,0.,1.]],dtype=torch.float64))


def test_first_euler_position_uses_old_velocity_and_persists_velocity():
    controller,q=setup_controller()
    target=controller.cached_tip_pose.clone()
    target[:,0]+=0.01
    wrench=torch.zeros((1,6),dtype=q.dtype)
    first=controller.update(q,target,wrench)
    torch.testing.assert_close(first,q,rtol=0,atol=0)
    old_velocity=controller.velocity.clone()
    assert old_velocity.abs().max()>0
    shifted=q+0.001
    second=controller.update(shifted,target,wrench)
    torch.testing.assert_close(second,shifted+old_velocity*0.02,rtol=0,atol=1e-15)


def test_hardware_saturation_uses_previous_command():
    controller,q=setup_controller()
    step=controller.chain.velocity_limit*0.001
    one=controller.enforce_position_limits(q+1.,0.001)
    two=controller.enforce_position_limits(q+1.,0.001)
    torch.testing.assert_close(one,q+step)
    torch.testing.assert_close(two,q+2*step)


def test_batch_matches_independent_controller_states():
    batch,q=setup_controller(3,iterations=2)
    target=batch.cached_tip_pose.clone()
    target[:,0]+=torch.tensor([0.005,0.01,-0.01],dtype=q.dtype)
    wrench=torch.arange(18,dtype=q.dtype).reshape(3,6)*0.01
    singles=[]
    for i in range(3):
        single=FDCCController(batch.chain,batch.config)
        single.reset(q[i:i+1])
        singles.append(single)
    for _ in range(3):
        actual=batch.update(q,target,wrench)
        reference=torch.cat([c.update(q[i:i+1],target[i:i+1],wrench[i:i+1]) for i,c in enumerate(singles)])
        torch.testing.assert_close(actual,reference,rtol=1e-12,atol=1e-14)
        torch.testing.assert_close(batch.velocity,torch.cat([c.velocity for c in singles]),rtol=1e-12,atol=1e-14)


def test_force_reference_shift_matches_spatial_wrench_formula():
    controller,q=setup_controller()
    frames,_,_=controller.chain(q,mass_matrix=False)
    wrench=q.new_tensor([[1.,2.,3.,0.1,0.2,0.3]])
    result=controller.force_error(frames,wrench,torch.zeros_like(wrench))
    from cutting_gpu_backend.controller.fdcc import quat_apply
    sensor=frames[controller.config.sensor_ref_link]
    ref=frames[controller.config.compliance_ref_link]
    force=quat_apply(sensor[:,3:],wrench[:,:3])
    torque=quat_apply(sensor[:,3:],wrench[:,3:])+torch.linalg.cross(sensor[:,:3]-ref[:,:3],force)
    torch.testing.assert_close(result,torch.cat((force,torque),-1))
