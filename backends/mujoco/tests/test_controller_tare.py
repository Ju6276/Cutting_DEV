import torch
from cutting_gpu_backend.controller.fdcc import (
    ControllerConfig, FDCCController, SerialChain, quat_matrix,
)

XML = """<robot name="tare_test">
<link name="base"/><link name="arm"/><link name="sensor"/><link name="tool"/>
<joint name="joint" type="revolute"><parent link="base"/><child link="arm"/>
<axis xyz="0 1 0"/><limit lower="-3" upper="3" velocity="2" effort="10"/></joint>
<joint name="mount" type="fixed"><parent link="arm"/><child link="sensor"/>
<origin xyz="0.1 0.2 0.3" rpy="1.5707963267948966 0 0"/></joint>
<joint name="tool_mount" type="fixed"><parent link="sensor"/><child link="tool"/>
<origin xyz="0 0 0.1"/></joint></robot>"""

def setup(gravity=(0., 0., -9.8)):
    chain=SerialChain(XML,'base','tool',['joint'])
    config=ControllerConfig('tool','sensor',gravity=gravity,tool_mass=.35,
                            tool_com=(.02,-.03,.07))
    return FDCCController(chain,config)

def sensor_gravity(controller,q):
    frames=controller.chain(q,mass_matrix=False)[0]
    rotation=quat_matrix(frames['sensor'][:,3:])
    force=(rotation.transpose(-1,-2)@q.new_tensor([0.,0.,-.35*9.8])[None,:,None]).squeeze(-1)
    com=q.new_tensor([.02,-.03,.07]).expand_as(force)
    return frames,torch.cat((force,torch.linalg.cross(com,force)),-1)

def test_reset_cancels_zeroed_sensor_with_rotated_mount_and_com():
    controller=setup()
    q=torch.tensor([[0.],[.3],[-.7]],dtype=torch.double)
    controller.reset(q)
    frames,expected=sensor_gravity(controller,q)
    torch.testing.assert_close(controller.tare_compensation,expected,atol=1e-14,rtol=0)
    assert expected[:,3:].abs().max()>.01
    zero=torch.zeros_like(expected)
    torch.testing.assert_close(controller.force_error(frames,zero,zero),zero,atol=1e-14,rtol=0)
    # Each world retains its own sensor-frame baseline, not a broadcast base wrench.
    assert not torch.equal(expected[0],expected[1])

def test_changed_orientation_removes_only_gravity_delta_after_tare():
    controller=setup()
    start=torch.tensor([[0.],[.3]],dtype=torch.double)
    end=torch.tensor([[.4],[-.5]],dtype=torch.double)
    controller.reset(start)
    _,baseline=sensor_gravity(controller,start)
    frames,current=sensor_gravity(controller,end)
    sensor_reading=current-baseline  # A tared sensor, no external contact.
    zero=torch.zeros_like(sensor_reading)
    torch.testing.assert_close(controller.force_error(frames,sensor_reading,zero),zero,atol=1e-14,rtol=0)

def test_explicit_retare_replaces_reset_baseline():
    controller=setup()
    controller.reset(torch.tensor([[0.]],dtype=torch.double))
    q=torch.tensor([[.8]],dtype=torch.double)
    controller.tare(q)
    frames,expected=sensor_gravity(controller,q)
    torch.testing.assert_close(controller.tare_compensation,expected,atol=1e-14,rtol=0)
    zero=torch.zeros_like(expected)
    torch.testing.assert_close(controller.force_error(frames,zero,zero),zero,atol=1e-14,rtol=0)

def test_zero_gravity_legacy_reset_keeps_zero_compensation():
    controller=setup(gravity=(0.,0.,0.))
    q=torch.tensor([[0.],[.3]],dtype=torch.double)
    controller.reset(q)
    torch.testing.assert_close(controller.tare_compensation,torch.zeros((2,6),dtype=q.dtype),atol=0,rtol=0)
