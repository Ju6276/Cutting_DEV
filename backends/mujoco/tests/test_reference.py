import math

import pytest
import torch

from cutting_gpu_backend.controller.fdcc import quat_apply, quat_matrix
from cutting_gpu_backend.coupling import fixed_pose_from_urdf
from cutting_gpu_backend.reference import KnifeTrajectory


def unit_quaternion(seed):
    q = torch.randn(4, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return q / q.norm()


def test_constant_trajectory_is_bitwise_the_original_reference_formula():
    velocity = torch.tensor([0., -.05, 0.], dtype=torch.float32).double()
    duration = 2.
    rotation = unit_quaternion(0)
    time = torch.cat((torch.arange(0, 2.5, .002, dtype=torch.float64) + .05, torch.tensor([0., 2., 9.])))
    original = quat_apply(rotation, velocity) * time.clamp(max=duration)[:, None]
    trajectory = KnifeTrajectory.constant(velocity, duration)
    assert torch.equal(trajectory.displacement(time, rotation), original)


def test_file_round_trip_preserves_values_and_rejects_bad_files(tmp_path):
    trajectory = KnifeTrajectory([0., .1, .3], [[.01, -.05, 0.], [0., -.02, .03]], {'kind': 'test'})
    trajectory.save(tmp_path / 'knife.json')
    loaded = KnifeTrajectory.load(tmp_path / 'knife.json')
    assert torch.equal(loaded.times, trajectory.times)
    assert torch.equal(loaded.linear_velocity, trajectory.linear_velocity)
    assert loaded.metadata == {'kind': 'test'} and loaded.digest() == trajectory.digest()
    with pytest.raises(ValueError, match='start at zero'):
        KnifeTrajectory([.1, .2], [[0., 0., 0.]])
    with pytest.raises(ValueError, match='one 3-D velocity'):
        KnifeTrajectory([0., .1, .2], [[0., 0., 0.]])


def test_piecewise_velocity_integrates_and_holds_after_the_end():
    trajectory = KnifeTrajectory([0., .1, .3], [[.01, -.05, 0.], [0., -.02, .03]])
    time = torch.tensor([0., .05, .1, .2, .3, 1.], dtype=torch.float64)
    expected = torch.tensor([[0., 0., 0.], [.0005, -.0025, 0.], [.001, -.005, 0.],
                             [.001, -.007, .003], [.001, -.009, .006], [.001, -.009, .006]],
                            dtype=torch.float64)
    torch.testing.assert_close(trajectory.displacement(time), expected, atol=1e-15, rtol=0)


def test_displacement_rotates_disect_velocity_into_the_robot_base():
    urdf = """<robot name="r">
      <link name="base"/><link name="board"/>
      <joint name="fixed" type="fixed"><parent link="base"/><child link="board"/>
        <origin xyz="0.3 -0.2 0.1" rpy="1.5707963267948966 0 0.4"/></joint>
    </robot>"""
    base_from_disect = fixed_pose_from_urdf(urdf, 'base', 'board', dtype=torch.float64)
    rotation = base_from_disect[3:]
    trajectory = KnifeTrajectory([0., .2, .5], [[.02, -.05, .01], [-.01, -.03, .04]])
    time = torch.tensor([.05, .25, .7], dtype=torch.float64)
    matrix = quat_matrix(rotation)
    expected = trajectory.displacement(time) @ matrix.T
    torch.testing.assert_close(trajectory.displacement(time, rotation), expected, atol=1e-15, rtol=0)
    # Roll +90 degrees about x then yaw: DiSECt pressing (-y) points down (-z) in the base.
    down = KnifeTrajectory.constant([0., -1., 0.], 1.).displacement(torch.ones(1, dtype=torch.float64), rotation)
    torch.testing.assert_close(down[0], torch.tensor([0., 0., -1.], dtype=torch.float64), atol=1e-12, rtol=0)
    # Translation of the fixed transform does not enter a displacement.
    assert math.isclose(float(down.norm()), 1., rel_tol=1e-12)


def test_shared_reference_uses_commanded_speed_and_holds_board_goal():
    from cutting_gpu_backend.environment import CuttingBatch
    batch=CuttingBatch.__new__(CuttingBatch)
    batch.reference_start=torch.tensor([[0.,0.,.10,0.,0.,0.,1.],
                                        [0.,0.,.08,0.,0.,0.,1.]],dtype=torch.double)
    batch.target=batch.reference_start.clone()
    batch.target[:,2]=.02
    batch.rigid_tool_geometry={"board_normal_base":[0.,0.,1.]}
    batch.shared_reference_speed=torch.tensor([.01,.03],dtype=torch.double)
    result=batch.reference_target(torch.tensor([1.,1.],dtype=torch.double))
    torch.testing.assert_close(result[:,2],torch.tensor([.09,.05],dtype=torch.double))
    final=batch.reference_target(torch.tensor([100.,100.],dtype=torch.double))
    torch.testing.assert_close(final,batch.target)
    # Sampling a reference must not mutate the per-world reset pose.
    torch.testing.assert_close(batch.reference_start[:,2],torch.tensor([.10,.08],dtype=torch.double))


def test_explicit_trajectory_takes_precedence_over_shared_descent():
    from cutting_gpu_backend.environment import CuttingBatch
    from cutting_gpu_backend.reference import KnifeTrajectory
    batch=CuttingBatch.__new__(CuttingBatch)
    batch.reference_start=torch.tensor([[0.,0.,.10,0.,0.,0.,1.]],dtype=torch.double)
    batch.target=batch.reference_start.clone(); batch.target[:,2]=.02
    batch.rigid_tool_geometry={"board_normal_base":[0.,0.,1.]}
    batch.coupling=type('C',(),{'base_from_disect':torch.tensor([0.,0.,0.,0.,0.,0.,1.],dtype=torch.double)})()
    batch.dtype=torch.double
    batch.shared_reference_speed=torch.tensor([.03],dtype=torch.double)
    time=torch.tensor([1.],dtype=torch.double)
    # Default reference in a shared-scene config: the commanded descent.
    batch.explicit_reference=False
    torch.testing.assert_close(batch.reference_target(time)[:,2],torch.tensor([.07],dtype=torch.double))
    # A caller-set trajectory (here: no motion) is executed instead of the descent.
    batch.set_reference_trajectory(KnifeTrajectory.constant([0.,0.,0.],10.))
    assert batch.explicit_reference
    torch.testing.assert_close(batch.reference_target(time),batch.reference_start)


def test_explicit_trajectory_survives_shared_speed_installed_after_it():
    # Real order: the caller sets the trajectory, then GPUEnvironment.step() assigns
    # shared_reference_speed before every physics step.
    from cutting_gpu_backend.environment import CuttingBatch
    from cutting_gpu_backend.reference import KnifeTrajectory
    batch=CuttingBatch.__new__(CuttingBatch)
    batch.reference_start=torch.tensor([[0.,0.,.10,0.,0.,0.,1.]],dtype=torch.double)
    batch.target=batch.reference_start.clone(); batch.target[:,2]=.02
    batch.rigid_tool_geometry={"board_normal_base":[0.,0.,1.]}
    batch.coupling=type('C',(),{'base_from_disect':torch.tensor([0.,0.,0.,0.,0.,0.,1.],dtype=torch.double)})()
    batch.dtype=torch.double
    batch.set_reference_trajectory(KnifeTrajectory.constant([0.,0.,0.],10.))
    batch.shared_reference_speed=torch.tensor([.03],dtype=torch.double)   # as in step()
    torch.testing.assert_close(batch.reference_target(torch.tensor([1.],dtype=torch.double)),
                               batch.reference_start)
