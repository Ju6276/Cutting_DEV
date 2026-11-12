import os
import sys

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('CUTTING_BACKEND_ROOT'), reason='CUDA backend assets')
def test_reference_trajectory_targets_match_original_branch_and_frame(tmp_path):
    root = os.environ['CUTTING_BACKEND_ROOT']
    sys.path.insert(0, root)
    from cutting_gpu_backend.controller.fdcc import quat_apply
    from cutting_gpu_backend.reference import KnifeTrajectory
    from cutting_world_model.gpu_env import GPUEnvironment
    env = GPUEnvironment(root, 1)
    physics = env.physics
    material = physics.material
    rotation = physics.coupling.base_from_disect[3:]
    times = [torch.full((1,), .05 * k + .05, device='cuda', dtype=torch.float64) for k in range(60)]

    def original(time):
        velocity = quat_apply(rotation, material.reference_linear_velocity.to(physics.dtype))
        target = physics.reference_start.clone()
        target[:, :3] += velocity * time.clamp(max=material.reference_duration)[:, None]
        return target

    default = [physics.reference_target(time) for time in times]
    for time, target in zip(times, default):
        assert torch.equal(target, original(time))
    KnifeTrajectory.constant(material.reference_linear_velocity.double().cpu(),
                             material.reference_duration).save(tmp_path / 'constant.json')
    physics.set_reference_trajectory(KnifeTrajectory.load(tmp_path / 'constant.json'))
    for time, target in zip(times, default):
        assert torch.equal(physics.reference_target(time), target)

    lateral = KnifeTrajectory([0., .4], [[0., -.05, .02]])
    physics.set_reference_trajectory(lateral)
    time = times[9]
    delta = physics.reference_target(time) - physics.reference_start
    expected = quat_apply(rotation, torch.tensor([0., -.05, .02], device='cuda', dtype=torch.float64) * .4)
    torch.testing.assert_close(delta[0, :3], expected, atol=1e-15, rtol=0)
    assert torch.equal(delta[0, 3:], torch.zeros_like(delta[0, 3:]))
    assert float(delta[0, 2]) < 0  # DiSECt pressing direction moves the target down in the base frame.

    env.reset()
    for _ in range(3):
        frame, reward, terminated, truncated, info = env.step(torch.zeros(1, 6, device='cuda'), reference=True)
        assert torch.isfinite(reward).all() and not (terminated | truncated).any()
    physics.set_reference_trajectory()
    assert torch.equal(physics.reference_target(times[0]), default[0])


def test_lifted_reference_duration_adds_the_lift_travel():
    import pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'backends/mujoco'))
    from cutting_gpu_backend.reference import lifted_durations
    lift = torch.tensor([0., .02, -.01, -1.], dtype=torch.float64)
    durations = lifted_durations(1.3, .05, lift)
    assert torch.allclose(durations, torch.tensor([1.3, 1.7, 1.1, 0.], dtype=torch.float64))
    # travel = speed * duration = scene travel + lift (clamped at no motion)
    assert torch.allclose(.05 * durations[:3], .065 + lift[:3])
    with pytest.raises(ValueError):
        lifted_durations(1.3, 0., lift)


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('CUTTING_BACKEND_ROOT'), reason='CUDA backend assets')
def test_default_reference_compensates_the_start_lift():
    root = os.environ['CUTTING_BACKEND_ROOT']
    sys.path.insert(0, root)
    from cutting_world_model.gpu_env import GPUEnvironment
    lift = torch.tensor([0., .015])
    physics = GPUEnvironment(root, 2, initial_lift=lift).physics
    late = torch.full((2,), 100., device='cuda', dtype=torch.float64)
    end = physics.reference_target(late)
    # Both worlds end at the same pose although world 1 started 15 mm higher.
    # The lifted start comes from inverse kinematics; compare within its tolerance.
    assert torch.allclose(end[0], end[1], atol=1e-5)
    travel = (physics.reference_start[:, :3] - end[:, :3]).norm(dim=-1)
    assert torch.allclose(travel[1] - travel[0], torch.tensor(.015, dtype=torch.float64, device="cuda"), atol=1e-5)
