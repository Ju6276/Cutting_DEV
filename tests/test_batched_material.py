from pathlib import Path

import pytest
import torch

from cutting_gpu_backend.batched_disect import BatchedDiSECt
from cutting_gpu_backend.disect import DiSECt


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA material integration')
def test_packed_material_matches_independent_knives_and_selective_reset():
    scene = Path('runs/disect_contact/scene.pt')
    batch = BatchedDiSECt(scene, 2)
    single = [DiSECt(scene), DiSECt(scene)]
    twist = torch.tensor([[0., -.03, 0., 0., 0., 0.], [0., .03, 0., .1, 0., 0.]], device='cuda')
    batch.set_motion(twist)
    batch.advance(.002)
    for i, sim in enumerate(single):
        sim.set_motion(twist[i])
        sim.advance(.002)
        torch.testing.assert_close(batch.state_field('particle_q')[i], sim.state.particle_q, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(batch.state_field('particle_qd')[i], sim.state.particle_qd, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(batch.state_field('cut_spring_ke')[i], sim.state.cut_spring_ke, atol=1e-3, rtol=1e-5)
    assert not torch.equal(batch.state_field('particle_q')[0], batch.state_field('particle_q')[1])
    other = {k: v.clone() for k, v in vars(batch.state).items() if isinstance(v, torch.Tensor)}
    batch.reset(torch.tensor([True, False], device='cuda'))
    for name, value in other.items():
        target = batch.state_field(name)
        torch.testing.assert_close(target[1], value.reshape_as(target)[1], atol=0, rtol=0)
        torch.testing.assert_close(target[0], batch.initial_state[name].reshape_as(target)[0], atol=0, rtol=0)

    old_pose = torch.cat((batch.position, batch.rotation), -1).clone()
    next_pose = old_pose.clone()
    next_pose[:, 1] += .001
    mask = torch.tensor([True, False], device='cuda')
    batch.set_motion(twist, pose=next_pose, active=mask)
    torch.testing.assert_close(batch.position[1], old_pose[1, :3], atol=0, rtol=0)
    frozen = batch.capture_rows(~mask)
    batch.advance(.002, active=mask)
    for key, expected in frozen[0].items():
        torch.testing.assert_close(batch.state_field(key)[1:], expected, atol=0, rtol=0)
