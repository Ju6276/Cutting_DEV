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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA material integration')
def test_material_scales_match_independent_scaled_worlds():
    scene = Path('runs/disect_contact/scene.pt')
    scales = {'youngs_modulus': [1., 2., 1.], 'density': [1., 1., 1.5], 'cut_spring_stiffness': [1., 1., .5]}
    batch = BatchedDiSECt(scene, 3)
    base = {k: v.clone() for k, v in batch.base_material.items()}
    reset_ke = batch.initial_state['cut_spring_ke'].reshape(3, -1).clone()
    every = torch.ones(3, dtype=torch.bool, device='cuda')
    batch.set_material_scales(every, **{k: torch.tensor(v, device='cuda') for k, v in scales.items()})
    batch.reset(every)
    m = batch.model
    torch.testing.assert_close(m.tet_mu.reshape(3, -1)[1], 2 * base['tet_mu'][1], atol=0, rtol=0)
    torch.testing.assert_close(m.tet_lambda.reshape(3, -1)[0], base['tet_lambda'][0], atol=0, rtol=0)
    torch.testing.assert_close(m.particle_mass.reshape(3, -1)[2], 1.5 * base['particle_mass'][2], atol=0, rtol=0)
    torch.testing.assert_close(m.particle_inv_mass.reshape(3, -1)[2], base['particle_inv_mass'][2] / 1.5,
                               atol=0, rtol=0)
    assert torch.equal(m.particle_inv_mass.reshape(3, -1)[2] == 0, base['particle_inv_mass'][2] == 0)
    torch.testing.assert_close(batch.state_field('cut_spring_ke')[2], .5 * reset_ke[2], atol=0, rtol=0)
    torch.testing.assert_close(batch.state_field('cut_spring_ke')[0], reset_ke[0], atol=0, rtol=0)
    twist = torch.tensor([[0., -.03, 0., 0., 0., 0.]], device='cuda').repeat(3, 1)
    batch.set_motion(twist)
    batch.advance(.004)
    for i in range(3):
        single = BatchedDiSECt(scene, 1)
        one = torch.ones(1, dtype=torch.bool, device='cuda')
        single.set_material_scales(one, **{k: v[i] for k, v in scales.items()})
        single.reset(one)
        single.set_motion(twist[:1])
        single.advance(.004)
        torch.testing.assert_close(batch.state_field('particle_q')[i], single.state_field('particle_q')[0],
                                   atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(batch.state_field('cut_spring_ke')[i], single.state_field('cut_spring_ke')[0],
                                   atol=1e-3, rtol=1e-5)
    q = batch.state_field('particle_qd')
    assert not torch.allclose(q[0], q[1]) and not torch.allclose(q[0], q[2])
    with pytest.raises(ValueError, match='Unknown material'):
        batch.set_material_scales(every, poisson=1.)
    with pytest.raises(ValueError, match='positive'):
        batch.set_material_scales(every, density=0.)
