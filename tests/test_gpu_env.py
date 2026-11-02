import os
import sys
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('CUTTING_BACKEND_ROOT'), reason='CUDA backend assets')
def test_gpu_task_reset_preserves_other_world_and_observation_state():
    root = os.environ['CUTTING_BACKEND_ROOT']
    sys.path.insert(0, root)
    from cutting_world_model.gpu_env import GPUEnvironment
    from cutting_world_model.state_spec import CuttingStateSpec
    env = GPUEnvironment(root, 2)
    initial = {k: v.clone() for k, v in env.current.items()}
    frame, reward, term, trunc, info = env.step(torch.tensor([[0.] * 6, [.2] * 6], device='cuda'))
    assert reward.shape == term.shape == trunc.shape == (2,)
    assert torch.isfinite(reward).all()
    before = {k: v.clone() for k, v in frame.items()}
    controller_before = env.physics.controller.velocity.clone()
    frame = env.reset(torch.tensor([True, False], device='cuda'))
    for k in frame:
        torch.testing.assert_close(frame[k][1], before[k][1], atol=0, rtol=0)
        torch.testing.assert_close(frame[k][0], initial[k][0], atol=0, rtol=0)
    torch.testing.assert_close(env.physics.controller.velocity[1], controller_before[1], atol=0, rtol=0)
    one = {k: v[0].cpu().numpy() for k, v in frame.items()}
    spec = CuttingStateSpec(CuttingStateSpec.metadata_from_frame(env.metadata, one)).cuda()
    state = spec.assemble(frame).float()
    torch.testing.assert_close(spec.field(state, 'observation'), frame['observation'].float())
    continued, *_ = env.step(torch.zeros(2, 6, device='cuda'))
    assert torch.isfinite(spec.assemble(continued)).all()

    before_stop = {k: v.clone() for k, v in continued.items()}
    env.physics.feedback['observation_wrench'][0, 0] = env.physics.force_limits[0] + 1
    stopped, _, term, _, info = env.step(torch.zeros(2, 6, device='cuda'))
    assert term[0] and info['collision'][0] and not info['collision'][1]
    torch.testing.assert_close(stopped['particle_q'][0], before_stop['particle_q'][0], atol=0, rtol=0)
    assert not torch.equal(stopped['particle_q'][1], before_stop['particle_q'][1])
