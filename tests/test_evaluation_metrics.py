import os
import sys

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('CUTTING_BACKEND_ROOT'), reason='CUDA backend assets')
def test_reference_pressing_into_material_does_positive_work():
    root = os.environ['CUTTING_BACKEND_ROOT']
    sys.path.insert(0, root)
    from cutting_world_model.gpu_env import GPUEnvironment
    env = GPUEnvironment(root, 1)
    physics = env.physics
    work = gross = impulse = 0.
    # The scene reference presses at v_y = -0.05 m/s (DiSECt y up); contact starts near step 6.
    for _ in range(8):
        frame, _, terminated, truncated, info = env.step(torch.zeros(1, 6, device='cuda'), reference=True)
        assert not (terminated | truncated).any()
        margin = (frame['peak_wrench'] / physics.force_limits).amax(-1)
        torch.testing.assert_close(info['safety_margin'], margin, atol=0, rtol=0)
        assert float(info['material_impulse'][0]) <= float(info['peak_force'][0]) * .05 * (1 + 1e-6)
        work += float(info['knife_work'][0])
        gross += float(info['knife_gross_work'][0])
        impulse += float(info['material_impulse'][0])
    assert impulse > 0 and work > 0 and gross >= work
    # Mean downward knife speed weighted by force stays near the commanded 0.05 m/s.
    assert 0 < work / impulse < .1


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('CUTTING_BACKEND_ROOT'), reason='CUDA backend assets')
def test_linear_reward_environment_changes_only_the_force_term():
    root = os.environ['CUTTING_BACKEND_ROOT']
    sys.path.insert(0, root)
    from cutting_world_model.gpu_env import GPUEnvironment
    from cutting_world_model.task_heads import reward_settings, without_reward_settings
    source = GPUEnvironment(root, 1)
    linear = GPUEnvironment(root, 1, reward={'force_penalty': 'linear', 'force_penalty_scale': 50.})
    assert reward_settings(source.metadata['environment']) == {}
    assert 'force_penalty' not in source.physics.task['environment']
    assert without_reward_settings(linear.metadata) == source.metadata
    with pytest.raises(ValueError, match='Unknown reward settings'):
        GPUEnvironment(root, 1, reward={'w_force': 2.})
    action = torch.zeros(1, 6, device='cuda')
    for _ in range(8):
        frame_s, reward_s, *_ = source.step(action, reference=True)
        frame_l, reward_l, *_ = linear.step(action, reference=True)
        # Two GPU instances agree to float32 accumulation-order differences, not bitwise.
        torch.testing.assert_close(frame_l['observation'], frame_s['observation'], atol=1e-4, rtol=0)
        rest_l = reward_l.double() - linear.heads.weighted_force_reward(frame_l['observation'].double())
        rest_s = reward_s.double() - source.heads.weighted_force_reward(frame_s['observation'].double())
        torch.testing.assert_close(rest_l, rest_s, atol=1e-4, rtol=0)
        assert linear.heads.force_penalty == 'linear' and source.heads.force_penalty == 'sliceit'
