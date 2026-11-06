import os
import sys

import pytest
import torch

from cutting_world_model.reset_bank import (
    SCHEMA, STAGES, CuttingResetBank, capture_snapshot, restore_snapshot,
)


def test_bank_rejects_mixed_task_and_samples_complete_stage_rows(tmp_path):
    contract = {'scene': 'test'}
    states = {name: {'history/state': torch.full((3, 2), float(i)),
                     'frame/joint_q': torch.arange(3)[:, None] + 10 * i}
              for i, name in enumerate(STAGES)}
    path = tmp_path / 'bank.pt'
    torch.save(dict(schema=SCHEMA, contract=contract, states=states,
                    split='test', seed=1, provenance={}), path)
    bank = CuttingResetBank(path, contract)
    sample, categories = bank.sample(100, torch.Generator().manual_seed(4))
    assert set(categories.tolist()) == set(range(4))
    assert torch.equal(sample['history/state'][:, 0].long(), categories)
    assert torch.equal(sample['frame/joint_q'][:, 0] // 10, categories)
    with pytest.raises(ValueError, match='expected'):
        CuttingResetBank(path, {'scene': 'different'})


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('CUTTING_BACKEND_ROOT'),
                    reason='CUDA backend assets')
def test_complete_snapshot_restores_material_history_and_preserves_other_world():
    root = os.environ['CUTTING_BACKEND_ROOT']
    sys.path.insert(0, root)
    from cutting_world_model.gpu_env import GPUEnvironment
    from cutting_world_model.control_state import ControlStateSpec
    env = GPUEnvironment(root, 2)
    one = {k: v[0].cpu().numpy() for k, v in env.current.items()}
    spec = ControlStateSpec(ControlStateSpec.metadata_from_frame(env.metadata, one)).cuda()
    history = spec.pack_batch(env.current)
    action = torch.zeros(2, 6, device=env.device)
    for _ in range(12):
        frame, _, term, trunc, _ = env.step(action, reference=True)
        assert not (term | trunc).any()
        history = spec.pack_batch_transition(frame, history)
    mask = torch.tensor([True, False], device=env.device)
    snapshot = capture_snapshot(env, history, mask)
    expected, reward, term, trunc, _ = env.step(action)
    untouched = {k: v[1].clone() for k, v in expected.items()}
    env.reset(mask)
    returned = restore_snapshot(env, snapshot, mask, new_episode=False)
    torch.testing.assert_close(returned, history[:1], atol=0, rtol=0)
    for key in ('particle_q', 'particle_qd', 'cut_spring_ke', 'joint_q', 'joint_qd',
                'filter_zi', 'controller_velocity', 'coupling_previous_pose'):
        torch.testing.assert_close(env.current[key][1], untouched[key], atol=0, rtol=0)
        torch.testing.assert_close(env.current[key][:1].cpu(), snapshot['frame/' + key], atol=0, rtol=0)
    actual, reward2, term2, trunc2, _ = env.step(action)
    torch.testing.assert_close(actual['joint_q'][0], expected['joint_q'][0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reward2[0], reward[0], atol=1e-4, rtol=1e-4)
    assert term2[0] == term[0] and trunc2[0] == trunc[0]
