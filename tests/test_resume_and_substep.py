"""Resume restores the replay written with the checkpoint; object groups carry a DiSECt substep."""
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backends/mujoco"))
from cutting_world_model.multi_env import MultiConditionEnvironment  # noqa: E402
from cutting_world_model.online_replay import BatchedOnlineReplay  # noqa: E402


def filled_replay(worlds=2, steps=5):
    replay = BatchedOnlineReplay(worlds, 3, torch.device('cpu'), capacity=16)
    for t in range(steps):
        ended = torch.tensor([t == 2, False])
        replay.add(torch.full((worlds, 3), float(t)), torch.zeros(worlds, 6), torch.zeros(worlds, 6),
                   torch.zeros(worlds, 7), torch.zeros(worlds), torch.full((worlds, 3), t + 1.),
                   ended, torch.zeros(worlds, dtype=torch.bool))
    return replay


def train_gpu():
    sys.path.append(str(ROOT / 'scripts'))
    from scripts import train_gpu as module
    return module


def test_resume_restores_replay_and_closes_open_episodes(tmp_path):
    source = filled_replay()
    source.save(tmp_path / 'replay.pt', iteration=7, checkpoint='last.pt')
    target = BatchedOnlineReplay(2, 3, torch.device('cpu'), capacity=16)
    train_gpu().restore_replay(target, tmp_path / 'last.pt', {'iteration': 7})
    assert torch.equal(target.buffer.s, source.buffer.s)
    assert torch.equal(target.buffer.ep_id, source.buffer.ep_id)
    assert torch.equal(target.buffer.ep_counter, source.buffer.ep_counter + 1)
    assert target.frame.tolist() == [0, 0]
    assert torch.equal(target.episode, source.episode + 1)
    target.add(torch.zeros(2, 3), torch.zeros(2, 6), torch.zeros(2, 6), torch.zeros(2, 7), torch.zeros(2),
               torch.ones(2, 3), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    # The first post-resume row has a new episode id, so no window joins it to pre-resume rows.
    rows = torch.arange(2)
    p = (target.buffer.ptr - 1) % target.buffer.T
    assert (target.buffer.ep_id[rows, p] != target.buffer.ep_id[rows, p - 1]).all()


def test_resume_rejects_replay_from_another_checkpoint(tmp_path):
    filled_replay().save(tmp_path / 'replay.pt', iteration=5, checkpoint='checkpoint_00000005.pt')
    target = BatchedOnlineReplay(2, 3, torch.device('cpu'), capacity=16)
    with pytest.raises(ValueError, match='belongs to'):
        train_gpu().restore_replay(target, tmp_path / 'last.pt', {'iteration': 7})
    with pytest.raises(FileNotFoundError):
        train_gpu().restore_replay(target, tmp_path / 'missing' / 'last.pt', {'iteration': 7})


def test_group_substep_from_config(monkeypatch):
    captured = {}

    def init(self, backend_root, groups, **kwargs):
        captured['groups'] = groups

    monkeypatch.setattr(MultiConditionEnvironment, '__init__', init)
    config = {'force_limits': [50.] * 3 + [5.] * 3, 'groups': [
        {'name': 'a', 'scene': 'a.pt', 'worlds': 2, 'sim_dt': 8e-6},
        {'name': 'b', 'scene': 'b.pt', 'worlds': 2}]}
    MultiConditionEnvironment.from_config(ROOT / 'backends/mujoco', config)
    assert [g.sim_dt for g in captured['groups']] == [8e-6, None]


def test_s0_configs_use_the_stable_potato_substep():
    for path in (ROOT / 'configs/multi_condition').glob('s0_*.json'):
        for group in json.loads(path.read_text())['groups']:
            if 'potato_003' in group['scene']:
                assert group['sim_dt'] == 8e-6, path.name
