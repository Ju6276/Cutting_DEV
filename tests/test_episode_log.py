"""Episode records and per-object metrics written by the training loop."""
import json
from types import SimpleNamespace

import pytest
import torch

from cutting_world_model.episode_log import ended_metrics, episode_row


def data():
    ended = torch.tensor([True, False, True, True])
    info = {'success': torch.tensor([True, False, False, False]),
            'collision': torch.tensor([False, False, True, False]),
            'outside': torch.tensor([False, False, False, True]),
            'released_springs': torch.tensor([300, 0, 12, 4])}
    truncated = torch.tensor([False, False, False, False])
    returns = torch.tensor([80., 0., -90., -100.])
    lengths = torch.tensor([20, 3, 5, 9])
    peak = torch.tensor([30., 0., 51., 8.])
    ratio = torch.tensor([.6, 0., 1.02, .16])
    return ended, returns, lengths, info, truncated, peak, ratio


def test_overall_and_per_object_metrics():
    ended, returns, lengths, info, truncated, peak, ratio = data()
    groups = [SimpleNamespace(name='prism'), SimpleNamespace(name='potato')]
    index = torch.tensor([0, 0, 1, 1])
    m = ended_metrics(ended, returns, lengths, info, truncated, peak, ratio, groups, index)
    assert m['episode/episodes'] == 3
    assert m['episode/success_rate'] == pytest.approx(1 / 3)
    assert m['episode_prism/episodes'] == 1 and m['episode_prism/success_rate'] == 1.
    assert m['episode_potato/collision_rate'] == .5 and m['episode_potato/outside_rate'] == .5
    assert m['episode_potato/peak_material_force_mean'] == pytest.approx(29.5)
    assert ended_metrics(torch.zeros(4, dtype=torch.bool), returns, lengths, info, truncated, peak, ratio) == {}


def test_episode_row_endings_are_json():
    ended, returns, lengths, info, truncated, peak, ratio = data()
    rows = [episode_row(e, 0, 7, returns, lengths, info, truncated, peak, ratio, {'group': 'x'}) for e in (0, 2, 3)]
    assert [r['ending'] for r in rows] == ['success', 'force_limit', 'workspace']
    assert rows[1]['peak_limit_ratio'] == pytest.approx(1.02) and rows[0]['condition'] == {'group': 'x'}
    json.dumps(rows)
