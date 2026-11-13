from types import SimpleNamespace

import pytest
import torch

from cutting_world_model.metric_counts import RATE_COUNTS, COUNT_KEYS
from cutting_world_model.policy import SliceItCritic
from cutting_world_model.trainer import OnlineLearner
from third_party.supermimic.trainer_config import WMTrainerCfg
from test_free_cut_v3 import v3_fixture


def batch_fixture(model, state, goal, lengths):
    n, horizon = len(lengths), 8
    states = state.float().repeat(horizon + 1, n, 1)
    valid = torch.arange(horizon)[:, None] < torch.tensor(lengths)[None]
    done = torch.zeros(horizon, n)
    for world, length in enumerate(lengths):
        done[length - 1, world] = 1
        if world % 2 == 0:
            states[length:, world, model.spec.layout['cut_progress']] = 1.
        states[length + 1:, world] = float('nan')  # padding must never be counted
    return SimpleNamespace(states=states, a_exec=torch.zeros(horizon, n, 6),
                           xi=goal.float().repeat(n, 1), done=done, valid_steps=valid)


def test_window_exposes_int64_counts_without_padding():
    model, state, goal = v3_fixture()
    model = model.float()
    batch = batch_fixture(model, state, goal, [1, 2, 8, 3, 4, 1, 7])
    loss, metrics = model.window_loss(batch.states, batch.a_exec, batch.xi, batch.done, batch.valid_steps)
    assert torch.isfinite(loss)
    for key in COUNT_KEYS:
        assert metrics[key].dtype == torch.int64
    assert metrics['metric_transition_count'].item() == 26
    assert metrics['completion_positive_count'].item() == 4
    assert metrics['completion_incomplete_count'].item() == 22
    for rate, numerator in RATE_COUNTS.items():
        assert 0 <= metrics[rate].item() <= 1
        assert metrics[rate].item() == pytest.approx(metrics[numerator].item() / 26)


@pytest.mark.parametrize('microbatch', [2, 7])
def test_fit_pools_integer_counts_across_unequal_batches(microbatch):
    torch.set_num_threads(2)
    model, state, goal = v3_fixture()
    model = model.float()
    batches = [batch_fixture(model, state, goal, lengths) for lengths in
               ([1, 2, 8, 3, 4, 1, 7], [8, 7, 8, 8, 8, 8, 8])]
    sequence = [batches[i % 2] for i in range(21)]
    iterator = iter(sequence)
    learner = OnlineLearner(model, torch.nn.Linear(1, 1), SliceItCritic(model.spec.state_dim, 8),
                            microbatch=microbatch, wm_cfg=WMTrainerCfg(
                                compile=None, lr=0., window=8, batch_size=7, batches_per_update=21))
    result = learner.fit_world_model(SimpleNamespace(sample_windows=lambda *args: next(iterator)))
    expected = sum(int(batch.valid_steps.sum()) for batch in sequence)
    assert result['metric_transition_count'] == result['sampled_transitions'] == expected
    assert result['completion_positive_count'] == 21 * 4
    assert result['sample_terminal_transitions'] == 21 * 7
    for key in COUNT_KEYS:
        assert isinstance(result[key], int)
    for rate, numerator in RATE_COUNTS.items():
        assert result[rate] == result[numerator] / expected
        assert 0 <= result[rate] <= 1
    assert result['completion_positive_fraction'] + result['completion_incomplete_fraction'] == 1.
    # Deliberately unequal valid lengths: this must not be an average of batch rates.
    average = sum(4/int(b.valid_steps.sum()) for b in sequence)/len(sequence)
    assert abs(result['completion_positive_fraction'] - average) > .001
