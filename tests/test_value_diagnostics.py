import pytest
import torch

from cutting_world_model.value_diagnostics import (
    complete_episode_returns, replay_sampling_metrics, sampled_reuse_metrics,
    value_return_metrics,
)
from third_party.supermimic.replay_buffer import ReplayBuffer


def test_returns_use_next_reward_and_zero_terminal_continuation():
    rewards = torch.tensor([-1., -2., 100.], dtype=torch.float64)
    returns = complete_episode_returns(rewards, [False, False, True], [False] * 3, .5)
    torch.testing.assert_close(returns, torch.tensor([23., 48., 100.], dtype=torch.float64))
    metrics = value_return_metrics(returns - 7., returns)
    assert metrics['bias_mean'] == -7.
    assert metrics['rmse'] == 7.


@pytest.mark.parametrize('terminated,truncated', [
    ([False, False], [False, True]),
    ([False, False], [False, False]),
    ([True, True], [False, False]),
])
def test_partial_or_cross_episode_paths_are_not_zero_bootstrap_mc(terminated, truncated):
    with pytest.raises(ValueError):
        complete_episode_returns(torch.tensor([-1., -100.]), terminated, truncated, .99)


def test_replay_counts_match_actual_sampler_masks_after_ring_overwrite():
    buffer = ReplayBuffer(1, 12, 1, 1, 0, 'cpu')
    env = torch.tensor([0])
    for i in range(18):
        terminal = i in (5, 14)
        buffer.add(env, torch.tensor([[float(i)]]), torch.zeros(1, 1), torch.zeros(1, 1),
                   torch.empty(1, 0), torch.zeros(1), torch.tensor([float(terminal)]),
                   torch.tensor([i]), torch.zeros(1, dtype=torch.long),
                   torch.tensor([terminal]), torch.tensor([i in (0, 6, 15)]))
        if terminal:
            buffer.end_episode(env)
    masks = []
    original = buffer._sample_from_mask

    def capture(mask, n):
        masks.append(mask.clone())
        return original(mask, n)

    buffer._sample_from_mask = capture
    before = buffer.valid.clone()
    metrics = replay_sampling_metrics(buffer, window=2, history=3, wm_batch_size=8, policy_batch_size=8)
    buffer.sample_windows(8, 2)
    buffer.sample_starts(8, 3)
    assert metrics['valid_windows'] == int(masks[0].sum())
    assert metrics['valid_starts'] == int(masks[1].sum())
    assert metrics['valid_rows'] == 12
    assert metrics['terminal_rows'] == 1
    assert torch.equal(before, buffer.valid)
    reuse = sampled_reuse_metrics(torch.tensor([0, 0, 1, 13]))
    assert reuse == {'sample_draws': 4, 'sample_unique': 3, 'sample_repeated_fraction': .25}
