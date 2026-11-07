"""Observed episode returns and replay sampling diagnostics."""
import math

import torch


@torch.no_grad()
def complete_episode_returns(rewards, terminated, truncated, gamma):
    """G_t uses rewards[t] = r_(t+1), with zero continuation at natural termination."""
    rewards = torch.as_tensor(rewards)
    terminated = torch.as_tensor(terminated, dtype=torch.bool, device=rewards.device)
    truncated = torch.as_tensor(truncated, dtype=torch.bool, device=rewards.device)
    if rewards.ndim != 1 or not rewards.numel() or not rewards.is_floating_point():
        raise ValueError('Rewards must be a nonempty floating-point episode vector')
    if terminated.shape != rewards.shape or truncated.shape != rewards.shape:
        raise ValueError('Episode flags must align with rewards')
    if not 0 <= gamma <= 1 or not torch.isfinite(rewards).all():
        raise ValueError('Returns require finite rewards and gamma in [0, 1]')
    if terminated[:-1].any() or truncated[:-1].any():
        raise ValueError('Episode contains a transition after an episode boundary')
    if not terminated[-1] or truncated[-1]:
        raise ValueError('Monte Carlo returns require a complete naturally terminated episode')
    returns = torch.empty_like(rewards)
    continuation = rewards.new_zeros(())
    for t in range(len(rewards) - 1, -1, -1):
        continuation = rewards[t] + gamma * continuation
        returns[t] = continuation
    return returns


@torch.no_grad()
def reward_under(heads, recorded_heads, rewards, next_observations):
    """Recorded rewards re-expressed under ``heads``' force penalty.

    The reward variants differ only in the force term, a function of the observation
    after each transition, so r' = r - w f_recorded(o_(t+1)) + w f(o_(t+1)).
    """
    rewards = torch.as_tensor(rewards)
    if next_observations.shape[:-1] != rewards.shape:
        raise ValueError('One next observation per recorded reward is required')
    if heads.env.get('force_penalty') == recorded_heads.env.get('force_penalty') and \
            heads.env.get('force_penalty_scale') == recorded_heads.env.get('force_penalty_scale'):
        return rewards
    observation = next_observations.to(device=rewards.device, dtype=torch.float64)
    delta = heads.weighted_force_reward(observation) - recorded_heads.weighted_force_reward(observation)
    return rewards + delta.to(rewards.dtype)


@torch.no_grad()
def value_return_metrics(values, returns):
    """Compare nonterminal-state predictions with observed behavior-policy returns."""
    values = torch.as_tensor(values).detach().double()
    returns = torch.as_tensor(returns, device=values.device).detach().double()
    if values.ndim != 1 or values.shape != returns.shape or not values.numel():
        raise ValueError('Values and returns must be aligned nonempty episode vectors')
    if not torch.isfinite(values).all() or not torch.isfinite(returns).all():
        raise ValueError('Value comparison requires finite values and returns')
    error = values - returns
    return {
        'transition_count': values.numel(),
        'mc_mean': float(returns.mean()),
        'mc_min': float(returns.min()),
        'mc_max': float(returns.max()),
        'critic_mean': float(values.mean()),
        'critic_min': float(values.min()),
        'critic_max': float(values.max()),
        'bias_mean': float(error.mean()),
        'rmse': float(error.square().mean().sqrt()),
    }


def _expected_reuse(available, draws):
    if available == 0:
        return {'expected_unique': 0., 'expected_repeated_fraction': None}
    unique = 1. if available == 1 else -available * math.expm1(draws * math.log1p(-1. / available))
    return {'expected_unique': unique, 'expected_repeated_fraction': 1. - unique / draws}


@torch.no_grad()
def replay_sampling_metrics(buffer, window, history, wm_batch_size=None, policy_batch_size=None):
    """Counts follow the active window sampler and history overwrite mask; rows include terminal rows."""
    if window < 1 or history < 1:
        raise ValueError('Window and history must be positive')
    if any(n is not None and n < 1 for n in (wm_batch_size, policy_batch_size)):
        raise ValueError('Batch sizes must be positive')
    starts = buffer.valid & ~buffer.terminal
    positions = torch.arange(buffer.T, device=buffer.valid.device)[None]
    distance = (buffer.ptr[:, None] - positions - 1) % buffer.T
    starts &= distance < buffer.T - history - 1
    windows = int(buffer.window_mask(window).sum())
    start_count = int(starts.sum())
    result = {
        'valid_rows': int(buffer.valid.sum()),
        'terminal_rows': int((buffer.valid & buffer.terminal).sum()),
        'valid_windows': windows,
        'valid_starts': start_count,
    }
    for prefix, available, draws in (
        ('wm', windows, wm_batch_size), ('policy', start_count, policy_batch_size)
    ):
        if draws is not None:
            result[prefix + '_batch_size'] = draws
            result.update({prefix + '_' + key: value for key, value in _expected_reuse(available, draws).items()})
    return result


@torch.no_grad()
def sampled_reuse_metrics(flat_indices):
    """Indices identify replay rows as env_id * capacity + row_id."""
    indices = torch.as_tensor(flat_indices)
    if indices.ndim != 1 or not indices.numel() or indices.is_floating_point():
        raise ValueError('Sample indices must be a nonempty integer vector')
    draws = indices.numel()
    unique = torch.unique(indices).numel()
    return {'sample_draws': draws, 'sample_unique': unique,
            'sample_repeated_fraction': 1. - unique / draws}
