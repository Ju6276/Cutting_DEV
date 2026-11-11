"""Per-episode records and aggregate episode metrics of the online training loop."""


def ending(success, collision, outside, truncated):
    return ('success' if success else 'force_limit' if collision else 'workspace' if outside
            else 'time_limit' if truncated else 'other')


def episode_metrics(prefix, mask, returns, lengths, info, truncated, peak_material, peak_ratio):
    """Means over the episodes ending in ``mask`` (at least one)."""
    return {prefix + 'episodes': int(mask.sum()),
            prefix + 'return_mean': float(returns[mask].mean()),
            prefix + 'length_mean': float(lengths[mask].float().mean()),
            prefix + 'success_rate': float(info['success'][mask].float().mean()),
            prefix + 'collision_rate': float(info['collision'][mask].float().mean()),
            prefix + 'outside_rate': float(info['outside'][mask].float().mean()),
            prefix + 'truncated_rate': float(truncated[mask].float().mean()),
            prefix + 'peak_material_force_mean': float(peak_material[mask].float().mean()),
            prefix + 'peak_limit_ratio_mean': float(peak_ratio[mask].float().mean())}


def ended_metrics(ended, returns, lengths, info, truncated, peak_material, peak_ratio, groups=None, group_index=None):
    """Overall and (with ``groups``) per-object episode metrics for this iteration."""
    if not bool(ended.any()):
        return {}
    args = (returns, lengths, info, truncated, peak_material, peak_ratio)
    metrics = episode_metrics('episode/', ended, *args)
    if groups is not None:
        for g, group in enumerate(groups):
            mask = ended & (group_index == g)
            if bool(mask.any()):
                metrics.update(episode_metrics(f'episode_{group.name}/', mask, *args))
    return metrics


def episode_row(e, episode, iteration, returns, lengths, info, truncated, peak_material, peak_ratio,
                condition=None):
    row = {'env': e, 'episode': episode, 'iteration': iteration,
           'return': float(returns[e]), 'steps': int(lengths[e]),
           'ending': ending(bool(info['success'][e]), bool(info['collision'][e]),
                            bool(info['outside'][e]), bool(truncated[e])),
           'success': bool(info['success'][e]), 'collision': bool(info['collision'][e]),
           'outside': bool(info['outside'][e]), 'truncated': bool(truncated[e]),
           'released_springs': int(info['released_springs'][e]),
           'peak_material_force_n': float(peak_material[e]), 'peak_limit_ratio': float(peak_ratio[e])}
    if condition is not None:
        row['condition'] = condition
    return row
