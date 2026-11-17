"""Per-episode records and aggregate episode metrics of the online training loop."""


def ending(success, collision, outside, truncated, timeout=False, object_slip_failure=False, tool_motion_failure=False):
    return ('success' if success else 'force_limit' if collision else 'workspace' if outside
            else 'tool_motion' if tool_motion_failure else 'object_motion' if object_slip_failure else 'time_limit' if truncated or timeout else 'other')


def episode_metrics(prefix, mask, returns, lengths, info, truncated, peak_material, peak_ratio):
    """Means over the episodes ending in ``mask`` (at least one)."""
    return {prefix + 'episodes': int(mask.sum()),
            prefix + 'return_mean': float(returns[mask].mean()),
            prefix + 'length_mean': float(lengths[mask].float().mean()),
            prefix + 'success_rate': float(info['success'][mask].float().mean()),
            prefix + 'collision_rate': float(info['collision'][mask].float().mean()),
            prefix + 'outside_rate': float(info['outside'][mask].float().mean()),
            prefix + 'truncated_rate': float(truncated[mask].float().mean()),
            **({prefix + 'timeout_rate': float(info['timeout'][mask].float().mean())}
               if 'timeout' in info else {}),
            **({prefix + 'separated_rate': float(info['separated'][mask].float().mean())}
               if 'separated' in info else {}),
            **({prefix + 'object_slip_failure_rate': float(info['object_slip_failure'][mask].float().mean()),
                prefix + 'terminal_object_slip_m_mean': float(info['object_slip_m'][mask].float().mean())}
               if 'object_slip_failure' in info else {}),
            **({prefix + 'tool_motion_failure_rate': float(info['tool_motion_failure'][mask].float().mean()),
                prefix + 'completion_xy_error_m_mean': float(info['completion_xy_error_m'][mask].float().mean())}
               if 'tool_motion_failure' in info else {}),
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
                            bool(info['outside'][e]), bool(truncated[e]),
                            bool(info['timeout'][e]) if 'timeout' in info else False,
                            bool(info['object_slip_failure'][e]) if 'object_slip_failure' in info else False,
                            bool(info['tool_motion_failure'][e]) if 'tool_motion_failure' in info else False),
           'success': bool(info['success'][e]), 'collision': bool(info['collision'][e]),
           'outside': bool(info['outside'][e]), 'truncated': bool(truncated[e]),

           'peak_material_force_n': float(peak_material[e]), 'peak_limit_ratio': float(peak_ratio[e])}
    for key, cast in (('completion_xy_error_m', float), ('tool_motion_failure', bool), ('object_slip_failure', bool), ('object_slip_m', float), ('timeout', bool), ('separated', bool), ('cut_progress', float),
                      ('depth_progress', float), ('released_springs', int),
                      ('material_components', int), ('damage_saturated_fraction', float)):
        if key in info:
            row[key] = cast(info[key][e])
    diagnostics = info.get('separation_diagnostics')
    if diagnostics is not None and diagnostics[e] is not None:
        # BladeMaster pose_depth_v3 cut-topology record: logging only (separated is None).
        row['bm_separation'] = diagnostics[e]
    if condition is not None:
        row['condition'] = condition
    return row
