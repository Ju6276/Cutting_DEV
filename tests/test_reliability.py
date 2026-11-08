"""Alignment checks of the WM reliability evaluation on an episode the WM generates itself."""
import numpy as np
import torch

from cutting_world_model.reliability import (evaluate_episode, paired_bootstrap, predicate,
                                             rollout, summarize)
from test_critic_target import control_learner
from test_wm_terminal_gradient import perturbed


def self_generated_episode(learner, replay, steps=12, seed=4):
    """States from the WM's own open loop; labels from the task predicate on those states."""
    wm = learner.wm
    torch.manual_seed(seed)
    starts = replay.sample_starts(1, learner.pcfg.history)
    actions = 0.3 * torch.randn(steps, 1, 6)
    states, _ = rollout(wm, starts.state, actions, starts.xi)
    done, reason = predicate(wm.spec, states[1:, 0], states[:-1, 0])
    end = int(done.nonzero()[0, 0]) + 1 if bool(done.any()) else steps
    flags = {k: torch.zeros(end, dtype=torch.bool) for k in ('success', 'collision', 'outside', 'truncated')}
    if bool(done.any()):
        flags[('success', 'collision', 'outside')[int(reason[end - 1])]][-1] = True
    else:
        flags['truncated'][-1] = True
    return {'id': 'self', 'states': states[:end + 1, 0], 'actions': actions[:end, 0],
            'goals': starts.xi.expand(end, -1), 'terminated': done[:end].clone(), **flags}


def test_self_generated_episode_has_zero_error_and_consistent_routes():
    learner, replay = control_learner()
    perturbed(learner)
    ep = self_generated_episode(learner, replay)
    row = evaluate_episode(learner.wm, ep, horizons=(1, 3))
    assert row['label_mismatches'] == 0 and row['reason_mismatches'] == 0
    for h in ('H1', 'H3'):
        assert row[h]['joint_rmse_rad'] < 1e-5 and row[h]['tip_error_mm'] < 1e-3
        assert abs(row[h]['force_error_n']) < 1e-4 and row[h]['missed_limit_steps'] == 0
        # With predicted = true states, routes 1 and 2 agree exactly.
        assert row[h]['predicted_predicate'] == row[h]['true_predicate']
        assert row[h]['true_predicate']['missed'] == 0 and row[h]['true_predicate']['false_alarms'] == 0
    if ep['terminated'].any():
        assert row['full']['predicted_predicate']['offset_steps'] == 0


def test_summary_and_paired_bootstrap_outcomes():
    rows_b = [{'id': i, 'reason': 'success', 'H8': {'e': 1.0 + 0.01 * i}} for i in range(20)]
    better = [{'id': i, 'reason': 'success', 'H8': {'e': 0.5 + 0.01 * i}} for i in range(20)]
    worse = [{'id': i, 'reason': 'success', 'H8': {'e': 1.5 + 0.01 * i}} for i in range(20)]
    rng = np.random.default_rng(0)
    noise = [{'id': i, 'reason': 'success', 'H8': {'e': 1.0 + 0.01 * i + rng.normal(0, .3)}} for i in range(20)]
    assert paired_bootstrap(better, rows_b, 'H8/e')['outcome'] == 'improved'
    assert paired_bootstrap(worse, rows_b, 'H8/e')['outcome'] == 'worsened'
    assert paired_bootstrap(noise, rows_b, 'H8/e')['outcome'] == 'inconclusive'
    assert paired_bootstrap(better, rows_b, 'H8/e', lower_is_better=False)['outcome'] == 'worsened'
    summary = summarize(rows_b + [{'id': 99, 'reason': 'collision', 'H8': {'e': 3.0}}], ['H8/e'])
    assert summary['collision'] == {'episodes': 1, 'H8/e': 3.0}
    assert summary['all']['episodes'] == 21
