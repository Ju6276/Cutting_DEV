"""Route B1 force-risk screen: WM features, episode-unit accounting and development selection."""
import numpy as np
import torch

from cutting_world_model import force_risk as fr
from test_critic_target import control_learner
from test_wm_terminal_gradient import perturbed


def test_wm_forward_matches_transition():
    learner, replay = control_learner()
    perturbed(learner)
    wm = learner.wm
    batch = replay.sample_windows(5, 1)
    state, action, goal = batch.states[0], 1.5 * torch.randn(5, 6), batch.xi
    hidden, predicted = fr.wm_forward(wm, state, action, goal)
    next_state, _ = wm.transition(state, action, goal)
    assert hidden.shape == (5, wm.dynamics.mlp.net[-1].in_features)
    assert torch.allclose(predicted, fr.utilization(wm.spec, next_state), atol=1e-6)


def record(name, true, force_limit, phase=None):
    true = np.asarray(true, float)
    return {'id': name, 'force_limit': force_limit, 'true': true, 'current': np.zeros_like(true),
            'phase': np.zeros(len(true), int) if phase is None else np.asarray(phase)}


def test_episode_unit_detection_and_false_alarms():
    records = [record('fl', [0.2, 0.4, 1.3], True), record('ok', [0.1, 0.9, 0.2], False),
               record('ok2', [0.1, 0.1], False)]
    fr.check_force_limit_labels(records)
    # Early flags in a force-limit episode do not count as a detection.
    flags = {'fl': [True, False, False], 'ok': [False, True, False], 'ok2': [False, False]}
    row = fr.summarize_setting(records, flags)
    assert row['detected'] == 0 and row['force_limit_episodes'] == 1
    assert row['false_alarm_episodes'] == 1 and row['other_episodes'] == 2
    assert row['step_rates']['all'] == {'intervals': 5, 'flagged': 1, 'rate': 0.2}
    flags['fl'] = [False, False, True]
    assert fr.summarize_setting(records, flags)['detected_ids'] == ['fl']


def test_force_limit_label_check_rejects_earlier_violation():
    import pytest
    with pytest.raises(ValueError):
        fr.check_force_limit_labels([record('fl', [1.2, 1.3], True)])
    with pytest.raises(ValueError):
        fr.check_force_limit_labels([record('ok', [0.2, 1.1], False)])


def test_selection_respects_budget_and_prefers_recall_then_fewer_alarms():
    records = [record('fl', [0.2, 1.3], True)] + [record('ok%d' % i, [0.2, 0.5], False) for i in range(10)]
    scores = {'fl': np.array([0.1, 0.8])}
    scores.update({'ok%d' % i: np.array([0.1, 0.3 + 0.05 * i]) for i in range(10)})
    settings = [(tau, {k: s > tau for k, s in scores.items()}) for tau in (0.2, 0.6, 0.7, 0.9)]
    chosen = fr.select_setting(records, settings, budget=0.10)
    # 0.2 and 0.6 alarm on too many episodes; 0.7 detects the event with one false alarm.
    assert chosen['parameter'] == 0.7
    assert chosen['dev']['detected'] == 1 and chosen['dev']['false_alarm_episodes'] == 1
    assert fr.select_setting(records, settings[:2], budget=0.10) is None


def test_pinball_is_asymmetric():
    target = torch.zeros(4)
    under, over = fr.pinball(-torch.ones(4), target, 0.95), fr.pinball(torch.ones(4), target, 0.95)
    assert torch.isclose(under, torch.tensor(0.95)) and torch.isclose(over, torch.tensor(0.05))
