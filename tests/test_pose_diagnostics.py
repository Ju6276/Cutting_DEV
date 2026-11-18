"""scripts/diagnose_pose_predictions.py helpers on small synthetic inputs (no physics)."""
import pytest
import torch

from test_pose_depth_v3 import survival_fixture
from scripts.diagnose_pose_predictions import (check_version,
                                               compare_first, confusion, discounted_return_to_go,
                                               event_flags, first_step, replay_rollout, wrench_errors,
                                               stats, finite_json, require_finite)


def test_replay_uses_training_absorption_and_stops_after_it():
    # Row 0: latched, step_count = horizon-3 -> analytic timeout on the 3rd step. Row 1: never ends.
    learner, wm, state, goal = survival_fixture('pose_depth_v3')
    policy, horizon = learner.policy, learner.actor_horizon
    k = 6
    actions = torch.full((k, 2, 6), .3, dtype=state.dtype)
    with torch.no_grad():
        ro = replay_rollout(learner, state, state.repeat(10, 1, 1), goal, actions)
    assert learner.policy is policy and learner.actor_horizon == horizon
    assert ro.states.shape[0] == k + 1
    flags, terminal = event_flags(wm.spec, ro.states[1:])
    everywhere = torch.ones(k, 2, dtype=torch.bool)
    assert first_step(terminal, everywhere).tolist() == [3, 0]
    assert flags['timeout'][2, 0] and not flags['success'][2, 0] and not flags['force'][2, 0]
    # Absorption order: the terminal transition keeps its reward once, later steps are frozen/zero.
    assert ro.rewards[2, 0] != 0 and (ro.rewards[3:, 0] == 0).all()
    for j in range(4, k + 1):
        assert torch.equal(ro.states[j, 0], ro.states[3, 0])
    # Event and latch timing are only read inside the recorded window (k-step ends at the episode end).
    window = torch.arange(1, k + 1)[:, None] <= torch.tensor([2, k])[None]
    assert first_step(terminal, window).tolist() == [0, 0]
    # v3 latched row stays latched through the replay.
    assert wm.spec.contact(ro.states[1:, 0]).all()


def test_first_contact_timing_compares_unlatched_starts_inside_the_window():
    k = 4
    real = torch.zeros(k, 4, dtype=torch.bool)
    wm = torch.zeros(k, 4, dtype=torch.bool)
    real[1:, 0] = True; wm[2:, 0] = True   # physical j=2, WM j=3: timing error +1 step
    wm[:, 1] = True                        # WM latches at j=1, physics never
    real[2:, 2] = True                     # physical j=3, but this window ends at j=2
    real[:, 3] = wm[:, 3] = True           # already latched start: not a first contact
    window = torch.tensor([4, 4, 2, 4])
    valid = torch.arange(1, k + 1)[:, None] <= window[None]
    start_latched = torch.tensor([False, False, False, True])
    out = compare_first(first_step(real, valid), first_step(wm, valid), ~start_latched)
    assert (out['both'], out['wm_only'], out['physical_only'], out['neither']) == (1, 1, 0, 1)
    assert out['timing_error_steps']['mean'] == 1 and out['timing_exact_fraction'] == 0
    assert confusion(real[0], wm[0], ~start_latched) == {'both': 0, 'physical_only': 0, 'wm_only': 1, 'neither': 2}


def test_units_and_discounted_return():
    true = torch.tensor([[1., 2., 3., .1, .2, .3] * 2])
    force, torque = wrench_errors(torch.zeros_like(true), true)
    assert force.item() == pytest.approx(2.) and torque.item() == pytest.approx(.2)  # N and Nm, separately
    rewards = torch.tensor([[1.], [2.], [3.]])
    valid = torch.tensor([[True], [True], [False]])
    assert discounted_return_to_go(rewards, valid, .5)[:, 0].tolist() == [2., 2., 0.]


def test_only_pose_depth_v3_checkpoints_are_read():
    good = {'free_cut_task': {'version': 'pose_depth_v3'}, 'physical_backend': 'mujoco_warp_blademaster'}
    assert check_version(good) == 'pose_depth_v3'
    with pytest.raises(ValueError, match='pose_depth_v2 is read-only history'):
        check_version({**good, 'free_cut_task': {'version': 'pose_depth_v2'}})
    with pytest.raises(ValueError, match='pose_depth_v3 only'):
        check_version({**good, 'free_cut_task': {'version': 'free_cut_v3'}})
    with pytest.raises(ValueError, match='backend'):
        check_version({**good, 'physical_backend': 'gazebo'})


@pytest.mark.parametrize("bad", [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_diagnostic_values_are_not_silently_dropped(bad):
    with pytest.raises(FloatingPointError, match='1/2 non-finite'):
        stats(torch.tensor([1., bad]))
    with pytest.raises(FloatingPointError, match='replayed states'):
        require_finite('replayed states', torch.tensor([bad]))
    with pytest.raises(FloatingPointError, match='refusing to write'):
        finite_json({'prediction_error': bad})


def test_missing_measurement_is_distinct_from_invalid_prediction():
    assert finite_json({'physical_separation': None, 'mean_error': 0.}) == {
        'physical_separation': None, 'mean_error': 0.}
    assert stats(torch.tensor([])) == {'count': 0}
    assert stats(torch.tensor([1., 3.]))['mean'] == 2.
