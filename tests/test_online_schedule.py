import pytest

from cutting_world_model.online_schedule import OnlineSchedule


@pytest.mark.parametrize('worlds', [1, 32, 1024, 4096])
def test_source_vector_step_cadence_is_independent_of_world_count(worlds):
    schedule = OnlineSchedule(worlds=worlds)
    updates = []
    for step in range(1, 51):
        count = schedule.after_environment_step()
        assert schedule.update_iterations == [step] * count
        updates.extend(schedule.update_iterations)
    assert updates == [42, 44, 47, 49]
    assert schedule.iteration == 50
    assert schedule.transitions == 50 * worlds
    assert schedule.replay_capacity == 1000


def test_resume_refills_empty_replay_and_retains_global_lr_iteration():
    original = OnlineSchedule(worlds=1024)
    for _ in range(169):
        original.after_environment_step()
    restored = OnlineSchedule.resume(original.state_dict(), worlds=1024, replay_restored=False)
    assert restored.transitions == 169 * 1024
    for _ in range(41):
        assert restored.after_environment_step() == 0
    assert restored.after_environment_step() == 1
    assert restored.update_iterations == [211]
    assert restored.transitions == 211 * 1024


def test_checkpoint_boundary_uses_vector_steps():
    schedule = OnlineSchedule(start_iteration=3999, worlds=1024)
    assert not schedule.checkpoint_due
    schedule.after_environment_step()
    assert schedule.checkpoint_due
    schedule.after_environment_step()
    assert not schedule.checkpoint_due


@pytest.mark.parametrize('saved', [
    {'iteration': 872},
    {'unit': 'physical_transitions_v1', 'iteration': 122, 'worlds': 1024,
     'transitions': 124928, 'reference_worlds': 4096},
])
def test_rescaled_checkpoints_require_explicit_conversion(saved):
    with pytest.raises(ValueError, match='vector-step schedule'):
        OnlineSchedule.resume(saved, worlds=1024)


def test_resume_rejects_changed_collection_regime():
    with pytest.raises(ValueError, match='environment count'):
        OnlineSchedule.resume(OnlineSchedule(worlds=1024).state_dict(), worlds=1)


@pytest.mark.parametrize('split', [0, 13, 39, 40, 41, 42, 169])
@pytest.mark.parametrize('updates_per_step', [.4, 1.7])
def test_resume_with_replay_preserves_warmup_phase_and_future_updates(split, updates_per_step):
    import json
    from dataclasses import asdict
    from third_party.supermimic.schedule_config import ScheduleCfg
    cfg = ScheduleCfg(updates_per_env_step=updates_per_step, checkpoint_interval=11)
    original = OnlineSchedule(config=cfg, worlds=32)
    for _ in range(split):
        original.after_environment_step()
    # Serialized state, including a fractional phase and nondefault config.
    resumed = OnlineSchedule.resume(json.loads(json.dumps(original.state_dict())), worlds=32)
    assert asdict(resumed.config) == asdict(cfg)
    for step in range(140):
        assert resumed.after_environment_step() == original.after_environment_step()
        assert resumed.update_iterations == original.update_iterations
        assert resumed.checkpoint_due == original.checkpoint_due
        assert resumed.state_dict() == original.state_dict()
        if step % 7 == 0:
            resumed = OnlineSchedule.resume(resumed.state_dict(), worlds=32)


def test_resume_during_explicit_replay_refill_keeps_original_refill_deadline():
    original = OnlineSchedule(start_iteration=1000, worlds=32)
    for _ in range(13):
        original.after_environment_step()
    resumed = OnlineSchedule.resume(original.state_dict(), worlds=32)
    for _ in range(70):
        assert resumed.after_environment_step() == original.after_environment_step()
        assert resumed.state_dict() == original.state_dict()


def test_legacy_checkpoint_migration_is_explicit_and_does_not_repeat_warmup():
    legacy = dict(unit='vector_steps_v1', worlds=32, iteration=169, transitions=5408)
    with pytest.warns(RuntimeWarning, match='not exact cadence recovery'):
        resumed = OnlineSchedule.resume(legacy, worlds=32)
    assert resumed.state_dict()['resume_provenance'] == 'legacy_defaults_origin0_phase0'
    assert [resumed.after_environment_step() for _ in range(3)] == [0, 0, 1]
    fresh = OnlineSchedule.resume(legacy, worlds=32, replay_restored=False)
    assert fresh.start_iteration == 169
    assert all(fresh.after_environment_step() == 0 for _ in range(41))
    assert fresh.after_environment_step() == 1


@pytest.mark.parametrize('key,value', [('accumulator', float('nan')), ('accumulator', 1.),
    ('accumulator', -.1), ('warmup_start_iteration', -1), ('state_version', 42)])
def test_invalid_saved_cadence_is_rejected(key, value):
    saved = OnlineSchedule(worlds=1).state_dict()
    saved[key] = value
    with pytest.raises(ValueError):
        OnlineSchedule.resume(saved, worlds=1)
