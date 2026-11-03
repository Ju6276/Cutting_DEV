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
    restored = OnlineSchedule.resume(original.state_dict(), worlds=1024)
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
