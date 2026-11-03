from cutting_world_model.online_schedule import OnlineSchedule
import pytest


def test_original_warmup_and_fractional_update_cadence():
    schedule = OnlineSchedule()
    updates = []
    for step in range(1, 51):
        count = schedule.after_environment_step()
        updates.extend([step] * count)
    assert updates == [42, 44, 47, 49]


def test_resume_refills_replay_before_updates():
    schedule = OnlineSchedule(start_iteration=4000)
    for _ in range(41):
        assert schedule.after_environment_step() == 0
    assert schedule.after_environment_step() == 1


@pytest.mark.parametrize('worlds', [1, 32, 1024, 4096])
def test_equal_physical_data_gives_equal_updates(worlds):
    schedule = OnlineSchedule(worlds=worlds)
    events = []
    for _ in range(50 * 4096 // worlds):
        count = schedule.after_environment_step()
        events.extend([schedule.transitions] * count)
    assert events == [step * 4096 for step in [42, 44, 47, 49]]
    assert schedule.source_iteration == 50
    assert schedule.replay_capacity * worlds == 4096000


def test_resume_preserves_partial_source_step_and_refills():
    original = OnlineSchedule(worlds=1024)
    for _ in range(169):
        original.after_environment_step()
    restored = OnlineSchedule.resume(original.state_dict(), worlds=1024)
    assert restored.transitions == 169 * 1024
    assert restored.source_iteration == 42
    for _ in range(167):
        assert restored.after_environment_step() == 0
    assert restored.after_environment_step() == 1
    assert restored.source_iteration == 84


def test_multiple_source_updates_keep_their_own_lr_steps():
    schedule = OnlineSchedule(worlds=16384)
    for _ in range(10):
        schedule.after_environment_step()
    assert schedule.after_environment_step() == 2
    assert schedule.update_source_iterations == [42, 44]


def test_checkpoint_boundary_fires_once_for_small_world_count():
    schedule = OnlineSchedule(start_iteration=15999, worlds=1024)
    assert not schedule.checkpoint_due
    schedule.after_environment_step()
    assert schedule.checkpoint_due
    schedule.after_environment_step()
    assert not schedule.checkpoint_due


def test_legacy_checkpoint_cannot_silently_use_new_units():
    with pytest.raises(ValueError, match='physical-transition schedule'):
        OnlineSchedule.resume({'iteration': 872}, worlds=1024)
