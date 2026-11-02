from cutting_world_model.online_schedule import OnlineSchedule


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
