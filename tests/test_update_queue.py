import pytest
from cutting_world_model.online_schedule import TransitionSchedule
from cutting_world_model.update_queue import UpdateQueue, drain_updates


def test_mid_vector_checkpoint_recovers_remaining_learning_clocks():
    schedule = TransitionSchedule(3072, warmup_transitions=0)
    schedule.after_environment_step()
    assert len(schedule.update_iterations) == 4
    q = UpdateQueue()
    q.enqueue(schedule.update_iterations)
    completed, snapshots = [], []
    drain_updates(q, lambda clock: clock, lambda clock, result: (
        completed.append(clock), snapshots.append(q.state_dict())))
    restored = UpdateQueue.resume(snapshots[0])
    after = []
    drain_updates(restored, lambda clock: clock, lambda clock, result: after.append(clock))
    assert completed[:1] + after == schedule.update_iterations
    assert restored.state_dict()['active'] is None
    assert not restored.pending


def test_budget_stop_before_start_preserves_all_unstarted_work():
    q = UpdateQueue()
    q.enqueue([1., 2.])
    def stop():
        raise TimeoutError('budget')
    with pytest.raises(TimeoutError):
        drain_updates(q, lambda _: pytest.fail('must not start'), lambda *a: None, stop)
    assert q.pending == [1., 2.] and q.active is None
    assert UpdateQueue.resume(q.state_dict()).pending == [1., 2.]


def test_interrupted_round_is_disclosed_and_not_replayed_as_if_atomic():
    q = UpdateQueue()
    q.enqueue([1., 2., 3.])
    calls = []
    def partial(clock):
        calls.append(clock)
        raise InterruptedError('after an optimizer step')
    with pytest.raises(InterruptedError):
        drain_updates(q, partial, lambda *a: None)
    with pytest.warns(RuntimeWarning, match='not exact mid-round'):
        resumed = UpdateQueue.resume(q.state_dict())
    drain_updates(resumed, calls.append, lambda *a: None)
    assert calls == [1., 2., 3.]
    assert resumed.discarded == [{'clock': 1., 'reason': 'interrupted_round'}]


def test_empty_replay_explicitly_discards_pending_learning():
    q = UpdateQueue()
    q.enqueue([2., 3.])
    with pytest.warns(RuntimeWarning, match='intentionally empty'):
        r = UpdateQueue.resume(q.state_dict(), replay_restored=False)
    assert not r.pending
    assert len(r.discarded) == 2


def test_legacy_queue_missing_is_not_silently_called_exact_resume():
    with pytest.warns(RuntimeWarning, match='cannot be recovered'):
        q = UpdateQueue.resume(None)
    assert q.provenance == 'legacy_missing_queue' and not q.pending


@pytest.mark.parametrize('clock', [float('nan'), float('inf'), -1, True, '2'])
def test_invalid_pending_clocks_rejected(clock):
    with pytest.raises(ValueError):
        UpdateQueue().enqueue([clock])


def test_pending_rounds_cannot_be_overwritten_by_another_collection_step():
    q = UpdateQueue()
    q.enqueue([1.])
    with pytest.raises(RuntimeError, match='Drain'):
        q.enqueue([2.])
