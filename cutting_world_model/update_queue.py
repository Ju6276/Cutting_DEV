"""Checkpoint unstarted learning rounds separately from collection credits."""
import copy
import math
import warnings


def _clock(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid queued update clock')
    return value


class UpdateQueue:
    def __init__(self):
        self.pending = []
        self.active = None
        self.discarded = []
        self.provenance = 'fresh'

    def enqueue(self, clocks):
        if self.pending or self.active is not None:
            raise RuntimeError('Drain pending learning rounds before collecting again')
        self.pending = [_clock(x) for x in clocks]

    def begin(self):
        if self.active is not None or not self.pending:
            raise RuntimeError('No unstarted update available')
        self.active = self.pending.pop(0)
        return self.active

    def complete(self):
        if self.active is None:
            raise RuntimeError('No active update to acknowledge')
        self.active = None

    def state_dict(self):
        return copy.deepcopy(dict(version=1, pending=self.pending, active=self.active,
                                  discarded=self.discarded, provenance=self.provenance))

    @classmethod
    def resume(cls, saved, *, replay_restored=True):
        result = cls()
        if saved is None:
            warnings.warn('Legacy checkpoint lacks pending learning rounds; their count '
                          'cannot be recovered. Starting an empty queue, not claiming exact '
                          'update continuity.', RuntimeWarning, stacklevel=2)
            result.provenance = 'legacy_missing_queue'
            return result
        if saved.get('version') != 1 or not isinstance(saved.get('pending'), list):
            raise ValueError('Unsupported or invalid update queue')
        result.pending = [_clock(x) for x in saved['pending']]
        active = saved['active']
        if active is not None:
            _clock(active)
        result.discarded = copy.deepcopy(saved['discarded'])
        if not isinstance(result.discarded, list):
            raise ValueError('Invalid discarded update record')
        result.provenance = 'saved_queue_v1'
        if active is not None:
            # Optimizer steps may already have changed parameters. Replaying the
            # entire round would silently duplicate those steps. Preserve its
            # partial result and disclose the dropped unfinished remainder.
            result.discarded.append(dict(clock=active, reason='interrupted_round'))
            warnings.warn('Checkpoint interrupted an active learning round. Keeping its '
                          'saved optimizer steps without replaying the whole round; only '
                          'unstarted rounds are recovered. This is not exact mid-round '
                          'continuation.', RuntimeWarning, stacklevel=2)
        if not replay_restored and result.pending:
            result.discarded.extend(dict(clock=x, reason='empty_replay_refill') for x in result.pending)
            warnings.warn('Dropping pending learning rounds because replay is intentionally '
                          'empty; refill before new updates.', RuntimeWarning, stacklevel=2)
            result.pending = []
        return result


def drain_updates(queue, update, completed, before_update=None):
    """A completed-round checkpoint sees only the remaining unstarted rounds."""
    while queue.pending:
        if before_update is not None:
            before_update()
        clock = queue.begin()
        result = update(clock)
        queue.complete()
        completed(clock, result)
