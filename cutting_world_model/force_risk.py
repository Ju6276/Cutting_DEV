"""Route B1 force-risk screen (PLAN): one-step upper predictions of next-interval
sensor-limit utilization R and their episode-level detection/false-alarm accounting.

R(s) = max_i |peak_wrench_i| / force_limit_i over the six wrench components; R > 1 is
the task's force-limit event. Every score is computed from the recorded state at t and
the executed action a_t, i.e. before the interval t -> t+1 is executed.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .sequence_replay import SequenceReplay
from .reliability import add_episodes

# Observable phase of the current interval, declared before any B1 data access:
# measured utilization of the interval that just ended, R(s_t), in
# [0, 0.1) free/approach, [0.1, 0.5) light load, [0.5, inf) loaded.
PHASE_EDGES = (0.1, 0.5)
PHASES = ('free', 'light', 'loaded')


def utilization(spec, state, goal=None):
    """Sensor-limit utilization of the interval ending at ``state`` (per-goal limits)."""
    return (spec.field(state, 'peak_wrench').abs() / spec.limits(goal)).amax(-1)


def phase(spec, state):
    edges = torch.tensor(PHASE_EDGES, device=state.device, dtype=state.dtype)
    return torch.bucketize(utilization(spec, state), edges, right=True)


def wm_features(wm, state, action, goal):
    """Last hidden layer of the WM dynamics MLP and the predicted next utilization; the same
    computation as ``CuttingWorldModel.transition``. Differentiable in ``action``."""
    spec = wm.spec
    executed = spec.execute(action)
    net = wm.dynamics.mlp.net
    hidden = net[:-1](torch.cat((spec.dynamics_features(state), executed, goal[..., :7]), -1))
    fields = spec.integrate_physical(state, net[-1](hidden))
    next_state, _, _ = spec.derive(state, fields, executed, goal)
    return hidden, utilization(spec, next_state, goal)


@torch.no_grad()
def wm_forward(wm, state, action, goal):
    return wm_features(wm, state, action, goal)


class QuantileHead(nn.Module):
    """Upper-quantile head on frozen WM features: (hidden, predicted R, measured R_t) -> R_{t+1}."""

    def __init__(self, hidden_dim, width=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim + 2, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, hidden, predicted, current):
        return self.net(torch.cat((hidden, predicted[..., None], current[..., None]), -1)).squeeze(-1)


def pinball(prediction, target, quantile):
    error = target - prediction
    return torch.maximum(quantile * error, (quantile - 1) * error).mean()


def load_fit_replay(replay_path, episodes, state_dim, device):
    """Training replay of a run plus recorded episodes, each in an extra world."""
    saved = torch.load(replay_path, map_location='cpu', weights_only=False)
    saved = saved.get('buffer', saved)
    if saved.get('format') != 'sequence_replay_v1':
        raise ValueError('Unsupported replay format')
    tensors = saved['tensors']
    worlds, rows, capacity = tensors['s'].shape[0], saved['rows'], saved['capacity']
    replay = SequenceReplay(worlds + len(episodes), capacity, state_dim, 6, 7, device)
    for name in SequenceReplay.tensors:
        value = tensors[name].to(device)
        target = getattr(replay, name)
        if name in SequenceReplay.tensors[:12]:
            target[:worlds, :rows] = value
        else:
            target[:worlds] = value
    replay.total_rows = saved['total_rows']
    add_episodes(replay, episodes, worlds)
    return replay


@torch.no_grad()
def replay_transitions(replay):
    """(world, row) of every recorded transition row -> row + 1 inside one episode."""
    if bool(replay.valid[:, -1].any()):
        raise ValueError('Wrapped replay rings are not supported')
    ok = (replay.valid[:, :-1] & replay.valid[:, 1:] & ~replay.terminal[:, :-1]
          & (replay.ep_id[:, :-1] == replay.ep_id[:, 1:]))
    return ok.nonzero(as_tuple=True)


@torch.no_grad()
def episode_records(spec, episodes):
    """Per episode: measured R_{t+1}, current R_t, phase_t and its force-limit label."""
    out = []
    for ep in episodes:
        states = ep['states']
        out.append({'id': ep['id'], 'force_limit': bool(ep['collision'][-1]),
                    'reason': next((k for k in ('success', 'collision', 'outside') if bool(ep[k][-1])),
                                   'truncated'),
                    'true': utilization(spec, states[1:]).cpu().numpy(),
                    'current': utilization(spec, states[:-1]).cpu().numpy(),
                    'phase': phase(spec, states[:-1]).cpu().numpy()})
    return out


def check_force_limit_labels(records, spec=None):
    """The recorded force-limit ending must be the first interval with R > 1."""
    for r in records:
        above = np.nonzero(r['true'] > 1)[0]
        if r['force_limit'] and (len(above) == 0 or above[0] != len(r['true']) - 1):
            raise ValueError('Force-limit episode %s: first R > 1 is not its last interval' % r['id'])
        if not r['force_limit'] and len(above):
            raise ValueError('Episode %s has R > 1 without a force-limit ending' % r['id'])


def episode_outcomes(records, flags):
    """Episode-unit accounting (PLAN B1). ``flags`` maps id -> bool per interval.

    Force-limit episode: detected iff the prediction for its first violating interval
    (the last recorded interval) is flagged. Other episodes: one false alarm if any of
    their recorded intervals is flagged."""
    detected, alarmed = {}, {}
    for r in records:
        f = np.asarray(flags[r['id']], dtype=bool)
        if r['force_limit']:
            detected[r['id']] = bool(f[-1])
        else:
            alarmed[r['id']] = bool(f.any())
    return detected, alarmed


def step_rates(records, flags):
    """Per-interval flag rates on non-force-limit episodes, overall and by phase."""
    out = {}
    for name, keep in [('all', None)] + list(enumerate(PHASES)):
        n = k = 0
        for r in records:
            if r['force_limit']:
                continue
            mask = np.ones(len(r['true']), bool) if keep is None else r['phase'] == name
            n += int(mask.sum())
            k += int(np.asarray(flags[r['id']], bool)[mask].sum())
        label = 'all' if keep is None else keep
        out[label] = {'intervals': n, 'flagged': k, 'rate': k / n if n else None}
    return out


def bound_quality(records, upper):
    """Empirical coverage P(upper >= R_{t+1}) over intervals and slack on force-limit events."""
    covered = total = 0
    slack = []
    for r in records:
        u = np.asarray(upper[r['id']])
        covered += int((u >= r['true']).sum())
        total += len(u)
        if r['force_limit']:
            slack.append(float(u[-1] - r['true'][-1]))
    return {'interval_coverage': covered / total if total else None,
            'event_slack_median': float(np.median(slack)) if slack else None,
            'event_slack': slack}


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def summarize_setting(records, flags):
    detected, alarmed = episode_outcomes(records, flags)
    k, n = sum(detected.values()), len(detected)
    a, m = sum(alarmed.values()), len(alarmed)
    return {'detected': k, 'force_limit_episodes': n, 'recall': k / n if n else None,
            'recall_95': wilson(k, n), 'false_alarm_episodes': a, 'other_episodes': m,
            'false_alarm_rate': a / m if m else None, 'false_alarm_95': wilson(a, m),
            'step_rates': step_rates(records, flags),
            'detected_ids': sorted(i for i, v in detected.items() if v),
            'false_alarm_ids': sorted(i for i, v in alarmed.items() if v)}


def select_setting(records, settings, budget):
    """Development choice of one operating setting from ``(parameter, flags)`` pairs: largest
    episode recall with episode false-alarm rate <= budget; ties go to fewer false-alarm
    episodes, then to the later (more conservative) parameter in the given order."""
    best = None
    for order, (parameter, flags) in enumerate(settings):
        row = summarize_setting(records, flags)
        if row["false_alarm_rate"] is None or row["false_alarm_rate"] > budget:
            continue
        key = (row["detected"], -row["false_alarm_episodes"], order)
        if best is None or key > best[0]:
            best = (key, parameter, row)
    return None if best is None else {"parameter": best[1], "dev": best[2]}
