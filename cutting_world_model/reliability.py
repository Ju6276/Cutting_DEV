"""World-model reliability check (PLAN): recorded episodes, per-episode prediction metrics,
three-way terminal check and paired bootstrap comparison.

Episode directory layout (as written by the 2026-11-05 fixed-data run and the supplement
collector): ``episodes.jsonl`` with one record per complete episode (``path``, ``split``)
and ``episodes/<path>.npz`` with per-step arrays ``state``, ``next_state``, ``clean_action``,
``action``, ``goal``, ``reward``, ``terminated``, ``truncated``, ``success``, ``collision``,
``outside``.
"""
import json
from pathlib import Path

import numpy as np
import torch

REASONS = ('success', 'collision', 'outside', 'truncated')
ROUTES = ('true_predicate', 'predicted_predicate', 'terminal_head')


def read_index(root):
    path = Path(root) / 'episodes.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_episodes(root, split, device):
    episodes = []
    for item in read_index(root):
        if item['split'] != split:
            continue
        with np.load(Path(root) / 'episodes' / item['path']) as data:
            arrays = {k: data[k] for k in data.files}
        if not np.array_equal(arrays['state'][1:], arrays['next_state'][:-1]):
            raise ValueError('State chain broken in ' + item['path'])
        tensor = lambda a: torch.as_tensor(a, device=device)
        episodes.append({
            'id': item['path'], 'item': item,
            'states': tensor(np.concatenate((arrays['state'][:1], arrays['next_state']), 0)),
            'actions': tensor(arrays['action']), 'clean': tensor(arrays['clean_action']),
            'goals': tensor(arrays['goal']), 'reward': tensor(arrays['reward']),
            **{k: tensor(arrays[k]) for k in ('terminated', 'truncated', 'success', 'collision', 'outside')},
        })
    return episodes


def terminal_reason(ep):
    for key in ('success', 'collision', 'outside'):
        if bool(ep[key][-1]):
            return key
    if bool(ep['truncated'][-1]):
        return 'truncated'
    raise ValueError('Episode %s ends without a terminal reason' % ep['id'])


def build_replay(spec, episodes, device):
    """SequenceReplay with one world per episode, as in the 2026-11-05 fixed-data fit."""
    from .sequence_replay import SequenceReplay
    capacity = max(9, max(len(e['actions']) + 2 for e in episodes))
    replay = SequenceReplay(len(episodes), capacity, spec.state_dim, 6, 7, device)
    add_episodes(replay, episodes)
    return replay


def add_episodes(replay, episodes, first_world=0):
    """Write each recorded episode into its own world ``first_world + i`` of ``replay``."""
    device = replay.device
    for index, ep in enumerate(episodes):
        world = first_world + index
        ids = torch.tensor([world], device=device)
        n = len(ep['actions'])
        if n + 1 > replay.T:
            raise ValueError('Episode longer than replay capacity: ' + str(ep['id']))
        zero = torch.zeros(1, 6, device=device)
        for t in range(n + 1):
            terminal = t == n
            replay.add(ids, ep['states'][t:t + 1], zero if terminal else ep['clean'][t:t + 1],
                       zero if terminal else ep['actions'][t:t + 1], ep['goals'][0:1],
                       torch.tensor([0. if terminal else float(ep['reward'][t])], device=device),
                       torch.tensor([float(bool(ep['terminated'][-1])) if terminal else 0.], device=device),
                       torch.tensor([t], device=device), torch.tensor([world], device=device),
                       torch.tensor([terminal], device=device), torch.tensor([t == 0], device=device))


def predicate(spec, state, previous):
    """Task termination of ``state`` reached from ``previous`` (whose success counter it uses)."""
    collision = (spec.field(state, 'peak_wrench') > spec.force_limits).any(-1)
    done, success, outside, _ = spec.heads.termination(
        spec.field(state, 'observation'), spec.field(previous, 'success_counter'), collision)
    reason = torch.full(done.shape, -1, dtype=torch.long, device=done.device)
    reason = torch.where(outside, 2, reason)
    reason = torch.where(collision, 1, reason)
    reason = torch.where(success, 0, reason)
    return done, reason


def force_summary(spec, state):
    """Largest force component (N) and largest limit ratio over all six components."""
    peak = spec.field(state, 'peak_wrench')
    return peak[..., :3].abs().amax(-1), (peak.abs() / spec.force_limits).amax(-1)


@torch.no_grad()
def rollout(wm, start, actions, goal):
    """Open-loop WM prediction from ``start`` (n, d) over ``actions`` (h, n, 6)."""
    states, logits = [start], []
    state = start
    for action in actions:
        state, _, logit = wm.step(state, action, goal)
        states.append(state)
        logits.append(logit)
    return torch.stack(states), torch.stack(logits)


@torch.no_grad()
def evaluate_episode(wm, ep, horizons=(1, 8)):
    spec = wm.spec
    n = len(ep['actions'])
    reason = terminal_reason(ep)
    recorded_reason = torch.full((n,), -1, dtype=torch.long, device=ep['states'].device)
    for index, key in enumerate(('success', 'collision', 'outside')):
        recorded_reason = torch.where(ep[key].bool(), index, recorded_reason)
    labels = ep['terminated'].bool()
    # Route 1 on every recorded step: the task predicate must reproduce the recorded labels.
    done1, reason1 = predicate(spec, ep['states'][1:], ep['states'][:-1])
    result = {'id': ep['id'], 'reason': reason, 'steps': n,
              'label_mismatches': int((done1 != labels).sum()),
              'reason_mismatches': int(((reason1 != recorded_reason) & labels).sum())}
    for h in horizons:
        if n < h:
            continue
        starts = torch.arange(n - h + 1, device=labels.device)
        states, logits = rollout(wm, ep['states'][starts],
                                 torch.stack([ep['actions'][starts + j] for j in range(h)]),
                                 ep['goals'][starts])
        pred, prev_pred = states[-1], states[-2]
        true, prev_true = ep['states'][starts + h], ep['states'][starts + h - 1]
        q_err = spec.field(pred, 'joint_q') - spec.field(true, 'joint_q')
        tip = 1000 * (spec.fk(spec.field(pred, 'joint_q'))[..., :3] - spec.fk(spec.field(true, 'joint_q'))[..., :3])
        f_pred, r_pred = force_summary(spec, pred)
        f_true, r_true = force_summary(spec, true)
        f_persist, _ = force_summary(spec, ep['states'][starts])
        label = labels[starts + h - 1]
        routes = {'true_predicate': predicate(spec, true, prev_true)[0],
                  'predicted_predicate': predicate(spec, pred, prev_pred)[0],
                  'terminal_head': logits[-1] > 0}
        row = {'windows': int(len(starts)),
               'joint_rmse_rad': float(q_err.square().mean().sqrt()),
               'tip_error_mm': float(tip.norm(dim=-1).mean()),
               'force_error_n': float((f_pred - f_true).mean()),
               'force_abs_error_n': float((f_pred - f_true).abs().mean()),
               'force_underestimate_n': float((f_true - f_pred).clamp_min(0).mean()),
               'persistence_force_abs_error_n': float((f_persist - f_true).abs().mean()),
               'limit_ratio_error': float((r_pred - r_true).mean()),
               'true_limit_steps': int((r_true > 1).sum()),
               'missed_limit_steps': int(((r_true > 1) & (r_pred <= 1)).sum())}
        for name, done in routes.items():
            row[name] = {'terminal_windows': int(label.sum()), 'missed': int((label & ~done).sum()),
                         'nonterminal_windows': int((~label).sum()),
                         'false_alarms': int((~label & done).sum())}
        result['H%d' % h] = row
    # Full open loop: first predicted terminal step against the recorded end.
    states, logits = rollout(wm, ep['states'][:1], ep['actions'][:, None], ep['goals'][:1])
    done2, _ = predicate(spec, states[1:, 0], states[:-1, 0])
    first = lambda d: int(d.nonzero()[0, 0]) + 1 if bool(d.any()) else None
    full = {}
    for name, done in (('predicted_predicate', done2), ('terminal_head', logits[:, 0] > 0)):
        step = first(done)
        full[name] = {'first_terminal_step': step, 'offset_steps': None if step is None else step - n}
    tip = 1000 * (spec.fk(spec.field(states[-1, 0], 'joint_q'))[:3]
                  - spec.fk(spec.field(ep['states'][-1], 'joint_q'))[:3])
    full['end_tip_error_mm'] = float(tip.norm())
    result['full'] = full
    return result


def evaluate(wm, episodes, horizons=(1, 8)):
    return [evaluate_episode(wm, ep, horizons) for ep in episodes]


def metric(row, path):
    value = row
    for key in path.split('/'):
        if value is None or key not in value:
            return None
        value = value[key]
    return value


def summarize(rows, paths):
    """Mean of each metric over episodes, overall and by terminal reason."""
    groups = {'all': rows, **{r: [x for x in rows if x['reason'] == r] for r in REASONS}}
    out = {}
    for group, members in groups.items():
        if not members:
            continue
        out[group] = {'episodes': len(members)}
        for path in paths:
            values = [metric(x, path) for x in members]
            values = [v for v in values if v is not None]
            out[group][path] = float(np.mean(values)) if values else None
    return out


def paired_bootstrap(rows_a, rows_b, path, lower_is_better=True, samples=10000, seed=0):
    """Episode-paired difference a − b of one per-episode metric with a 95% percentile
    interval; classified as improved / worsened / inconclusive for ``a`` against ``b``."""
    by_id = {x['id']: x for x in rows_b}
    diffs = []
    for row in rows_a:
        other = by_id.get(row['id'])
        a, b = metric(row, path), metric(other, path) if other else None
        if a is not None and b is not None:
            diffs.append(a - b)
    if not diffs:
        return {'episodes': 0, 'outcome': 'no data'}
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), (samples, len(diffs)))].mean(1)
    low, high = np.percentile(means, [2.5, 97.5])
    better = (high < 0) if lower_is_better else (low > 0)
    worse = (low > 0) if lower_is_better else (high < 0)
    outcome = 'improved' if better else 'worsened' if worse else 'inconclusive'
    return {'episodes': len(diffs), 'mean_difference': float(diffs.mean()),
            'interval_95': [float(low), float(high)], 'outcome': outcome}


@torch.no_grad()
def imagine_closed_loop(wm, policy, state, history, goal, horizon):
    """Closed-loop WM rollout with ``policy`` under the rule of ``OnlineLearner.imagine``:
    task terminals of the predicted state absorb (reward counted once, state held).
    ``history`` is the policy's observation history (time, n, obs). Returns rewards (h, n),
    states (h+1, n, d), the first absorbed step (0 = none) and its predicate reason."""
    spec = wm.spec
    n = state.shape[0]
    absorbed = torch.zeros(n, dtype=torch.bool, device=state.device)
    first = torch.zeros(n, dtype=torch.long, device=state.device)
    reason = torch.full((n,), -1, dtype=torch.long, device=state.device)
    states, rewards = [state], []
    for step in range(1, horizon + 1):
        action = policy(history, goal)
        next_state, reward, _ = wm.predict(state, action, goal)
        done, why = predicate(spec, next_state, state)
        reward = torch.where(absorbed, torch.zeros_like(reward), reward)
        next_state = torch.where(absorbed[:, None], state, next_state)
        new = done & ~absorbed
        first = torch.where(new, step, first)
        reason = torch.where(new, why, reason)
        absorbed = absorbed | done
        state = next_state
        history = torch.cat((history[1:], spec.field(state, 'observation')[None]), 0)
        states.append(state)
        rewards.append(reward)
    return torch.stack(rewards), torch.stack(states), first, reason
