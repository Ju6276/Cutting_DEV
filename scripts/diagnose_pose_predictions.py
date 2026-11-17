#!/usr/bin/env python3
"""Same-trajectory WM prediction diagnostic for pose_depth_v3 checkpoints (the current task).

pose_depth_v2 checkpoints are refused: v2 is read-only history and is not diagnosed here.

Diagnostic only: zero optimizer updates, no task/reward/network change.

1. Collect: the checkpoint's deterministic policy (no exploration noise; executed action =
   spec.execute(policy(history, goal)) as in train_gpu without noise) runs one first episode
   per world on the physical backend named by the checkpoint metadata (DiSECt or BladeMaster).
   Cases are regenerated from --seed / --worlds-per-object with evaluate_pose_policy's
   evaluation_config. Later episodes (after a world's reset) are not used.
2. Predict, per recorded transition (start t of world w), with the checkpoint's own networks:
   open-loop on the recorded executed actions from the recorded start state for up to k steps,
   through trainer.OnlineLearner.imagine (a replay "policy" returns the recorded actions), so
   transition (world_model.step), settlement (spec.derive) and absorption
   (spec.hard_termination; absorbed state frozen, its reward kept once) are the training code.
   Step j=1 is the one-step prediction. A window ends at k or at the physical episode end,
   because no recorded action exists beyond it.
3. Compare against the recorded physical successors: EE position, joints, wrench_history,
   peak_wrench, last_wrench, depth progress, board latch timing and task events.
4. Terminal head: only as a predictor of the analytic termination label of the recorded
   successor (spec.hard_termination). It is not a completion probability of the policy
   objective (v3 imagined survival is analytic: prod(1 - absorbed)).
5. Critic value and the imagined actor objective (trainer.imagine with the policy, terminal
   critic, trainer.discount_weights) vs the realised discounted return-to-go, as gaps.

Physical material separation is recorded separately and never used as a task label.
"""
import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco'), str(ROOT / 'scripts')]

EVENTS = ('success', 'force', 'workspace', 'object_slip', 'tool_motion', 'timeout')
# Real (physical environment) info keys for the events above.
INFO_KEYS = {'success': 'success', 'force': 'collision', 'workspace': 'outside',
             'object_slip': 'object_slip_failure', 'tool_motion': 'tool_motion_failure',
             'timeout': 'timeout'}
BACKENDS = {'mujoco_warp_disect': 'disect', 'mujoco_warp_blademaster': 'blademaster'}
UNITS = {
    'ee_position_error_mm': 'Euclidean EE position error, millimetres',
    'joint_abs_error_max_rad': 'max over 6 joints of |q_wm - q_real|, radians',
    'wrench_history_force_abs_error_n': 'mean |error| over force components (Fx,Fy,Fz) of every history sample, newtons',
    'wrench_history_torque_abs_error_nm': 'mean |error| over torque components (Mx,My,Mz) of every history sample, newton metres',
    'peak_wrench_force_abs_error_n': 'mean |error| over the 3 force components of |peak_wrench|, newtons',
    'peak_wrench_torque_abs_error_nm': 'mean |error| over the 3 torque components of |peak_wrench|, newton metres',
    'last_wrench_force_abs_error_n': 'last 2 ms force-limit reading, force components, newtons',
    'last_wrench_torque_abs_error_nm': 'last 2 ms force-limit reading, torque components, newton metres',
    'depth_progress_abs_error': 'dimensionless geometric depth progress (fraction of the reset depth span)',
    'depth_progress_abs_error_mm': 'depth progress error times the world depth span, millimetres',
    'timing_error_steps': 'WM first step minus physical first step, control steps (50 ms)',
    'critic_gap': 'V(s_t) minus realised discounted return-to-go (reward units); a gap, not an error of a target',
}


# ----------------------------------------------------------------------------- pure helpers
def first_step(flags, valid):
    """First 1-based step j with flags[j-1] & valid[j-1]; 0 if none. flags/valid: [k, N] bool."""
    hit = flags & valid
    k = hit.shape[0]
    steps = torch.arange(1, k + 1, device=hit.device)[:, None].expand_as(hit)
    first = torch.where(hit, steps, torch.full_like(steps, k + 1)).amin(0)
    return torch.where(first > k, torch.zeros_like(first), first)


def compare_first(real_j, wm_j, select=None):
    """Confusion counts of 'event within window' and timing errors (wm_j - real_j) where both.

    real_j / wm_j: 1-based first step or 0 for none. select: optional [N] bool subset."""
    if select is None:
        select = torch.ones_like(real_j, dtype=torch.bool)
    real, wm = (real_j > 0) & select, (wm_j > 0) & select
    both = real & wm
    errors = (wm_j - real_j)[both].float()
    return {'both': int(both.sum()), 'physical_only': int((real & ~wm).sum()),
            'wm_only': int((wm & ~real).sum()), 'neither': int((select & ~real & ~wm).sum()),
            'timing_error_steps': stats(errors),
            'timing_exact_fraction': float((errors == 0).float().mean()) if errors.numel() else None}


def confusion(real, wm, select=None):
    """2x2 counts for boolean event labels (physical vs WM)."""
    if select is None:
        select = torch.ones_like(real, dtype=torch.bool)
    real, wm = real & select, wm & select
    return {'both': int((real & wm).sum()), 'physical_only': int((real & ~wm).sum()),
            'wm_only': int((wm & ~real).sum()), 'neither': int((select & ~real & ~wm).sum())}


def wrench_errors(pred, true):
    """Mean |error| of force (N) and torque (Nm) components; inputs [..., 6*m] wrench samples."""
    diff = (pred - true).reshape(*pred.shape[:-1], -1, 6).abs()
    return diff[..., :3].mean((-1, -2)), diff[..., 3:].mean((-1, -2))


def stats(x):
    x = x.detach().double().flatten()
    x = x[torch.isfinite(x)]
    if not x.numel():
        return {'count': 0}
    return {'count': int(x.numel()), 'mean': float(x.mean()), 'median': float(x.median()),
            'p95': float(torch.quantile(x, .95)), 'max': float(x.max())}


def event_flags(spec, state):
    """Task events of a (WM or recorded) state from the spec's own rules.

    failure = force | workspace | object_slip | tool_motion is asserted to equal
    spec.hard_events; timeout uses the physical precedence (no success/failure)."""
    failure, timeout = spec.hard_events(state)
    task = spec.free_task
    q = spec.field(state, 'joint_q')
    force = (spec.field(state, 'peak_wrench').abs() > spec.force_limits).any(-1)
    workspace = task.workspace(spec.field(state, 'observation'), spec.fk(q))
    slip = (task.slip_failure(spec.field(state, 'object_slip')[..., 0], force)
            if 'object_slip' in task.config else torch.zeros_like(force))
    tool = (task.tool_motion_failure(spec.field(state, 'completion_xy_error')[..., 0], force)
            if 'completion_position' in task.config else torch.zeros_like(force))
    if not torch.equal(failure, force | workspace | slip | tool):
        raise AssertionError('Event decomposition differs from spec.hard_events')
    success = spec.pose_success(state)
    flags = {'success': success, 'force': force, 'workspace': workspace, 'object_slip': slip,
             'tool_motion': tool, 'timeout': timeout & ~failure & ~success}
    terminal = spec.hard_termination(state)
    if not torch.equal(terminal, failure | timeout | success):
        raise AssertionError('Event decomposition differs from spec.hard_termination')
    return flags, terminal


class ReplayActions(torch.nn.Module):
    """Stands in for the policy inside trainer.imagine: returns recorded actions in order."""
    def __init__(self, actions):
        super().__init__()
        self.actions, self.step = actions, 0

    def forward(self, history, goal):
        action = self.actions[self.step]
        self.step += 1
        return action


def replay_rollout(learner, state, history_states, goal, actions):
    """trainer.imagine on recorded actions [k, N, 6]; returns its rollout (states [k+1, N, D])."""
    policy, horizon = learner.policy, learner.actor_horizon
    learner.policy, learner.actor_horizon = ReplayActions(actions), len(actions)
    try:
        return learner.imagine(state, history_states, goal)
    finally:
        learner.policy, learner.actor_horizon = policy, horizon


def discounted_return_to_go(rewards, valid, gamma):
    """rewards/valid [T, n]: G_t = sum_{i>=t} gamma^(i-t) r_i within the valid prefix."""
    out = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in range(len(rewards) - 1, -1, -1):
        running = torch.where(valid[t], rewards[t] + gamma * running, 0.)
        out[t] = running
    return out


def check_version(metadata):
    from cutting_world_model.pose_depth import REVISION, REVISION_V3
    version = metadata.get('free_cut_task', {}).get('version')
    if version == REVISION:
        raise ValueError('pose_depth_v2 is read-only history and is not diagnosed by this script; '
                         'it supports pose_depth_v3 only')
    if version != REVISION_V3:
        raise ValueError('diagnose_pose_predictions reads pose_depth_v3 only, got ' + str(version))
    if metadata.get('physical_backend') not in BACKENDS:
        raise ValueError('Unsupported physical backend: ' + str(metadata.get('physical_backend')))
    return version


# ----------------------------------------------------------------------------- loading
def load(checkpoint, device):
    from cutting_world_model.policy import SliceItPolicy, SliceItCritic
    from cutting_world_model.trainer import OnlineLearner
    from cutting_world_model.world_model import CuttingWorldModel
    from third_party.supermimic.trainer_config import WMTrainerCfg, PolicyTrainerCfg
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    metadata = saved['metadata']
    check_version(metadata)
    data = saved['learner']
    if data.get('imagination_terminal_rule') != 'task_absorbing_v1':
        raise ValueError('Checkpoint imagination terminal rule differs')
    wm = CuttingWorldModel(metadata).to(device)
    goal_dim = int(metadata.get('goal_dim', 7))
    policy = SliceItPolicy(goal_dim, int(metadata['state_shapes']['observation'][0])).to(device)
    critic = SliceItCritic(wm.spec.state_dim, goal_dim).to(device)
    stage = data['training_stage']
    pcfg = PolicyTrainerCfg(**{**stage['policy_config'], 'betas': tuple(stage['policy_config']['betas'])})
    wcfg = replace(WMTrainerCfg(**stage['world_model_config']), compile=None)
    learner = OnlineLearner(wm, policy, critic, wm_cfg=wcfg, policy_cfg=pcfg,
                            actor_horizon=int(data.get('actor_horizon', pcfg.horizon)))
    wm.load_state_dict(data['world_model'])
    policy.load_state_dict(data['policy'])
    critic.load_state_dict(data['critic'])
    for module in (wm, policy, critic):
        module.eval()
        module.requires_grad_(False)
    return saved, metadata, learner


def build_environment(metadata, source, worlds_per_object, seed):
    from evaluate_pose_policy import evaluation_config
    blademaster = BACKENDS[metadata['physical_backend']] == 'blademaster'
    config = evaluation_config(copy.deepcopy(source), worlds_per_object, seed, blademaster=blademaster)
    if blademaster:
        from cutting_world_model.blademaster_backend import BladeMasterEnvironment
        env = BladeMasterEnvironment(ROOT / 'backends/mujoco', config)
    else:
        from cutting_world_model.multi_env import MultiConditionEnvironment
        env = MultiConditionEnvironment.from_config(ROOT / 'backends/mujoco', config)
    return env, config


# ----------------------------------------------------------------------------- collection
@torch.no_grad()
def collect(env, learner, horizon):
    """One deterministic first episode per world; returns stacked per-step records [T, n, ...]."""
    spec, policy = learner.wm.spec, learner.policy
    device = spec.force_limits.device
    frame = env.current
    state = spec.pack_batch(frame).float()
    history = frame['observation'].float()[None].repeat(policy.history, 1, 1)
    hist_states = state[None].repeat(policy.history, 1, 1)
    n = env.worlds
    alive = torch.ones(n, dtype=torch.bool, device=device)
    rec = {}

    def put(key, value):
        rec.setdefault(key, []).append(value.detach().to(device).clone())
    for _ in range(horizon + 1):
        if not bool(alive.any()):
            break
        goal = env.goal(frame).to(device)
        clean = policy(history, goal)
        action = spec.execute(clean)
        if not bool(torch.isfinite(action).all()):
            raise FloatingPointError('Nonfinite policy action')
        put('state', state)
        put('history_states', hist_states)
        put('action', action)
        put('goal', goal)
        put('alive', alive)
        put('contact_start', frame['knife_board_contact'].reshape(n).bool())
        frame, reward, terminated, truncated, info = env.step(action)
        next_state = spec.pack_batch_transition(frame, state)
        put('next_state', next_state)  # full successor, saved before any reset
        put('reward', reward.float())
        put('terminated', (terminated | truncated).bool())
        put('contact_next', frame['knife_board_contact'].reshape(n).bool())
        put('depth_span', frame['depth_span'].reshape(n).double())
        for event, key in INFO_KEYS.items():
            value = info.get(key)
            put('real_' + event, torch.zeros(n, dtype=torch.bool) if value is None else value.reshape(n).bool())
        for key in ('separation_progress', 'separated', 'material_components', 'damage_saturated_fraction'):
            if key in info:
                put('physical_' + key, info[key].reshape(n).double())
        ended = (terminated | truncated).to(device)
        alive = alive & ~ended
        state = next_state
        history = torch.cat((history[1:], frame['observation'].float()[None]), 0)
        hist_states = torch.cat((hist_states[1:], state[None]), 0)
        if bool(ended.any()):
            frame = env.reset(ended)
            state = spec.reset_batch_history(frame, state, ended)
            history[:, ended] = frame['observation'][ended].float()[None]
            hist_states[:, ended] = state[ended][None]
    if bool(alive.any()):
        raise RuntimeError('Task did not terminate all first episodes within its horizon')
    return {k: torch.stack(v) for k, v in rec.items()}


# ----------------------------------------------------------------------------- analysis
@torch.no_grad()
def analyse(learner, rec, k, chunk=512):
    wm = learner.wm
    spec = wm.spec
    if not getattr(spec, 'board_latch', False):
        raise ValueError('pose_depth_v3 spec required')
    T, n = rec['alive'].shape
    device = rec['alive'].device
    valid = rec['alive']
    length = valid.sum(0)
    flat = valid.flatten()
    t_of = torch.arange(T, device=device)[:, None].expand(T, n).flatten()[flat]
    w_of = torch.arange(n, device=device)[None, :].expand(T, n).flatten()[flat]
    N = len(t_of)
    S, G = rec['state'][t_of, w_of], rec['goal'][t_of, w_of]
    Hs = rec['history_states'][t_of, :, w_of].transpose(0, 1)  # [T, hist, n, D] -> [hist, N, D]
    W = torch.clamp(length[w_of] - t_of, max=k)  # usable window per start
    js = torch.arange(1, k + 1, device=device)
    jvalid = js[:, None] <= W[None, :]  # [k, N]
    tj = (t_of[None, :] + js[:, None] - 1).clamp(max=T - 1)
    actions = torch.where(jvalid[..., None], rec['action'][tj, w_of], 0.)
    real_next = rec['next_state'][tj, w_of]  # [k, N, D]

    # Open-loop replay through trainer.imagine (chunked over starts).
    parts = {key: [] for key in ('states', 'rewards', 'dhat', 'logits')}
    first_step_gap = 0.
    for lo in range(0, N, chunk):
        sl = slice(lo, min(lo + chunk, N))
        ro = replay_rollout(learner, S[sl], Hs[:, sl], G[sl], actions[:, sl])
        parts['states'].append(ro.states)
        parts['rewards'].append(ro.rewards)
        parts['dhat'].append(ro.dhat)
        # One-step logits on the same batch (terminal head read as a predictor only).
        one, _, logits = wm.predict(S[sl], actions[0, sl], G[sl])
        parts['logits'].append(logits)
        # Consistency of the replay's first step with world_model.predict, in the WM's own
        # normalized error units (spec.error_vector); exact up to GPU kernel nondeterminism.
        first_step_gap = max(first_step_gap, float(spec.error_vector(one, ro.states[1]).abs().max()))
    if first_step_gap > 1e-3:
        raise AssertionError('Replay first step differs from world_model.predict: %g' % first_step_gap)
    pred = torch.cat(parts['states'], 1)  # [k+1, N, D]; pred[0] = start
    logits = torch.cat(parts['logits'])

    # ---- per-step errors [k, N]
    P, R = pred[1:], real_next
    fk = spec.fk
    ee_mm = 1000 * (fk(spec.field(P, 'joint_q').double())[..., :3]
                    - fk(spec.field(R, 'joint_q').double())[..., :3]).norm(dim=-1)
    joint = (spec.field(P, 'joint_q') - spec.field(R, 'joint_q')).abs().amax(-1)
    errors = {'ee_position_error_mm': ee_mm, 'joint_abs_error_max_rad': joint}
    for name in ('wrench_history', 'peak_wrench', 'last_wrench'):
        a, b = spec.field(P, name), spec.field(R, name)
        if name == 'peak_wrench':
            a, b = a.abs(), b.abs()
        force, torque = wrench_errors(a, b)
        errors[name + '_force_abs_error_n'] = force
        errors[name + '_torque_abs_error_nm'] = torque
    depth = (spec.field(P, 'cut_progress') - spec.field(R, 'cut_progress'))[..., 0].abs()
    errors['depth_progress_abs_error'] = depth
    errors['depth_progress_abs_error_mm'] = 1000 * depth * rec['depth_span'][t_of, w_of][None]

    # ---- events: one-step and first event within the window
    wm_flags, wm_terminal = event_flags(spec, P)  # states frozen after absorption
    real_flags = {e: rec['real_' + e][tj, w_of] for e in EVENTS}
    real_terminal = rec['terminated'][tj, w_of]
    wm_j = first_step(wm_terminal, jvalid)
    real_j = first_step(real_terminal, jvalid)
    take = lambda flags, j: torch.gather(flags, 0, (j.clamp(min=1) - 1)[None])[0] & (j > 0)
    events = {'one_step': {}, 'k_window': {}}
    for e in EVENTS:
        events['one_step'][e] = confusion(real_flags[e][0], wm_flags[e][0])
        rj = torch.where(take(real_flags[e], real_j), real_j, 0)
        wj = torch.where(take(wm_flags[e], wm_j), wm_j, 0)
        events['k_window'][e] = compare_first(rj, wj)
    events['k_window']['any_termination'] = compare_first(real_j, wm_j)
    # Labels of the recorded successors from the spec rules vs the environment's own flags.
    rec_flags, rec_terminal = event_flags(spec, real_next[0])
    events['analytic_vs_physical_label_agreement'] = {
        **{e: float((rec_flags[e] == real_flags[e][0]).float().mean()) for e in EVENTS},
        'termination': float((rec_terminal == real_terminal[0]).float().mean())}

    # ---- board latch: first contact vs post-latch consistency
    contact_start = rec['contact_start'][t_of, w_of]
    real_contact = rec['contact_next'][tj, w_of]
    wm_contact = spec.contact(P)[..., 0]  # frozen after absorption, like every field
    unlatched = ~contact_start
    latch = {
        'definition': ('physical knife_board_contact (software board-stop latch, recorded frame) vs the '
                       'WM board_contact state field of the predicted state'),
        'first_contact_one_step': confusion(real_contact[0], wm_contact[0], unlatched),
        'first_contact_k_window': compare_first(first_step(real_contact, jvalid),
                                                first_step(wm_contact, jvalid), unlatched),
        'unlatched_starts': int(unlatched.sum()), 'latched_starts': int(contact_start.sum()),
        'post_latch': {
            'physical_joint_motion_max_rad': stats((spec.field(real_next[0], 'joint_q')
                                                    - spec.field(S, 'joint_q')).abs().amax(-1)[contact_start]),
            'wm_joint_motion_max_rad': stats((spec.field(P[0], 'joint_q')
                                              - spec.field(S, 'joint_q')).abs().amax(-1)[contact_start]),
            'wm_latch_kept_fraction': (float(wm_contact[0][contact_start].float().mean())
                                       if bool(contact_start.any()) else None),
        },
    }
    state_flag = spec.contact(rec['next_state'][t_of, w_of])[..., 0]
    latch['state_flag_equals_frame_contact_fraction'] = float(
        (state_flag == rec['contact_next'][t_of, w_of]).float().mean())

    # ---- terminal head as a predictor of the analytic termination label
    label = rec_terminal.float()
    probability = logits.sigmoid()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, label, reduction='none')
    head = {'n': N, 'positives': int(label.sum()), 'bce': stats(bce),
            'accuracy_at_0.5': float(((probability >= .5) == label.bool()).float().mean()),
            'brier': float((probability - label).square().mean()),
            'label': 'spec.hard_termination(recorded successor); training uses the environment '
                     'terminated flag (agreement in events.analytic_vs_physical_label_agreement)',
            'calibration_bins': calibration(probability, label)}

    # ---- critic and imagined actor objective vs realised discounted return-to-go
    from cutting_world_model.trainer import discount_weights
    from third_party.supermimic.objective import policy_objective
    gamma = learner.pcfg.gamma
    rtg = discounted_return_to_go(rec['reward'].double(), valid, gamma)[t_of, w_of]
    value = learner.critic_values(S, G).double()
    def imagined(sl):
        ro = learner.imagine(S[sl], Hs[:, sl], G[sl])
        terminal = learner.critic_values(ro.states[-1], G[sl]) if learner.pcfg.use_critic else None
        J = (discount_weights(ro, gamma) * ro.rewards).sum(0)
        if terminal is not None:
            J = J + gamma ** ro.rewards.shape[0] * ro.survival[-1] * terminal
        if not math.isclose(float(J.mean()), float(policy_objective(ro, terminal, gamma)),
                            rel_tol=1e-4, abs_tol=1e-4):
            raise AssertionError('Per-start objective differs from policy_objective')
        return ro, J.double()
    objective = torch.cat([imagined(slice(lo, min(lo + chunk, N)))[1] for lo in range(0, N, chunk)])

    per_start = dict(t=t_of, world=w_of, window=W, jvalid=jvalid, contact_start=contact_start,
                     real_j=real_j, wm_j=wm_j, wm_contact=wm_contact, real_contact=real_contact,
                     head_probability=probability, head_label=label, critic=value, return_to_go=rtg,
                     imagined_objective=objective, wm_reward=torch.cat(parts['rewards'], 1),
                     wm_dhat=torch.cat(parts['dhat'], 1)[1:], **{'error_' + k_: v for k_, v in errors.items()},
                     **{'wm_' + e: wm_flags[e] for e in EVENTS}, **{'real_' + e: real_flags[e] for e in EVENTS})
    return {'errors': errors, 'events': events, 'latch': latch, 'head': head,
            'per_start': per_start, 'replay_first_step_max_normalized_gap': first_step_gap}


def calibration(probability, label, bins=10):
    edges = torch.linspace(0, 1, bins + 1, device=probability.device)
    index = torch.bucketize(probability, edges[1:-1])
    rows = []
    for b in range(bins):
        mask = index == b
        rows.append({'p_range': [float(edges[b]), float(edges[b + 1])], 'count': int(mask.sum()),
                     'mean_p': float(probability[mask].mean()) if bool(mask.any()) else None,
                     'positive_rate': float(label[mask].mean()) if bool(mask.any()) else None})
    return rows


def error_block(errors, jvalid, window, k, select):
    """One-step (j=1) and k-step (j=k, full windows only) error statistics for a start subset."""
    full = select & (window == k)
    return {'one_step': {name: stats(e[0][select]) for name, e in errors.items()},
            'k_step_at_k': {'full_windows': int(full.sum()),
                            **{name: stats(e[k - 1][full]) for name, e in errors.items()}},
            'mean_by_step_j': {name: [float(e[j][select & jvalid[j]].double().mean())
                                      if bool((select & jvalid[j]).any()) else None
                                      for j in range(k)] for name, e in errors.items()}}


def summarize(result, select, k):
    ps = result['per_start']
    if not bool(select.any()):
        return {'transitions': 0}
    errs = result['errors']
    gap = ps['critic'] - ps['return_to_go']
    return {
        'transitions': int(select.sum()),
        'all_starts': error_block(errs, ps['jvalid'], ps['window'], k, select),
        'pre_latch_starts': error_block(errs, ps['jvalid'], ps['window'], k, select & ~ps['contact_start']),
        'post_latch_starts': error_block(errs, ps['jvalid'], ps['window'], k, select & ps['contact_start']),
        'critic_gap': stats(gap[select]),
        'critic_abs_gap': stats(gap[select].abs()),
        'imagined_objective_gap': stats((ps['imagined_objective'] - ps['return_to_go'])[select]),
        'return_to_go': stats(ps['return_to_go'][select]),
        'terminal_head_bce': stats(torch.nn.functional.binary_cross_entropy(
            ps['head_probability'][select].double().clamp(1e-12, 1 - 1e-12),
            ps['head_label'][select].double(), reduction='none')),
        'first_contact_k_window': compare_first(first_step(ps['real_contact'], ps['jvalid']),
                                                first_step(ps['wm_contact'], ps['jvalid']),
                                                select & ~ps['contact_start']),
        'any_termination_k_window': compare_first(ps['real_j'], ps['wm_j'], select),
    }


def file_hash(path):
    from evaluate_pose_policy import file_hash as digest
    return digest(path)


def finite_json(value):
    """JSON-ready copy: numpy scalars to Python, non-finite floats to None (not measured)."""
    if isinstance(value, dict):
        return {key: finite_json(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path,
                   help='Generated DiSECt conditions or BladeMaster config (backend follows the checkpoint). '
                        'Default: the config embedded in the checkpoint metadata.')
    p.add_argument('--worlds-per-object', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261117)
    p.add_argument('--k', type=int, default=8, help='open-loop length (default 8 = actor horizon)')
    p.add_argument('--records', type=Path,
                   help='Re-analyse records.pt written by an earlier run of this script (no physics)')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.k < 1 or args.worlds_per_object < 1 or args.seed < 0:
        p.error('Positive k and world count, nonnegative seed required')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = 'cuda:0'
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    saved, metadata, learner = load(args.checkpoint, device)
    version = check_version(metadata)
    backend = BACKENDS[metadata['physical_backend']]
    if args.config is not None:
        source = json.loads(args.config.read_text())
    else:
        source = metadata.get('backend_config') if backend == 'blademaster' else metadata.get('task_conditions')
        if source is None:
            raise ValueError('Checkpoint has no embedded config; pass --config')
    if source.get('task') != metadata['free_cut_task']:
        raise ValueError('Config task differs from the checkpoint task')
    before = {name: [t.clone() for t in m.state_dict().values()]
              for name, m in (('wm', learner.wm), ('policy', learner.policy), ('critic', learner.critic))}
    if args.records is not None:
        stored = torch.load(args.records, map_location=device, weights_only=False)
        if stored['checkpoint_sha256'] != file_hash(args.checkpoint):
            raise ValueError('Records were collected with a different checkpoint')
        if (stored['seed'], stored['worlds_per_object']) != (args.seed, args.worlds_per_object):
            raise ValueError('Records differ in seed / world count')
        rec, group_names, conditions = stored['records'], stored['group_names'], stored['conditions']
    else:
        env, config = build_environment(metadata, source, args.worlds_per_object, args.seed)
        try:
            from cutting_world_model.control_state import ControlStateSpec
            from cutting_world_model.shared_baselines import validate_current_interface
            from evaluate_pose_policy import validate_interfaces
            frame = env.current
            actual = ControlStateSpec.metadata_from_frame(env.metadata, {key: v[0].cpu().numpy() for key, v in frame.items()})
            validate_interfaces(metadata, actual, config)
            validate_current_interface(metadata, actual)
            group_names = [g['name'] for g in config['groups'] for _ in range(g['worlds'])]
            conditions = env.conditions(range(env.worlds))
            rec = collect(env, learner, int(config['task']['horizon_steps']))
        finally:
            env.close()
        # Raw physical records first, so the analysis can be repeated without physics.
        torch.save({'records': rec, 'group_names': group_names, 'conditions': conditions,
                    'checkpoint_sha256': file_hash(args.checkpoint), 'seed': args.seed,
                    'worlds_per_object': args.worlds_per_object, 'config': config},
                   args.output / 'records.pt')
    rollout_seconds = time.monotonic() - started
    result = analyse(learner, rec, args.k)
    for name, m in (('wm', learner.wm), ('policy', learner.policy), ('critic', learner.critic)):
        if any(not torch.equal(a, b) for a, b in zip(before[name], m.state_dict().values())):
            raise RuntimeError('Frozen ' + name + ' changed during the diagnostic')

    ps = result['per_start']
    valid = rec['alive']
    n = valid.shape[1]
    length = valid.sum(0)
    end = length - 1
    idx = torch.arange(n, device=valid.device)
    group_of = np.array(group_names)
    worlds = []
    for w in range(n):
        L, e = int(length[w]), int(end[w])
        row = {'world': w, 'object': group_names[w], 'steps': L,
               **{event: bool(rec['real_' + event][e, w]) for event in EVENTS},
               'return': float(ps['return_to_go'][(ps['world'] == w) & (ps['t'] == 0)][0]),
               'final_depth_progress': float(learner.wm.spec.field(rec['next_state'][e, w], 'cut_progress')[0]),
               'first_physical_latch_step': (int(rec['contact_next'][:L, w].float().argmax()) + 1
                                             if bool(rec['contact_next'][:L, w].any()) else None),
               'condition': conditions[w]}
        separation = {}
        for key in ('separation_progress', 'separated'):
            if 'physical_' + key in rec:
                separation['final_' + key] = float(rec['physical_' + key][e, w])
        for key in ('material_components', 'damage_saturated_fraction'):
            if 'physical_' + key in rec:
                separation['terminal_' + key] = float(rec['physical_' + key][e, w])
        row['physical_separation'] = separation or 'not logged'
        worlds.append(row)
    episodes = lambda sel: {'episodes': len(sel), **{event: sum(x[event] for x in sel) for event in EVENTS}}
    objects = sorted(set(group_names))
    summary = {
        'diagnostic': 'pose-task same-trajectory WM prediction check (zero optimizer updates)',
        'checkpoint': str(args.checkpoint), 'task_version': version, 'backend': backend,
        'physical_backend': metadata['physical_backend'], 'k': args.k,
        'checkpoint_actor_horizon': int(saved['learner'].get('actor_horizon', learner.pcfg.horizon)),
        'gamma': learner.pcfg.gamma, 'seed': args.seed, 'worlds_per_object': args.worlds_per_object,
        'worlds': n, 'policy_updates': saved['learner'].get('updates'),
        'wm_updates': saved['learner'].get('training_stage', {}).get('wm_updates'),
        'command': sys.argv, 'rollout_seconds': rollout_seconds,
        'total_seconds': time.monotonic() - started, 'records_reused': args.records is not None,
        'replay_first_step_max_normalized_gap': result['replay_first_step_max_normalized_gap'],
        'units': UNITS,
        'semantics': {
            'one_step': 'j=1 of the open-loop replay from the recorded start state with the recorded action',
            'k_step': ('open-loop on recorded executed actions through trainer.OnlineLearner.imagine; after '
                       'spec.hard_termination the WM state is frozen and later steps add nothing; a window '
                       'ends at k or at the physical episode end (no recorded actions beyond it); '
                       'k_step_at_k uses full windows only'),
            'pre_latch_starts': 'recorded start state not physically latched at the board stop',
            'post_latch_starts': 'recorded start state physically latched (knife_board_contact)',
            'events': ('physical flags from the environment info; WM flags from spec.hard_events/'
                       'pose_success/hard_termination of the predicted state; timeout = deadline without '
                       'success/failure (physical precedence)'),
            'terminal_head': ('evaluated only as a predictor of the analytic termination label; v3 '
                              'imagined survival is analytic (prod(1-absorbed)) and the head does not '
                              'enter the policy objective, GAE targets or MPPI cost'),
            'critic': 'V(s_t) - discounted realised return-to-go of the first episode (gamma from task)',
            'imagined_objective': ('per-start trainer.imagine objective with the policy and the terminal '
                                   'critic (trainer.discount_weights) minus realised return-to-go'),
            'physical_separation': ('DiSECt: separation_progress/separated from info; BladeMaster: native '
                                    'component count / saturated damage fraction at termination; diagnostic only, '
                                    'never a task label'),
        },
        'episodes': {'overall': episodes(worlds),
                     'per_object': {o: episodes([x for x in worlds if x['object'] == o]) for o in objects}},
        'events': result['events'],
        'board_latch': result['latch'],
        'terminal_head': result['head'],
        'overall': summarize(result, torch.ones_like(ps['t'], dtype=torch.bool), args.k),
        'per_object': {o: summarize(result, torch.as_tensor(group_of[ps['world'].cpu().numpy()] == o,
                                                            device=ps['t'].device), args.k) for o in objects},
        'per_world': worlds,
    }
    (args.output / 'summary.json').write_text(json.dumps(finite_json(summary), indent=1, allow_nan=False) + '\n')
    cpu = lambda x: x.detach().cpu().numpy()
    np.savez_compressed(args.output / 'transitions.npz',
                        **{key: cpu(rec[key][ps['t'], ps['world']]) for key in rec if key != 'history_states'},
                        t=cpu(ps['t']), world=cpu(ps['world']), object=group_of[cpu(ps['world'])])
    np.savez_compressed(args.output / 'steps.npz',
                        **{key: cpu(value) for key, value in ps.items()},
                        object=group_of[cpu(ps['world'])])
    brief = {key: summary[key] for key in ('task_version', 'backend', 'worlds', 'episodes')}
    brief['overall_one_step'] = {key: summary['overall']['all_starts']['one_step'][key].get('mean')
                                 for key in summary['overall']['all_starts']['one_step']}
    print(json.dumps(brief, indent=1))


if __name__ == '__main__':
    main()
