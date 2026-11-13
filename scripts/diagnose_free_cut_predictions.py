#!/usr/bin/env python3
"""Same-trajectory prediction check for free_cut checkpoints (diagnostic; no training).

Runs the checkpoint's deterministic policy for one episode per world on a multi-condition
config and records, per control step: the packed WM state, the executed action, the goal,
the observation history, the physical reward, cut progress, separation/force/workspace events,
the end-effector position, the sawing output a4 and the integrated lateral target offset.

Then, on the recorded trajectories and with the checkpoint's own networks:
- 1-step WM prediction from the recorded state with the recorded action: progress, peak wrench,
  end-effector position, completion probability split into predicted release fraction and
  head probability;
- k-step open-loop prediction with the recorded actions: cumulative completion probability
  (no completion after a predicted hard ending, as in training) and predicted workspace exit,
  against what physically happened within k steps;
- critic value V(s_t) against the realised undiscounted return-to-go;
- the imagined actor objective from s_t (policy actions inside the WM for the actor horizon,
  survival-weighted rewards plus the terminal critic value, as in training) against the
  realised return-to-go.
Exits/endings are attributed per workspace axis. Physical worlds are reset after their first
episode only because the batched environment requires it; later data are not used.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model.multi_env import MultiConditionEnvironment  # noqa: E402
from cutting_world_model.policy import SliceItPolicy, SliceItCritic  # noqa: E402
from cutting_world_model.provenance import git_revision  # noqa: E402
from cutting_world_model.task_heads import pose_error  # noqa: E402
from cutting_world_model.world_model import CuttingWorldModel  # noqa: E402


def load(checkpoint, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    metadata = saved['metadata']
    if not metadata.get('free_cut_task'):
        raise ValueError('Not a free_cut checkpoint')
    wm = CuttingWorldModel(metadata).to(device)
    wm.load_state_dict(saved['learner']['world_model'])
    goal_dim = int(metadata.get('goal_dim', 7))
    policy = SliceItPolicy(goal_dim, int(metadata['state_shapes']['observation'][0])).to(device)
    policy.load_state_dict(saved['learner']['policy'])
    critic = SliceItCritic(wm.spec.state_dim, goal_dim).to(device)
    critic.load_state_dict(saved['learner']['critic'])
    for module in (wm, policy, critic):
        module.eval()
    return saved, metadata, wm, policy, critic


def value(spec, critic, states, goal):
    error = pose_error(goal[..., :7], spec.field(states, 'previous_ee_pose'))
    return critic(spec.canonical(states), goal, error)


def one_step(wm, state, action, goal):
    """Prediction with the completion probability split into its two factors."""
    spec = wm.spec
    next_state, step_reward, fraction = wm._transition(state, action, goal)
    logits = wm.terminal_logits(next_state, goal, state, fraction)
    feats = spec.terminal_features(next_state, goal)
    head = torch.sigmoid(wm.term_head(torch.cat((feats, feats.abs()), -1)).squeeze(-1))
    reward = step_reward + spec.completion_reward(next_state, logits)
    return next_state, reward, torch.sigmoid(logits), fraction, head


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--conditions', type=Path, required=True)
    p.add_argument('--k', type=int, default=8, help='open-loop prediction length (actor horizon)')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = 'cuda:0'
    torch.set_num_threads(2)
    saved, metadata, wm, policy, critic = load(args.checkpoint, device)
    spec, task = wm.spec, wm.spec.free_task
    config = json.loads(args.conditions.read_text())
    if config.get('task') != metadata['free_cut_task']:
        raise ValueError('Evaluation task must match the checkpoint task')
    config = dict(config, parallel=False)  # direct access to the action mapper
    env = MultiConditionEnvironment.from_config(ROOT / 'backends/mujoco', config, reward=None)
    if len(env.members) != 1:
        raise ValueError('This diagnostic reads one object group')
    physics = env.members[0].env.physics
    n = env.worlds
    horizon = task.horizon
    box = task.config['workspace']
    lower = torch.tensor(box['lower'], device=device, dtype=torch.float64)
    upper = torch.tensor(box['upper'], device=device, dtype=torch.float64)
    conditions = env.conditions(range(n))

    frame = env.current
    state = spec.pack_batch(frame).float()
    history = frame['observation'].float()[None].repeat(policy.history, 1, 1)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    rec = {k: [] for k in ('state', 'action', 'goal', 'history', 'reward', 'rho', 'separated', 'collision',
                           'outside', 'timeout', 'ee', 'peak', 'a4', 'offset_x', 'alive', 'speed')}
    with torch.no_grad():
        for t in range(horizon + 1):
            if not bool(alive.any()):
                break
            goal = MultiConditionEnvironment.goal(frame)
            clean = policy(history, goal)
            action = spec.execute(clean)
            rec['state'].append(state.clone())
            rec['action'].append(action.clone())
            rec['goal'].append(goal.clone())
            rec['history'].append(history.clone())
            rec['a4'].append(clean[:, 4].clone())
            rec['alive'].append(alive.clone())
            frame, reward, terminated, truncated, info = env.step(action)
            next_state = spec.pack_batch_transition(frame, state)
            rec['reward'].append(reward.float())
            rec['rho'].append(frame['cut_progress'][:, 0].float())
            for key in ('separated', 'collision', 'outside', 'timeout'):
                rec[key].append(info[key].to(device).bool())
            rec['ee'].append(spec.fk(frame['joint_q'].double())[:, :3])
            rec['peak'].append(frame['peak_wrench'].abs().float())
            rec['offset_x'].append(physics.mapper.added_motion_command[:, 0].clone())
            rec['speed'].append(task.heads.cutting_speed(frame['observation'].float()) - goal[:, 7])
            ended = terminated | truncated
            alive &= ~ended
            state = next_state
            history = torch.cat((history[1:], frame['observation'].float()[None]), 0)
            if bool(ended.any()):
                frame = env.reset(ended)
                state = spec.reset_batch_history(frame, state, ended)
                history[:, ended] = frame['observation'][ended].float()[None]
    env.close()
    rollout_seconds = time.monotonic() - started
    T = len(rec['state'])
    stack = {k: torch.stack(v) for k, v in rec.items()}  # [T, n, ...]
    valid = stack['alive']  # step t belongs to the first episode
    length = valid.sum(0)  # steps per world
    # Realised undiscounted return-to-go from each step (gamma = 1).
    rewards = torch.where(valid, stack['reward'], 0.)
    rtg = rewards.flip(0).cumsum(0).flip(0)
    end_t = length - 1
    idx = torch.arange(n, device=device)
    ending = {}
    for key in ('separated', 'collision', 'outside', 'timeout'):
        ending[key] = stack[key][end_t, idx]
    success = ending['separated'] & ~ending['collision'] & ~ending['outside']
    exit_axis = ((stack['ee'][end_t, idx] < lower) | (stack['ee'][end_t, idx] > upper))

    flat = valid.flatten()
    S = stack['state'].flatten(0, 1)[flat]
    A = stack['action'].flatten(0, 1)[flat]
    G = stack['goal'].flatten(0, 1)[flat]
    H = stack['history'].permute(0, 2, 1, 3).flatten(0, 1)[flat].permute(1, 0, 2)  # [hist, N, obs]
    t_of = torch.arange(T, device=device)[:, None].expand(T, n).flatten()[flat]
    w_of = idx[None, :].expand(T, n).flatten()[flat]
    N = len(S)

    with torch.no_grad():
        # ---- one step with the recorded action
        nxt, _, prob, fraction, head = one_step(wm, S, A, G)
        rho_pred = spec.field(nxt, 'cut_progress')[..., 0]
        rho_true = stack['rho'][t_of, w_of]
        sep_next = stack['separated'][t_of, w_of]
        peak_pred = spec.field(nxt, 'peak_wrench').abs()
        peak_true = stack['peak'][t_of, w_of]
        ee_pred = spec.fk(spec.field(nxt, 'joint_q').double())[:, :3]
        ee_true = stack['ee'][t_of, w_of]
        rho_now = spec.field(S, 'cut_progress')[..., 0]

        # ---- k-step open loop with the recorded actions
        k = args.k
        cum_survive = torch.ones(N, device=device)
        completion_mass = torch.zeros(N, device=device)
        hard_absorbed = torch.zeros(N, dtype=torch.bool, device=device)
        pred_exit = torch.zeros(N, dtype=torch.bool, device=device)
        pred_force = torch.zeros(N, dtype=torch.bool, device=device)
        state_k = S
        steps_used = torch.zeros(N, dtype=torch.long, device=device)
        for j in range(k):
            tj = t_of + j
            usable = (tj < length[w_of])
            act = stack['action'][tj.clamp(max=T - 1), w_of]
            nk, _, pk, _, _ = one_step(wm, state_k, act, G)
            pk = torch.where(usable, pk, 0.)
            failure, timeout_k = spec.hard_events(nk)
            # Completion is only reachable before a predicted hard ending (training rule).
            completion_mass = completion_mass + torch.where(hard_absorbed, 0., cum_survive * pk)
            cum_survive = torch.where(hard_absorbed, cum_survive, cum_survive * (1 - pk))
            hard_absorbed |= usable & (failure | timeout_k)
            out_axis = task.workspace_features(spec.field(nk, 'observation'), spec.fk(spec.field(nk, 'joint_q')))
            pred_exit |= usable & (out_axis > 0).any(-1)
            pred_force |= usable & (spec.field(nk, 'peak_wrench').abs() > spec.force_limits).any(-1)
            state_k = torch.where(usable[:, None], nk, state_k)
            steps_used += usable.long()
        pred_cum_completion = completion_mass  # with hard-event absorption
        window_end = (t_of + steps_used - 1)
        actual_sep_k = ending['separated'][w_of] & (end_t[w_of] <= window_end)
        actual_exit_k = ending['outside'][w_of] & (end_t[w_of] <= window_end)

        # ---- critic and imagined actor objective from the recorded state
        v_now = value(spec, critic, S, G)
        g_true = rtg[t_of, w_of]
        survival = torch.ones(N, device=device)
        absorbed = torch.zeros(N, dtype=torch.bool, device=device)
        objective = torch.zeros(N, device=device)
        state_i, hist_i = S, H
        a4_imag = []
        for _ in range(wm_horizon(saved)):
            act = spec.execute(policy(hist_i, G))
            a4_imag.append(act[:, 4])
            ni, ri, pi, _, _ = one_step(wm, state_i, act, G)
            done = spec.hard_termination(ni)
            ri = torch.where(absorbed, 0., ri)
            objective = objective + survival * ri
            ni = torch.where(absorbed[:, None], state_i, ni)
            absorbed = absorbed | done
            pi = torch.where(absorbed, 1., pi)
            survival = survival * (1 - pi)
            state_i = ni
            hist_i = torch.cat((hist_i[1:], spec.field(ni, 'observation')[None]), 0)
        objective = objective + survival * value(spec, critic, state_i, G)

    def m(x):
        return float(x.float().mean()) if x.numel() else None

    def summary(mask):
        out = {'transitions': int(mask.sum())}
        if not bool(mask.any()):
            return out
        out.update({
            'rho_abs_error_1step': m((rho_pred - rho_true).abs()[mask]),
            'peak_force_component_abs_error_1step_n': m((peak_pred - peak_true)[mask][:, :3].abs()),
            'peak_torque_component_abs_error_1step_nm': m((peak_pred - peak_true)[mask][:, 3:].abs()),
            'ee_position_error_1step_mm': m(1000 * (ee_pred - ee_true).norm(dim=-1)[mask]),
            'critic_minus_return_to_go': m((v_now - g_true)[mask]),
            'critic_abs_error': m((v_now - g_true).abs()[mask]),
            'imagined_objective_minus_return_to_go': m((objective - g_true)[mask]),
            'imagined_objective_mean': m(objective[mask]),
            'return_to_go_mean': m(g_true[mask]),
        })
        return out

    sep_mask = sep_next
    no_sep = ~sep_next
    completion = {
        'separation_transitions': int(sep_mask.sum()),
        'p_complete_on_separation_mean': m(prob[sep_mask]),
        'release_fraction_on_separation_mean': m(fraction[sep_mask]),
        'head_on_separation_mean': m(head[sep_mask]),
        'true_remaining_fraction_on_separation_mean': m(((rho_true - rho_now) / (1 - rho_now).clamp_min(1e-9))[sep_mask]),
        'p_complete_on_other_mean': m(prob[no_sep]),
        'p_complete_on_other_ge_0.5': m((prob[no_sep] >= .5)),
        'release_fraction_on_other_mean': m(fraction[no_sep]),
        'head_on_other_mean': m(head[no_sep]),
        'brier_1step': m((prob - sep_mask.float()).square()),
        f'cum_completion_{k}step_when_separated_within_{k}': m(pred_cum_completion[actual_sep_k]),
        f'cum_completion_{k}step_when_not_separated_within_{k}': m(pred_cum_completion[~actual_sep_k]),
    }
    exits = {
        f'actual_exit_within_{k}': int(actual_exit_k.sum()),
        f'predicted_exit_when_actual_exit_within_{k}': m(pred_exit[actual_exit_k].float()),
        f'predicted_exit_when_no_actual_exit_within_{k}': m(pred_exit[~actual_exit_k].float()),
        f'predicted_force_event_within_{k}': m(pred_force.float()),
    }
    worlds = []
    for w in range(n):
        L = int(length[w])
        a4 = stack['a4'][:L, w]
        worlds.append({
            'world': w, **{key: conditions[w][key] for key in ('youngs_modulus', 'cut_spring_stiffness', 'density',
                                                                  'target_speed_mps', 'start_lift_m')},
            'steps': L, 'success': bool(success[w]),
            **{key: bool(ending[key][w]) for key in ending},
            'exit_axes_xyz': [bool(x) for x in exit_axis[w]],
            'final_progress': float(stack['rho'][L - 1, w]),
            'a4_mean': float(a4.mean()), 'a4_abs_mean': float(a4.abs().mean()),
            'a4_same_sign_fraction': float(max((a4 > 0).float().mean(), (a4 < 0).float().mean())),
            'final_offset_x_mm': float(1000 * stack['offset_x'][L - 1, w]),
            'speed_abs_error_mps': float(stack['speed'][:L, w].abs().mean()),
            'return': float(rtg[0, w]),
        })
    exit_worlds = [x for x in worlds if x['outside']]
    near_exit = valid.clone() & False
    for w in range(n):
        if bool(ending['outside'][w]):
            L = int(length[w])
            near_exit[max(0, L - k):L, w] = True
    near_exit_flat = near_exit.flatten()[flat]
    report = {
        'git': git_revision(ROOT), 'command': sys.argv, 'checkpoint': str(args.checkpoint),
        'checkpoint_iteration': saved.get('iteration'), 'policy_updates': saved['learner'].get('updates'),
        'conditions': str(args.conditions), 'worlds': n, 'k': k, 'rollout_seconds': rollout_seconds,
        'total_seconds': time.monotonic() - started,
        'speed_abs_error_mps_median': float(np.median([x['speed_abs_error_mps'] for x in worlds])),
        'episodes': {'success': int(success.sum()), 'separated': int(ending['separated'].sum()),
                     'force_limit': int(ending['collision'].sum()), 'workspace': int(ending['outside'].sum()),
                     'timeout': int(ending['timeout'].sum()), 'mean_return': float(rtg[0].mean())},
        'all_transitions': summary(torch.ones(N, dtype=torch.bool, device=device)),
        'transitions_of_successful_episodes': summary(success[w_of]),
        'transitions_of_workspace_episodes': summary(ending['outside'][w_of]),
        f'last_{k}_transitions_before_workspace_exit': summary(near_exit_flat),
        'completion': completion, 'workspace': exits,
        'workspace_episodes': {
            'count': len(exit_worlds),
            'exit_axis_counts_xyz': [sum(x['exit_axes_xyz'][i] for x in exit_worlds) for i in range(3)],
            'steps': sorted({x['steps'] for x in exit_worlds}),
            'a4_same_sign_fraction_mean': (float(np.mean([x['a4_same_sign_fraction'] for x in exit_worlds]))
                                           if exit_worlds else None),
            'a4_abs_mean': float(np.mean([x['a4_abs_mean'] for x in exit_worlds])) if exit_worlds else None,
            'final_offset_x_mm': [round(x['final_offset_x_mm'], 1) for x in exit_worlds][:20]},
        'imagined_a4_abs_mean': float(torch.stack(a4_imag).abs().mean()),
        'per_world': worlds,
    }
    (args.output / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    np.savez_compressed(args.output / 'transitions.npz', state=S.cpu().numpy(), action=A.cpu().numpy(),
                        goal=G.cpu().numpy(), history=H.permute(1, 0, 2).cpu().numpy(),
                        t=t_of.cpu().numpy(), world=w_of.cpu().numpy(), reward=stack['reward'][t_of, w_of].cpu().numpy(),
                        rho_next=rho_true.cpu().numpy(), separated_next=sep_next.cpu().numpy(),
                        collision_next=stack['collision'][t_of, w_of].cpu().numpy(),
                        outside_next=stack['outside'][t_of, w_of].cpu().numpy(),
                        timeout_next=stack['timeout'][t_of, w_of].cpu().numpy())
    np.savez_compressed(args.output / 'steps.npz', t=t_of.cpu().numpy(), world=w_of.cpu().numpy(),
                        rho_true=rho_true.cpu().numpy(), rho_pred=rho_pred.cpu().numpy(),
                        p_complete=prob.cpu().numpy(), release_fraction=fraction.cpu().numpy(),
                        head=head.cpu().numpy(), separated_next=sep_next.cpu().numpy(),
                        cum_completion_k=pred_cum_completion.cpu().numpy(), predicted_exit_k=pred_exit.cpu().numpy(),
                        actual_exit_k=actual_exit_k.cpu().numpy(), critic=v_now.cpu().numpy(),
                        imagined_objective=objective.cpu().numpy(), return_to_go=g_true.cpu().numpy(),
                        peak_true=peak_true.cpu().numpy(), peak_pred=peak_pred.cpu().numpy(),
                        ee_true=ee_true.cpu().numpy(), ee_pred=ee_pred.cpu().numpy())
    print(json.dumps({k: v for k, v in report.items() if k not in ('per_world', 'command', 'git')}, indent=1))


def wm_horizon(saved):
    return int(saved['learner'].get('actor_horizon') or 8)


if __name__ == '__main__':
    main()
