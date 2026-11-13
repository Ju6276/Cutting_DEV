#!/usr/bin/env python3
"""Same-condition comparison of a free_cut policy with fixed reference cuts, plus reward decomposition.

References run on the same multi-condition config (same seed, worlds and first-episode conditions as
the policy diagnostic): knife pressed straight down at each world's commanded speed until the
travel of the scene reference plus that world's start lift is covered (then held), optionally with
a lateral sinusoid along the blade. Same task settlement, deadline, limits and reward as training;
the policy's action mapper is bypassed (reference branch), so these are control references, not
policies. Only the first episode of every world is used.

Analysis for each trajectory set (policy from `diagnose_free_cut_predictions.py` transitions,
press, press + saw): per-case ending, return, speed error, peak limit utilisation and final
progress; the return decomposed into progress, speed cost (progress-weighted and time parts),
load, jerk and terminal settlement (checked against the recorded reward); and the checkpoint
critic's value along each trajectory against the realised return-to-go.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_gpu_backend.controller.fdcc import quat_apply  # noqa: E402
from cutting_world_model.multi_env import MultiConditionEnvironment  # noqa: E402
from cutting_world_model.policy import SliceItCritic  # noqa: E402
from cutting_world_model.provenance import git_revision  # noqa: E402
from cutting_world_model.task_heads import pose_error  # noqa: E402
from cutting_world_model.world_model import CuttingWorldModel  # noqa: E402


def run_reference(config, spec, amplitude, frequency, device):
    env = MultiConditionEnvironment.from_config(ROOT / 'backends/mujoco', dict(config, parallel=False), reward=None)
    if len(env.members) != 1:
        raise ValueError('One object group expected')
    physics = env.members[0].env.physics
    n = env.worlds
    conditions = env.conditions(range(n))
    material = physics.material
    velocity = material.reference_linear_velocity.reshape(-1, 3)[0].to(physics.dtype)
    down = velocity / velocity.norm()
    scene_travel = float(velocity.norm()) * float(material.reference_duration)
    speed = torch.tensor([c['target_speed_mps'] for c in conditions], device=device, dtype=physics.dtype)
    travel = scene_travel + physics.initial_lift.to(physics.dtype)
    duration = travel / speed
    lateral = torch.tensor([0., 0., 1.], device=device, dtype=physics.dtype)
    rotation = physics.coupling.base_from_disect[3:].to(physics.dtype)

    def reference_target(sim_time):
        t = torch.minimum(sim_time.to(physics.dtype), duration)
        disp = down[None] * (speed * t)[:, None]
        if amplitude > 0:
            disp = disp + lateral[None] * (amplitude * torch.sin(2 * math.pi * frequency * t))[:, None]
        target = physics.reference_start.clone()
        target[:, :3] += quat_apply(rotation.expand(n, 4), disp)
        return target

    physics.reference_target = reference_target
    frame = env.current
    state = spec.pack_batch(frame).float()
    alive = torch.ones(n, dtype=torch.bool, device=device)
    rows = {k: [] for k in ('state', 'next_state', 'goal', 'reward', 'separated', 'collision', 'outside',
                            'timeout', 'alive')}
    zero = torch.zeros(n, 6, device=device)
    with torch.no_grad():
        for _ in range(spec.free_task.horizon + 1):
            if not bool(alive.any()):
                break
            goal = MultiConditionEnvironment.goal(frame)
            rows['state'].append(state.clone())
            rows['goal'].append(goal.clone())
            rows['alive'].append(alive.clone())
            frame, reward, terminated, truncated, info = env.step(zero, reference=True)
            next_state = spec.pack_batch_transition(frame, state)
            rows['next_state'].append(next_state.clone())
            rows['reward'].append(reward.float())
            for key in ('separated', 'collision', 'outside', 'timeout'):
                rows[key].append(info[key].to(device).bool())
            ended = terminated | truncated
            alive &= ~ended
            state = next_state
            if bool(ended.any()):
                frame = env.reset(ended)
                state = spec.reset_batch_history(frame, state, ended)
    env.close()
    stack = {k: torch.stack(v) for k, v in rows.items()}
    valid = stack['alive']
    flat = valid.flatten()
    T = valid.shape[0]
    w_of = torch.arange(n, device=device)[None].expand(T, n).flatten()[flat]
    t_of = torch.arange(T, device=device)[:, None].expand(T, n).flatten()[flat]
    data = {k: stack[k].flatten(0, 1)[flat] for k in ('state', 'next_state', 'goal', 'reward', 'separated',
                                                       'collision', 'outside', 'timeout')}
    return data, w_of, t_of, conditions


def decompose(spec, data, w_of, t_of, n):
    """Per-case sums of the v3 reward terms from the recorded transitions."""
    task = spec.free_task
    w = task.weights
    nxt, old, goal = data['next_state'], data['state'], data['goal']
    obs = spec.field(nxt, 'observation')
    peak = spec.field(nxt, 'peak_wrench')
    rho_next = spec.field(nxt, 'cut_progress')[..., 0]
    rho_old = spec.field(old, 'cut_progress')[..., 0]
    v_star = goal[..., 7]
    v = task.heads.cutting_speed(obs)
    e = ((v - v_star) / (v.abs() + v_star)).square()
    inc = rho_next - rho_old
    util = (peak.abs() / peak.new_tensor(task.heads.force_limits)).amax(-1)
    terms = {
        'progress': w['progress'] * inc,
        'speed_cost_progress_part': -w['speed'] * e * inc.clamp_min(0),
        'speed_cost_time_part': -w['speed'] * e / task.horizon,
        'load': -task.dt * w['load'] * util.clamp(max=1),
        'jerk': -task.dt * w['jerk'] * task.heads.jerkiness(obs).clamp(max=1),
    }
    success = data['separated'] & ~data['collision'] & ~data['outside']
    failure = data['collision'] | data['outside']
    terms['terminal'] = w['success'] * success.float() - w['failure'] * failure.float()
    total = sum(terms.values())
    check = float((total - data['reward']).abs().max())
    per_case = {k: torch.zeros(n, device=w_of.device).index_add_(0, w_of, v.float()) for k, v in terms.items()}
    speed_err = torch.zeros(n, device=w_of.device).index_add_(0, w_of, (v - v_star).abs().float())
    steps = torch.zeros(n, device=w_of.device).index_add_(0, w_of, torch.ones_like(v).float())
    max_util = torch.zeros(n, device=w_of.device).scatter_reduce_(0, w_of, util.float(), 'amax')
    last = torch.zeros(n, dtype=torch.long, device=w_of.device).scatter_reduce_(0, w_of, torch.arange(len(w_of), device=w_of.device), 'amax')
    ending = []
    for i in range(n):
        j = int(last[i])
        ending.append('success' if bool(success[j]) else 'force_limit' if bool(data['collision'][j])
                      else 'workspace' if bool(data['outside'][j]) else 'timeout' if bool(data['timeout'][j])
                      else 'separated_with_failure' if bool(data['separated'][j]) else 'unknown')
    return dict(terms=per_case, speed_err=speed_err / steps.clamp(min=1), steps=steps, max_util=max_util,
                final_rho=rho_next[last], ending=ending, reward_check=check)


def critic_bias(spec, critic, data, w_of, t_of, n):
    with torch.no_grad():
        states, goal = data['state'], data['goal']
        error = pose_error(goal[..., :7], spec.field(states, 'previous_ee_pose'))
        value = critic(spec.canonical(states), goal, error)
    # return-to-go per transition (undiscounted, within each case)
    order = torch.argsort(w_of * 100000 + t_of)
    rtg = torch.empty_like(data['reward'])
    r = data['reward'][order]
    w_sorted = w_of[order]
    acc = 0.
    for k in range(len(order) - 1, -1, -1):
        if k == len(order) - 1 or int(w_sorted[k]) != int(w_sorted[k + 1]):
            acc = 0.
        acc += float(r[k])
        rtg[order[k]] = acc
    diff = value - rtg
    first = t_of == 0
    return {'mean_bias': float(diff.mean()), 'mean_abs_error': float(diff.abs().mean()),
            'start_value_mean': float(value[first].mean()), 'start_return_mean': float(rtg[first].mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True, help='policy checkpoint (critic and task)')
    p.add_argument('--policy-diagnostic', type=Path, required=True, help='diagnose_free_cut_predictions output')
    p.add_argument('--conditions', type=Path, required=True)
    p.add_argument('--saw', default='0.02:1', help='amplitude(m):frequency(Hz) of the sawing reference')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = 'cuda:0'
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    metadata = saved['metadata']
    wm = CuttingWorldModel(metadata).to(device)
    spec = wm.spec
    critic = SliceItCritic(spec.state_dim, int(metadata.get('goal_dim', 7))).to(device)
    critic.load_state_dict(saved['learner']['critic'])
    critic.eval()
    config = json.loads(args.conditions.read_text())
    if config.get('task') != metadata['free_cut_task']:
        raise ValueError('Config task must match the checkpoint task')
    report = json.loads((args.policy_diagnostic / 'report.json').read_text())
    pol = np.load(args.policy_diagnostic / 'transitions.npz')
    n = report['worlds']
    sets = {}
    p_data = {k: torch.as_tensor(pol[k], device=device) for k in ('state', 'next_state', 'goal', 'reward')}
    for key, src in (('separated', 'separated_next'), ('collision', 'collision_next'),
                     ('outside', 'outside_next'), ('timeout', 'timeout_next')):
        p_data[key] = torch.as_tensor(pol[src], device=device).bool()
    sets['policy'] = (p_data, torch.as_tensor(pol['world'], device=device).long(),
                      torch.as_tensor(pol['t'], device=device).long())
    amp, freq = (float(x) for x in args.saw.split(':'))
    conditions = None
    for name, a, f in (('press', 0., 0.), ('press_saw', amp, freq)):
        data, w_of, t_of, conditions = run_reference(config, spec, a, f, device)
        sets[name] = (data, w_of, t_of)
    policy_conditions = [{k: x[k] for k in ('youngs_modulus', 'cut_spring_stiffness', 'density', 'target_speed_mps',
                                            'start_lift_m')} for x in report['per_world']]
    mismatch = sum(any(abs(policy_conditions[i][k] - conditions[i][k]) > 1e-6 for k in policy_conditions[i])
                   for i in range(n))
    out = {'git': git_revision(ROOT), 'command': sys.argv, 'checkpoint': str(args.checkpoint),
           'policy_updates': saved['learner'].get('updates'), 'condition_mismatches': mismatch, 'sets': {},
           'cases': []}
    results = {}
    for name, (data, w_of, t_of) in sets.items():
        d = decompose(spec, data, w_of, t_of, n)
        results[name] = d
        endings = {e: d['ending'].count(e) for e in sorted(set(d['ending']))}
        out['sets'][name] = {
            'endings': endings, 'mean_return': float(sum(d['terms'].values()).mean()),
            'mean_terms': {k: float(v.mean()) for k, v in d['terms'].items()},
            'median_speed_error_mps': float(d['speed_err'].median()),
            'median_steps': float(d['steps'].median()), 'max_utilisation_max': float(d['max_util'].max()),
            'max_utilisation_median': float(d['max_util'].median()),
            'reward_reconstruction_max_abs_error': d['reward_check'],
            'critic': critic_bias(spec, critic, data, w_of, t_of, n)}
    for i in range(n):
        case = {'world': i, **{k: conditions[i][k] for k in ('youngs_modulus', 'cut_spring_stiffness', 'density',
                                                               'target_speed_mps', 'start_lift_m')}}
        for name, d in results.items():
            case[name] = {'ending': d['ending'][i], 'return': float(sum(v[i] for v in d['terms'].values())),
                          'steps': int(d['steps'][i]), 'final_progress': float(d['final_rho'][i]),
                          'speed_error_mps': float(d['speed_err'][i]), 'max_utilisation': float(d['max_util'][i])}
        out['cases'].append(case)
    fail = [c for c in out['cases'] if c['policy']['ending'] != 'success']
    out['policy_failures'] = {
        'count': len(fail),
        'press_success_on_them': sum(c['press']['ending'] == 'success' for c in fail),
        'saw_success_on_them': sum(c['press_saw']['ending'] == 'success' for c in fail),
        'cases': [{k: c[k] for k in ('world', 'youngs_modulus', 'cut_spring_stiffness', 'target_speed_mps')}
                  | {s: (c[s]['ending'], round(c[s]['final_progress'], 3)) for s in ('policy', 'press', 'press_saw')}
                  for c in fail]}
    out['seconds'] = time.monotonic() - started
    (args.output / 'report.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('cases', 'command', 'git')}, indent=1))


if __name__ == '__main__':
    main()
