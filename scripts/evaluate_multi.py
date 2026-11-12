#!/usr/bin/env python3
"""Frozen evaluation of a WM-method policy on the multi-object task (PLAN S4).

Builds the multi-object environment from an evaluation config (e.g. held-out material
scales or start heights), runs the deterministic policy (no exploration noise) until
every world has completed ``--episodes`` episodes, and writes one row per episode:
object group and sampled conditions, ending (goal / force limit / workspace / time),
steps and simulated seconds, peak material force, peak sensor-limit ratio, material
impulse, released-spring fraction (a proxy for separation, not physical separation),
and the mean target-speed error while the cut depth is not reached.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model.multi_env import MultiConditionEnvironment
from cutting_world_model.policy import SliceItPolicy
from cutting_world_model.provenance import git_revision
from cutting_world_model.world_model import CuttingWorldModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--conditions', type=Path, required=True, help='evaluation distribution JSON')
    parser.add_argument('--episodes', type=int, default=4, help='episodes per world')
    parser.add_argument('--max-hours', type=float, default=6.)
    parser.add_argument('--max-steps', type=int, help='functional checks only: stop after this many control steps')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    device = 'cuda:0'
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    metadata = saved['metadata']
    goal_dim = int(metadata.get('goal_dim', 7))
    if goal_dim != 8:
        raise ValueError('The checkpoint was not trained on the multi-object task (goal_dim 8)')
    spec = CuttingWorldModel(metadata).spec.to(device)
    policy = SliceItPolicy(goal_dim, int(metadata["state_shapes"]["observation"][0])).to(device)
    policy.load_state_dict(saved['learner']['policy'])
    policy.eval()
    config = json.loads(args.conditions.read_text())
    reward = {'w_speed': float(config['w_speed'])} if config.get('w_speed') else None
    if config.get('task') != metadata.get('free_cut_task'):
        raise ValueError('Evaluation task must match checkpoint task')
    env = MultiConditionEnvironment.from_config(ROOT / 'backends/mujoco', config, reward=reward)
    heads, dt = env.heads, metadata['environment']['agent_control_dt']
    (args.output / 'manifest.json').write_text(json.dumps({
        'git': git_revision(ROOT), 'command': sys.argv, 'checkpoint': str(args.checkpoint),
        'conditions': config, 'episodes_per_world': args.episodes,
        'policy': 'deterministic, no exploration noise'}, indent=2) + '\n')
    frame = env.current
    history = frame['observation'].float()[None].repeat(policy.history, 1, 1)
    n = env.worlds
    done_count = torch.zeros(n, dtype=torch.long, device=device)
    acc = {k: torch.zeros(n, device=device) for k in
           ('steps', 'peak_material', 'peak_ratio', 'impulse', 'speed_error', 'speed_steps', 'return')}
    rows = []
    conditions = env.conditions(range(n))
    with (args.output / 'episodes.jsonl').open('w') as log, torch.no_grad():
        step = 0
        while bool((done_count < args.episodes).any()):
            step += 1
            if args.max_steps is not None and step > args.max_steps:
                break
            if time.monotonic() - started > args.max_hours * 3600:
                raise TimeoutError('Evaluation reached its wall budget')
            goal = MultiConditionEnvironment.goal(frame)
            action = spec.execute(policy(history, goal))
            frame, reward, terminated, truncated, info = env.step(action)
            obs = frame['observation']
            cutting = ((obs[:, 1] * heads.max_distance[1]).abs() > heads.env['goal_threshold'][1]
                       if not metadata.get('free_cut_task') else torch.ones(n, device=device, dtype=torch.bool))
            speed_error = (heads.cutting_speed(obs) - frame['target_speed'][:, 0]).abs()
            acc['steps'] += 1
            acc['return'] += reward
            acc['peak_material'] = torch.maximum(acc['peak_material'], info['peak_force'].float())
            acc['peak_ratio'] = torch.maximum(acc['peak_ratio'], info['safety_margin'].float())
            acc['impulse'] += info['material_impulse'].float()
            acc['speed_error'] += torch.where(cutting, speed_error, 0.)
            acc['speed_steps'] += cutting.float()
            history = torch.cat((history[1:], obs.float()[None]), 0)
            ended = terminated | truncated
            for w in ended.nonzero().flatten().tolist():
                if int(done_count[w]) < args.episodes:
                    ending = (('separated' if metadata.get('free_cut_task') else 'goal') if info['success'][w] else 'force_limit' if info['collision'][w]
                              else 'workspace' if info['outside'][w] else 'time_limit')
                    row = {'world': w, 'episode': int(done_count[w]), **conditions[w], 'ending': ending,
                           'return_undiscounted': float(acc['return'][w]),
                           'steps': int(acc['steps'][w]), 'seconds': float(acc['steps'][w]) * dt,
                           'peak_material_force_n': float(acc['peak_material'][w]),
                           'peak_limit_ratio': float(acc['peak_ratio'][w]),
                           'material_impulse_ns': float(acc['impulse'][w]),
                           'released_fraction': float(info['released_springs'][w]) / float(frame['total_springs'][w, 0]),
                           'released_springs': int(info['released_springs'][w]),
                           'speed_error_mps': float(acc['speed_error'][w] / acc['speed_steps'][w])
                           if acc['speed_steps'][w] > 0 else None}
                    if metadata.get('free_cut_task'):
                        row.update(separated=bool(info['separated'][w]), success=bool(info['success'][w]),
                                   timeout=bool(info['timeout'][w]), truncated=bool(truncated[w]))
                    rows.append(row)
                    log.write(json.dumps(row) + '\n')
                    log.flush()
                done_count[w] += 1
            if bool(ended.any()):
                for k in acc:
                    acc[k][ended] = 0
                frame = env.reset(ended)
                conditions = env.conditions(range(n))
                history[:, ended] = frame['observation'][ended].float()[None]
    env.close()
    summary = {}
    for group in sorted({r['group'] for r in rows}):
        members = [r for r in rows if r['group'] == group]
        count = len(members)
        summary[group] = {'episodes': count,
                          **{f'{e}_rate': sum(r['ending'] == e for r in members) / count
                             for e in (('separated' if metadata.get('free_cut_task') else 'goal'), 'force_limit', 'workspace', 'time_limit')},
                          'mean_seconds': sum(r['seconds'] for r in members) / count,
                          'mean_peak_material_force_n': sum(r['peak_material_force_n'] for r in members) / count,
                          'mean_peak_limit_ratio': sum(r['peak_limit_ratio'] for r in members) / count}
    (args.output / 'summary.json').write_text(json.dumps(
        {'groups': summary, 'episodes': len(rows), 'wall_seconds': time.monotonic() - started}, indent=2) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
