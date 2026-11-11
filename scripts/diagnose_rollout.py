#!/usr/bin/env python3
"""Per-step diagnosis of first episodes: reference trajectory or deterministic policy.

Records the three target-dim pose errors (x, z, pitch; metres/rad), released springs,
limit ratio and downward speed each step, the ending and, for workspace endings, which
error dimension crossed its limit.
"""
import argparse
import collections
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model.multi_env import MultiConditionEnvironment  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--conditions', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, help='deterministic policy; omit for the reference trajectory')
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.conditions.read_text())
    reward = {'w_speed': float(config['w_speed'])} if config.get('w_speed') else None
    env = MultiConditionEnvironment.from_config(args.root / 'backends/mujoco', config, reward=reward)
    heads = env.heads
    device = env.device
    scale = torch.tensor(heads.max_distance, device=device)
    limit = torch.tensor(heads.env['workspace_limit'], device=device)
    policy = spec = None
    if args.checkpoint:
        from cutting_world_model.policy import SliceItPolicy
        from cutting_world_model.world_model import CuttingWorldModel
        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        spec = CuttingWorldModel(saved['metadata']).spec.to(device)
        policy = SliceItPolicy(int(saved['metadata'].get('goal_dim', 7))).to(device)
        policy.load_state_dict(saved['learner']['policy'])
        policy.eval()
    frame = env.current
    n = env.worlds
    conditions = env.conditions(range(n))
    total = frame['total_springs'][:, 0].cpu()
    history = frame['observation'].float()[None].repeat(policy.history if policy else 1, 1, 1)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    trace = [[] for _ in range(n)]
    result = {}
    obs0 = frame['observation']
    start_error = (obs0[:, :3] * scale).cpu()
    with torch.no_grad():
        for step in range(1, args.steps + 1):
            if policy:
                action = spec.execute(policy(history, MultiConditionEnvironment.goal(frame)))
                frame, reward, terminated, truncated, info = env.step(action)
            else:
                frame, reward, terminated, truncated, info = env.step(torch.zeros(n, 6, device=device), reference=True)
            obs = frame['observation']
            error = obs[:, :3] * scale
            speed = heads.cutting_speed(obs)
            history = torch.cat((history[1:], obs.float()[None]), 0)
            ended = terminated | truncated
            for w in alive.nonzero().flatten().tolist():
                trace[w].append([step, *[round(float(x), 4) for x in error[w]], int(info['released_springs'][w]),
                                 round(float(info['safety_margin'][w]), 3), round(float(speed[w]), 4),
                                 round(float(info['peak_force'][w]), 1)])
                if ended[w]:
                    ending = ('goal' if info['success'][w] else 'force_limit' if info['collision'][w]
                              else 'workspace' if info['outside'][w] else 'time_limit')
                    crossed = [name for name, e, l in zip('x z pitch'.split(), error[w].abs().tolist(), limit.tolist()) if e > l]
                    result[w] = {'ending': ending, 'steps': step, 'crossed': crossed}
            alive &= ~ended
            if bool(ended.any()):
                frame = env.reset(ended)
                history[:, ended] = frame['observation'][ended].float()[None]
            if not bool(alive.any()):
                break
    env.close()
    for w in range(n):
        result.setdefault(w, {'ending': 'unfinished', 'steps': args.steps, 'crossed': []})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'trace.json').write_text(json.dumps({'conditions': conditions, 'result': result, 'trace': trace,
                                                         'total_springs': total.tolist(),
                                                         'start_error': start_error.tolist()}))
    print('max_distance', heads.max_distance, 'workspace_limit', heads.env['workspace_limit'],
          'goal_threshold', heads.env['goal_threshold'])
    groups = collections.defaultdict(list)
    for w in range(n):
        groups[conditions[w]['group']].append(w)
    for g, ws in groups.items():
        c = collections.Counter((result[w]['ending'], tuple(result[w]['crossed'])) for w in ws)
        print('==', g, 'total springs', int(total[ws[0]]), 'start error x/z/pitch', [round(x, 4) for x in start_error[ws[0]].tolist()], dict(c))
        shown = set()
        for w in ws:
            key = result[w]['ending']
            if key in shown:
                continue
            shown.add(key)
            cnd = conditions[w]
            print(f"  world {w} {key} steps {result[w]['steps']} crossed {result[w]['crossed']} E {cnd['youngs_modulus']:.2f} "
                  f"ks {cnd['cut_spring_stiffness']:.2f} lift {cnd['start_lift_m']:.3f} v* {cnd['target_speed_mps']:.3f}")
            t = trace[w]
            stride = max(1, len(t) // 14)
            for row in t[::stride] + t[-3:]:
                print('    step %3d  ex %+.4f ez %+.4f epitch %+.4f  springs %3d  ratio %.2f  vdown %+.4f  matF %.1f' % tuple(row))


if __name__ == '__main__':
    main()
