#!/usr/bin/env python3
"""Run predefined perturbation cases with one frozen policy, or compare two such runs.

    evaluate_perturbed.py run --checkpoint CKPT --cases 256 --seed-base 3000000 --output DIR
    evaluate_perturbed.py compare --a DIR_A --b DIR_B --output FILE

Every case runs to its own end (success, force limit, workspace or the 500-step time
limit); worlds take the next case when theirs ends. At the wall budget unfinished cases
are listed, not dropped silently.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.perturbed_eval import (CaseNoise, compare_cases, make_cases, outcome,
                                                summarize_cases)
from cutting_world_model.policy import SliceItPolicy
from cutting_world_model.provenance import git_revision
from cutting_world_model.task_heads import without_reward_settings


def run(args):
    from cutting_world_model.gpu_env import GPUEnvironment
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    env = GPUEnvironment(str((ROOT / 'backends/mujoco').resolve()), args.worlds)
    device = env.device
    frame = env.current
    metadata = ControlStateSpec.metadata_from_frame(env.metadata, {k: v[0].cpu().numpy() for k, v in frame.items()})
    spec = ControlStateSpec(metadata).to(device)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if without_reward_settings(saved['metadata']) != without_reward_settings(metadata):
        raise ValueError('Checkpoint and physical task metadata differ')
    policy = SliceItPolicy().to(device).eval()
    policy.load_state_dict(saved['learner']['policy'])
    cases = make_cases(args.cases, args.seed_base)
    dt = metadata['environment']['agent_control_dt']
    (args.output / 'manifest.json').write_text(json.dumps({
        'git': git_revision(ROOT), 'command': sys.argv, 'checkpoint': str(args.checkpoint),
        'policy_updates': saved['learner'].get('updates'), 'cases': cases}, indent=1) + '\n')
    queue = list(range(len(cases)))
    active = [None] * args.worlds
    rows, records = [], {}

    def start(world):
        if queue:
            index = queue.pop(0)
            active[world] = index
            records[index] = {'noise': CaseNoise(cases[index]), 'steps': 0, 'return': 0.,
                              'max_force': 0., 'impulse': 0., 'margin': 0.}
        else:
            active[world] = None

    for world in range(args.worlds):
        start(world)
    history = frame['observation'].float()[None].repeat(policy.history, 1, 1)
    stop_reason = None
    with torch.no_grad():
        while any(a is not None for a in active):
            if time.monotonic() - started > args.wall_seconds:
                stop_reason = 'wall budget'
                break
            clean = policy(history, frame['target_pose'].float())
            noise = torch.stack([records[a]['noise'](records[a]['steps']) if a is not None else torch.zeros(6)
                                 for a in active]).to(device)
            frame, reward, term, trunc, info = env.step(spec.execute(clean + noise))
            history = torch.cat((history[1:], frame['observation'].float()[None]), 0)
            ended = (term | trunc).cpu()
            for world, index in enumerate(active):
                if index is None:
                    continue
                rec = records[index]
                rec['steps'] += 1
                rec['return'] += float(reward[world])
                rec['max_force'] = max(rec['max_force'], float(info['peak_force'][world]))
                rec['impulse'] += float(info['material_impulse'][world])
                rec['margin'] = max(rec['margin'], float(info['safety_margin'][world]))
                if bool(ended[world]):
                    rows.append({'id': index, 'outcome': outcome(
                        bool(info['success'][world]), bool(info['collision'][world]),
                        bool(info['outside'][world]), bool(trunc[world])),
                        'steps': rec['steps'], 'return': rec['return'],
                        'max_material_force_n': rec['max_force'],
                        'mean_material_force_n': rec['impulse'] / (rec['steps'] * dt),
                        'safety_margin': rec['margin'],
                        'released_springs': int(info['released_springs'][world]),
                        **{k: cases[index][k] for k in ('sigma', 'start_step', 'mode')}})
                    start(world)
            if bool(ended.any()):
                mask = ended.to(device)
                frame = env.reset(mask)
                history[:, mask] = frame['observation'][mask].float()[None]
    rows.sort(key=lambda r: r['id'])
    unfinished = sorted(set(range(len(cases))) - {r['id'] for r in rows})
    with (args.output / 'cases.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    summary = {'summary': summarize_cases(rows) if rows else None, 'unfinished_cases': unfinished,
               'stop_reason': stop_reason, 'seconds': time.monotonic() - started}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


def compare(args):
    load = lambda d: [json.loads(line) for line in (d / 'cases.jsonl').read_text().splitlines()]
    a, b = load(args.a), load(args.b)
    report = {'a': str(args.a), 'b': str(args.b), 'summary_a': summarize_cases(a),
              'summary_b': summarize_cases(b), 'a_minus_b': compare_cases(a, b)}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['a_minus_b']['all_cases']['success']))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    r = sub.add_parser('run')
    r.add_argument('--checkpoint', type=Path, required=True)
    r.add_argument('--cases', type=int, default=256)
    r.add_argument('--seed-base', type=int, default=3_000_000)
    r.add_argument('--worlds', type=int, default=32)
    r.add_argument('--wall-seconds', type=float, default=2400.)
    r.add_argument('--output', type=Path, required=True)
    c = sub.add_parser('compare')
    c.add_argument('--a', type=Path, required=True)
    c.add_argument('--b', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args) if args.mode == 'run' else compare(args)


if __name__ == '__main__':
    main()
