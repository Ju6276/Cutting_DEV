#!/usr/bin/env python3
"""Supplementary episodes for the WM reliability check (PLAN, "Next experiment", Data).

Frozen policies act in the benchmark environment. Each episode draws its own configuration
from its own random stream and records it: policy, perturbation scale ``sigma`` (Gaussian
noise added to the policy's clean action before the action mapper clips it) and the first
perturbed step. One unperturbed episode per policy is recorded first (deterministic inputs
give the same trajectory again). Episodes are assigned to ``dev`` or ``final`` by their
index; the two sets use disjoint stream seeds and no episode belongs to both. Collection
stops at the wall budget, the episode limit, or when every terminal reason reaches its
target in both sets; it never changes thresholds or perturbations to fill a reason.
The stratified proportions are for diagnosis only, not policy success or exceedance rates.

Output: ``episodes.jsonl`` and ``episodes/*.npz`` in the layout read by
``cutting_world_model.reliability.load_episodes`` (split = dev / final), plus per-step
``material_peak_force`` and ``perturbed``; ``coverage.json`` reports the counts.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.gpu_env import GPUEnvironment
from cutting_world_model.policy import SliceItPolicy
from cutting_world_model.provenance import git_revision
from cutting_world_model.task_heads import without_reward_settings

REASONS = ('success', 'collision', 'outside')
SPLIT_SEED = {'dev': 1_000_000, 'final': 2_000_000}


def episode_reason(success, collision, outside, truncated):
    if success:
        return 'success'
    if collision:
        return 'collision'
    if outside:
        return 'outside'
    return 'truncated' if truncated else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-root', type=Path, default=ROOT / 'backends/mujoco')
    parser.add_argument('--policy', nargs=2, action='append', metavar=('LABEL', 'CHECKPOINT'), required=True)
    parser.add_argument('--perturbed-policies', nargs='+', required=True,
                        help='labels whose episodes are perturbed after the unperturbed one')
    parser.add_argument('--sigmas', type=float, nargs='+', default=[0.3, 0.6, 1.0])
    parser.add_argument('--start-steps', type=int, nargs='+', default=[0, 5, 10])
    parser.add_argument('--modes', nargs='+', choices=('iid', 'persistent'), default=['iid', 'persistent'],
                        help="'iid': fresh N(0, I) each step; 'persistent': one N(0, I) direction per episode")
    parser.add_argument('--worlds', type=int, default=32)
    parser.add_argument('--final-every', type=int, default=3, help='every n-th perturbed episode goes to final')
    parser.add_argument('--target-per-reason', type=int, nargs=2, default=[30, 15], metavar=('DEV', 'FINAL'))
    parser.add_argument('--max-episodes', type=int, default=400)
    parser.add_argument('--wall-seconds', type=float, default=3600.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    (args.output / 'episodes').mkdir(parents=True, exist_ok=False)
    env = GPUEnvironment(str(args.backend_root.resolve()), args.worlds)
    device = env.device
    frame = env.current
    one = {k: v[0].cpu().numpy() for k, v in frame.items()}
    metadata = ControlStateSpec.metadata_from_frame(env.metadata, one)
    spec = ControlStateSpec(metadata).to(device)
    labels = [label for label, _ in args.policy]
    policies = []
    for label, path in args.policy:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if without_reward_settings(saved['metadata']) != without_reward_settings(metadata):
            raise ValueError('Checkpoint and physical task metadata differ: ' + label)
        policy = SliceItPolicy().to(device).eval()
        policy.load_state_dict(saved['learner']['policy'])
        policies.append(policy)
    perturbed_ids = [labels.index(label) for label in args.perturbed_policies]
    manifest = {'git': git_revision(ROOT), 'command': sys.argv, 'policies': dict(args.policy),
                'perturbed_policies': args.perturbed_policies, 'sigmas': args.sigmas,
                'start_steps': args.start_steps, 'final_every': args.final_every,
                'split_seed_base': SPLIT_SEED, 'targets': dict(zip(('dev', 'final'), args.target_per_reason)),
                'modes': args.modes,
                'perturbation': ('clean + sigma * z on steps >= start, then the action mapper clips; '
                                 'z ~ N(0, I) per step (iid) or drawn once per episode (persistent)'),
                'use': 'diagnosis only; proportions are not policy rates'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    unperturbed_queue = list(range(len(policies)))
    counter = {'dev': 0, 'final': 0}
    perturbed_count = 0
    configs = [None] * args.worlds

    def new_config(world):
        nonlocal perturbed_count
        if unperturbed_queue:
            policy = unperturbed_queue.pop(0)
            split, sigma, start, seed, mode, direction = 'dev', 0., 0, None, None, None
        else:
            split = 'final' if perturbed_count % args.final_every == args.final_every - 1 else 'dev'
            perturbed_count += 1
            seed = SPLIT_SEED[split] + counter[split]
            gen = torch.Generator().manual_seed(seed)
            choice = lambda options: options[int(torch.randint(len(options), (1,), generator=gen))]
            policy, sigma, start = choice(perturbed_ids), choice(args.sigmas), choice(args.start_steps)
            mode = choice(args.modes)
            direction = torch.randn(6, generator=gen) if mode == 'persistent' else None
            configs_gen[world] = gen
        counter[split] += 1
        return {'policy': labels[policy], 'policy_index': policy, 'split': split, 'sigma': sigma,
                'start_step': start, 'seed': seed, 'mode': mode, 'direction': direction, 'rows': []}

    configs_gen = [None] * args.worlds
    for world in range(args.worlds):
        configs[world] = new_config(world)
    state = spec.pack_batch(frame).float()
    history = frame['observation'].float()[None].repeat(policies[0].history, 1, 1)
    steps = torch.zeros(args.worlds, dtype=torch.long)
    coverage = {split: {r: 0 for r in REASONS + ('truncated',)} for split in ('dev', 'final')}
    saved_episodes, stop_reason = 0, None
    with (args.output / 'episodes.jsonl').open('w') as index:
        while True:
            if time.monotonic() - started > args.wall_seconds:
                stop_reason = 'wall budget'
                break
            if saved_episodes >= args.max_episodes:
                stop_reason = 'episode limit'
                break
            if all(coverage[s][r] >= t for s, t in zip(('dev', 'final'), args.target_per_reason) for r in REASONS):
                stop_reason = 'targets met'
                break
            goal = frame['target_pose'].float()
            clean = torch.zeros(args.worlds, 6, device=device)
            with torch.no_grad():
                for p in {c['policy_index'] for c in configs}:
                    worlds = torch.tensor([w for w, c in enumerate(configs) if c['policy_index'] == p], device=device)
                    clean[worlds] = policies[p](history[:, worlds], goal[worlds])
            noise = torch.zeros(args.worlds, 6)
            perturbed = torch.zeros(args.worlds, dtype=torch.bool)
            for w, c in enumerate(configs):
                if c['sigma'] > 0 and int(steps[w]) >= c['start_step']:
                    z = c['direction'] if c['mode'] == 'persistent' else torch.randn(6, generator=configs_gen[w])
                    noise[w] = c['sigma'] * z
                    perturbed[w] = True
            action = spec.execute(clean + noise.to(device))
            frame, reward, term, trunc, info = env.step(action)
            nxt = spec.pack_batch_transition(frame, state).float()
            arrays = {'state': state, 'next_state': nxt, 'clean_action': clean, 'action': action, 'goal': goal,
                      'reward': reward, 'terminated': term, 'truncated': trunc, 'success': info['success'],
                      'collision': info['collision'], 'outside': info['outside'],
                      'material_peak_force': info['peak_force'], 'perturbed': perturbed}
            arrays = {k: v.cpu().numpy() for k, v in arrays.items()}
            steps += 1
            ended = (term | trunc).cpu()
            for w in range(args.worlds):
                configs[w]['rows'].append({k: v[w] for k, v in arrays.items()})
                if not bool(ended[w]):
                    continue
                c = configs[w]
                rows = c.pop('rows')
                data = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
                reason = episode_reason(bool(data['success'][-1]), bool(data['collision'][-1]),
                                        bool(data['outside'][-1]), bool(data['truncated'][-1]))
                name = f"{c['split']}_{saved_episodes:04d}.npz"
                np.savez_compressed(args.output / 'episodes' / name, **data)
                item = {'path': name, 'split': c['split'], 'reason': reason, 'steps': len(rows),
                        'policy': c['policy'], 'sigma': c['sigma'], 'start_step': c['start_step'],
                        'seed': c['seed'], 'mode': c['mode'], 'world': w,
                        'direction': None if c['direction'] is None else c['direction'].tolist(),
                        'max_material_peak_force': float(data['material_peak_force'].max()),
                        'return': float(data['reward'].sum())}
                index.write(json.dumps(item) + '\n')
                index.flush()
                coverage[c['split']][reason] += 1
                saved_episodes += 1
                configs[w] = new_config(w)
            state = nxt
            history = torch.cat((history[1:], frame['observation'].float()[None]), 0)
            if bool(ended.any()):
                mask = ended.to(device)
                frame = env.reset(mask)
                state = spec.reset_batch_history(frame, state, mask)
                history[:, mask] = frame['observation'][mask].float()[None]
                steps[ended] = 0
    result = {'stop_reason': stop_reason, 'episodes': saved_episodes, 'coverage': coverage,
              'seconds': time.monotonic() - started,
              'insufficient': [f'{s}/{r}' for s in coverage for r in REASONS if coverage[s][r] < 10],
              'discarded_partial_episodes': sum(1 for c in configs if c['rows'])}
    (args.output / 'coverage.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
