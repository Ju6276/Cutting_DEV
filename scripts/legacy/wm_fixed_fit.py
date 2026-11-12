#!/usr/bin/env python3
"""Fit a saved WM on fixed recorded episodes with one terminal-gradient route (PLAN, Step 3).

Repeats the 2026-11-05 fixed-data fit (`runs/wm_frozen_fit_20261105/run.py`) without its
collection: same start checkpoint, training split, replay layout, learner configuration and
round budget. Policy, critic and their optimizers stay frozen (checked). Two runs that
differ only in ``--route`` share initialization, optimizer state, seed (hence minibatch
order) and rounds.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model.policy import SliceItCritic, SliceItPolicy
from cutting_world_model.provenance import git_revision
from cutting_world_model.reliability import build_replay, load_episodes
from cutting_world_model.trainer import OnlineLearner
from cutting_world_model.world_model import TERMINAL_GRADIENTS, CuttingWorldModel


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=Path, required=True, help='learner checkpoint (WM562)')
    parser.add_argument('--episodes', type=Path, required=True, help='directory with episodes.jsonl')
    parser.add_argument('--route', choices=TERMINAL_GRADIENTS, required=True)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--save-at', type=int, nargs='*', default=[10, 25, 50, 100])
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--max-seconds', type=float, default=1500.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    device = 'cuda:0'
    checkpoint = torch.load(args.start, map_location=device, weights_only=False)
    wm = CuttingWorldModel(checkpoint['metadata']).to(device)
    learner = OnlineLearner(wm, SliceItPolicy().to(device), SliceItCritic(wm.spec.state_dim).to(device),
                            microbatch=64, wm_terminal_gradient=args.route)
    learner.restore(checkpoint['learner'], change_wm_terminal_gradient=True)
    learner.policy.requires_grad_(False)
    learner.critic.requires_grad_(False)
    frozen = {name: copy.deepcopy(obj.state_dict()) for name, obj in
              (('policy', learner.policy), ('critic', learner.critic),
               ('policy_optimizer', learner.popt), ('critic_optimizer', learner.copt))}
    train = load_episodes(args.episodes, 'train', device)
    if not train:
        raise RuntimeError('No training episodes')
    replay = build_replay(wm.spec, train, device)
    manifest = {'git': git_revision(ROOT), 'command': sys.argv, 'start': str(args.start),
                'episodes': str(args.episodes), 'train_episodes': len(train), 'route': args.route,
                'seed': args.seed, 'rounds': args.rounds, 'start_wm_updates': learner.wm_updates,
                'start_optimizer_steps': learner.optimizer_steps,
                'world_model_lr': learner.wopt.param_groups[0]['lr']}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    learner.wm.train()
    torch.manual_seed(args.seed)
    stop_reason = None
    for update in range(1, args.rounds + 1):
        if time.monotonic() - started > args.max_seconds:
            stop_reason = 'wall budget'
            break
        metrics = learner.fit_world_model(replay)
        with (args.output / 'updates.jsonl').open('a') as f:
            f.write(json.dumps({'update': update, 'seconds': time.monotonic() - started, **metrics}) + '\n')
        if update in args.save_at or update == args.rounds:
            torch.save({'metadata': checkpoint['metadata'], 'learner': learner.checkpoint()},
                       args.output / f'wm_{update:03d}.pt')
    unchanged = {name: same(state, obj.state_dict()) for (name, state), obj in zip(
        frozen.items(), (learner.policy, learner.critic, learner.popt, learner.copt))}
    if not all(unchanged.values()):
        raise RuntimeError('Frozen objects changed: %r' % unchanged)
    summary = {'completed_rounds': update if stop_reason is None else update - 1,
               'stop_reason': stop_reason, 'seconds': time.monotonic() - started,
               'frozen_unchanged': unchanged, 'wm_updates': learner.wm_updates}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
