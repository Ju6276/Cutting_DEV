#!/usr/bin/env python3
"""Compare world-model loss-term gradients on identical saved batches (no parameter update).

Question: do the state-prediction loss and the terminal BCE (and the state-loss fields
joint_q / joint_qd / wrench_history / peak_wrench) pull the shared dynamics MLP in conflicting
directions, or in a common direction? For each checkpoint the same windows are drawn from the
training split of a fixed-data WM run (e.g. runs/wm_frozen_fit_20261105) with the training batch
size and window, the loss is split exactly (wm_diagnostics.window_loss_components), and the
gradient of each component with respect to the dynamics MLP is compared, raw and preconditioned
by the checkpoint's Adam second moment (the update direction Adam would take).
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cutting_world_model.policy import SliceItCritic, SliceItPolicy
from cutting_world_model.provenance import git_revision
from cutting_world_model.sequence_replay import SequenceReplay
from cutting_world_model.trainer import OnlineLearner, chunks
from cutting_world_model.world_model import CuttingWorldModel
from cutting_world_model.wm_diagnostics import (adam_preconditioner, cosine_matrix, flat_grad,
                                                window_loss_components)


def load_learner(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    wm = CuttingWorldModel(checkpoint['metadata']).to(device)
    learner = OnlineLearner(wm, SliceItPolicy().to(device), SliceItCritic(wm.spec.state_dim).to(device),
                            microbatch=64)
    learner.restore(checkpoint['learner'])
    return learner, checkpoint['metadata']


def training_replay(run, spec, device):
    """Training-split episodes as the fixed-data run stored them (its run.py load/replay)."""
    episodes = [json.loads(line) for line in (run / 'episodes.jsonl').read_text().splitlines() if line]
    episodes = [e for e in episodes if e['split'] == 'train']
    data = []
    for ep in episodes:
        with np.load(run / 'episodes' / ep['path']) as d:
            data.append({'states': torch.as_tensor(np.concatenate((d['state'][:1], d['next_state']), 0), device=device),
                         'actions': torch.as_tensor(d['action'], device=device),
                         'clean': torch.as_tensor(d['clean_action'], device=device),
                         'goals': torch.as_tensor(d['goal'], device=device),
                         'reward': torch.as_tensor(d['reward'], device=device),
                         'done': torch.as_tensor(d['terminated'], device=device)})
    capacity = max(9, max(len(e['actions']) + 2 for e in data))
    buffer = SequenceReplay(len(data), capacity, spec.state_dim, 6, 7, device)
    one = lambda value, dtype=None: torch.tensor([value], device=device, dtype=dtype)
    for world, ep in enumerate(data):
        ids, n = one(world), len(ep['actions'])
        zero = torch.zeros(1, 6, device=device)
        for t in range(n + 1):
            terminal = t == n
            buffer.add(ids, ep['states'][t:t + 1], zero if terminal else ep['clean'][t:t + 1],
                       zero if terminal else ep['actions'][t:t + 1], ep['goals'][0:1],
                       one(0. if terminal else float(ep['reward'][t])),
                       one(float(bool(ep['done'][-1])) if terminal else 0.), one(t), one(world),
                       one(terminal), one(t == 0))
    return buffer, len(data)


def batch_gradients(learner, batch):
    """Per-component dynamics gradients of one training batch, normalized like fit_world_model."""
    wm = learner.wm
    dynamics = list(wm.dynamics.parameters())
    head = list(wm.term_head.parameters())
    valid = batch.valid_steps
    count = valid.sum().to(batch.states.dtype)
    grads, head_grad, sums = {}, None, {}
    for part in chunks(batch.states.shape[1], learner.microbatch):
        fields, terminal, _ = window_loss_components(
            wm, batch.states[:, part], batch.a_exec[:, part], batch.xi[part], batch.done[:, part], valid[:, part])
        components = {**fields, 'terminal_bce': terminal}
        for name, value in components.items():
            g = flat_grad(value / count, dynamics)
            grads[name] = grads.get(name, 0.) + g
            sums[name] = sums.get(name, 0.) + float(value.detach())
        g = flat_grad(terminal / count, head, retain_graph=False)
        head_grad = g if head_grad is None else head_grad + g
    grads['state_total'] = sum(grads[n] for n in wm.spec.physical_fields)
    grads['total'] = grads['state_total'] + grads['terminal_bce']
    losses = {name: value / float(count) for name, value in sums.items()}
    return grads, head_grad, losses, float(count), float(torch.where(valid, batch.done, 0.).sum())


def summarize(rows):
    keys = rows[0].keys()
    return {k: {'mean': float(np.mean([r[k] for r in rows])), 'std': float(np.std([r[k] for r in rows]))}
            for k in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True, help='fixed-data WM run with episodes/ and episodes.jsonl')
    parser.add_argument('--checkpoint', type=Path, action='append', required=True,
                        help='learner checkpoint(s); every one sees the same batches')
    parser.add_argument('--batches', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20261108)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = 'cuda:0'
    torch.set_num_threads(2)
    started = time.monotonic()
    report = {'git': git_revision(ROOT), 'command': sys.argv, 'run': str(args.run.resolve()),
              'batches': args.batches, 'seed': args.seed, 'checkpoints': {}}
    for path in args.checkpoint:
        learner, metadata = load_learner(path, device)
        learner.wm.train()
        buffer, episodes = training_replay(args.run, learner.wm.spec, device)
        cfg = learner.wcfg
        dynamics = list(learner.wm.dynamics.parameters())
        precondition = adam_preconditioner(learner.wopt, dynamics)
        torch.manual_seed(args.seed)
        per_batch, sums = [], {}
        for index in range(args.batches):
            batch = buffer.sample_windows(cfg.batch_size, cfg.window)
            grads, head_grad, losses, count, terminals = batch_gradients(learner, batch)
            names = list(grads)
            row = {'valid_steps': count, 'terminal_labels': terminals,
                   **{f'loss/{k}': v for k, v in losses.items()},
                   **{f'norm/{k}': float(g.norm()) for k, g in grads.items()},
                   'norm/terminal_head': float(head_grad.norm())}
            raw = cosine_matrix({k: grads[k] for k in names})
            adam = cosine_matrix({k: grads[k] * precondition for k in names})
            for a in names:
                for b in names:
                    if a < b:
                        row[f'cos/{a}|{b}'] = raw[a][b]
                        row[f'adam_cos/{a}|{b}'] = adam[a][b]
            per_batch.append(row)
            for k, g in grads.items():
                sums[k] = sums.get(k, 0.) + g
            print(json.dumps({'checkpoint': str(path), 'batch': index,
                              'cos_state_terminal': row['cos/state_total|terminal_bce'],
                              'adam_cos_state_terminal': row['adam_cos/state_total|terminal_bce']}), flush=True)
        report['checkpoints'][str(path)] = {
            'wm_updates': learner.wm_updates, 'wm_optimizer_steps': learner.optimizer_steps['world_model'],
            'train_episodes': episodes, 'batch_size': cfg.batch_size, 'window': cfg.window,
            'per_batch_summary': summarize(per_batch),
            'summed_cos': cosine_matrix(sums),
            'summed_adam_cos': cosine_matrix({k: v * precondition for k, v in sums.items()}),
            'summed_norm': {k: float(v.norm()) for k, v in sums.items()}}
        del learner
        torch.cuda.empty_cache()
    report['wall_seconds'] = time.monotonic() - started
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
