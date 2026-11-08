#!/usr/bin/env python3
"""Evaluate WM checkpoints on recorded episodes (PLAN, Steps 1 and 3).

Per episode: H1/H8 open-loop joint, knife-tip and force errors (with the persistence
reference), force-limit misses, the three-way terminal check (true state + task predicate,
predicted state + task predicate, predicted state + terminal head), and full open-loop
first-terminal timing. Summaries are split by terminal reason. ``--compare A B`` adds the
episode-paired bootstrap of A − B for the plan's metrics.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model.provenance import git_revision
from cutting_world_model.reliability import evaluate, load_episodes, paired_bootstrap, summarize
from cutting_world_model.world_model import CuttingWorldModel

# (metric path, lower is better); primary metrics first.
METRICS = [
    ('H8/tip_error_mm', True), ('H8/joint_rmse_rad', True),
    ('H1/tip_error_mm', True), ('H1/joint_rmse_rad', True),
    ('H1/force_abs_error_n', True), ('H8/force_abs_error_n', True),
    ('H1/force_underestimate_n', True), ('H8/force_underestimate_n', True),
    ('H1/missed_limit_steps', True), ('H8/missed_limit_steps', True),
    ('H1/predicted_predicate/missed', True), ('H8/predicted_predicate/missed', True),
    ('H1/predicted_predicate/false_alarms', True), ('H8/predicted_predicate/false_alarms', True),
    ('H1/terminal_head/missed', True), ('H8/terminal_head/missed', True),
    ('H1/terminal_head/false_alarms', True), ('H8/terminal_head/false_alarms', True),
    ('full/end_tip_error_mm', True),
]
REPORTED = [m for m, _ in METRICS] + [
    'label_mismatches', 'reason_mismatches', 'H1/persistence_force_abs_error_n',
    'H8/persistence_force_abs_error_n', 'H1/force_error_n', 'H8/force_error_n',
    'H1/true_predicate/missed', 'H1/true_predicate/false_alarms',
    'full/predicted_predicate/offset_steps', 'full/terminal_head/offset_steps']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--split', default='heldout')
    parser.add_argument('--checkpoint', nargs=2, action='append', metavar=('LABEL', 'PATH'), required=True)
    parser.add_argument('--compare', nargs=2, action='append', metavar=('A', 'B'), default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = 'cuda:0'
    episodes = load_episodes(args.episodes, args.split, device)
    if not episodes:
        raise RuntimeError('No episodes in split ' + args.split)
    report = {'git': git_revision(ROOT), 'command': sys.argv, 'episodes': len(episodes),
              'split': args.split, 'checkpoints': {}, 'comparisons': {}}
    rows = {}
    for label, path in args.checkpoint:
        saved = torch.load(path, map_location=device, weights_only=False)
        wm = CuttingWorldModel(saved['metadata']).to(device)
        wm.load_state_dict(saved['learner']['world_model'])
        wm.eval()
        rows[label] = evaluate(wm, episodes)
        report['checkpoints'][label] = {
            'path': path, 'route': saved['learner'].get('wm_terminal_gradient', 'dynamics'),
            'wm_updates': saved['learner']['training_stage']['wm_updates'],
            'summary': summarize(rows[label], REPORTED), 'episodes': rows[label]}
        mismatches = sum(r['label_mismatches'] + r['reason_mismatches'] for r in rows[label])
        print(label, 'route-1 label/reason mismatches:', mismatches, flush=True)
    for a, b in args.compare:
        report['comparisons'][f'{a} - {b}'] = {
            path: paired_bootstrap(rows[a], rows[b], path, lower) for path, lower in METRICS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
