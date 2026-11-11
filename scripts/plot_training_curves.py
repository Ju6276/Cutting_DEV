#!/usr/bin/env python3
"""Learning curves of online training runs (multi-object task), compared on one figure set.

Reads each run's ``episodes.jsonl`` (one row per finished episode, with the object in
``condition.group``) and ``updates.jsonl``. Episode outcomes are binned by physical
transitions at the episode's end (iteration × worlds). Writes PNG figures and a CSV of
the binned values. Needs only numpy and matplotlib (no simulator).

Usage: plot_training_curves.py --run base <dir> --run risk <dir> --output <dir>
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUTCOMES = (('success', 'goal reached'), ('force_limit', 'force-limit failure'),
            ('workspace', 'workspace failure'), ('time_limit', 'time limit'))


def rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def binned(episodes, worlds, width):
    """Per object: bin centres (transitions) -> outcome rates, mean return, mean peak force."""
    out = {}
    for group in sorted({e.get('condition', {}).get('group', 'all') for e in episodes}):
        members = [e for e in episodes if e.get('condition', {}).get('group', 'all') == group]
        x = np.array([e['iteration'] * worlds for e in members], float)
        edges = np.arange(0, x.max() + width, width)
        index = np.digitize(x, edges) - 1
        table = []
        for b in np.unique(index):
            m = [members[i] for i in np.nonzero(index == b)[0]]
            row = {'transitions': float(edges[b] + width / 2), 'episodes': len(m),
                   'return': float(np.mean([e['return'] for e in m])),
                   'steps': float(np.mean([e['steps'] for e in m])),
                   'peak_material_force_n': float(np.mean([e.get('peak_material_force_n', np.nan) for e in m])),
                   'peak_limit_ratio': float(np.mean([e.get('peak_limit_ratio', np.nan) for e in m]))}
            for key, _ in OUTCOMES:
                row[key] = float(np.mean([e.get('ending') == key for e in m]))
            table.append(row)
        out[group] = table
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', nargs=2, action='append', metavar=('LABEL', 'DIR'), required=True)
    parser.add_argument('--bin', type=float, default=1e5, help='bin width in physical transitions')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data, updates = {}, {}
    for label, directory in args.run:
        directory = Path(directory)
        worlds = json.loads((directory / 'progress.json').read_text())['worlds']
        episodes = rows(directory / 'episodes.jsonl') if (directory / 'episodes.jsonl').exists() else []
        data[label] = binned(episodes, worlds, args.bin) if episodes else {}
        updates[label] = rows(directory / 'updates.jsonl') if (directory / 'updates.jsonl').exists() else []
    groups = sorted({g for d in data.values() for g in d})
    with (args.output / 'curves.csv').open('w', newline='') as f:
        writer = None
        for label, by_group in data.items():
            for group, table in by_group.items():
                for row in table:
                    row = {'run': label, 'object': group, **row}
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
    metrics = [k for k, _ in OUTCOMES] + ['return', 'peak_material_force_n', 'peak_limit_ratio', 'steps']
    titles = dict(OUTCOMES, **{'return': 'episode return', 'peak_material_force_n': 'peak material force (N)',
                               'peak_limit_ratio': 'peak sensor-limit ratio', 'steps': 'episode steps'})
    for metric in metrics:
        fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 3.6), squeeze=False, sharey=True)
        for ax, group in zip(axes[0], groups):
            for label, by_group in data.items():
                table = by_group.get(group, [])
                if table:
                    ax.plot([r['transitions'] / 1e6 for r in table], [r[metric] for r in table], label=label)
            ax.set_title(group)
            ax.set_xlabel('physical transitions (M)')
            ax.grid(alpha=.3)
        axes[0][0].set_ylabel(titles[metric])
        axes[0][-1].legend()
        fig.tight_layout()
        fig.savefig(args.output / f'episodes_{metric}.png', dpi=130)
        plt.close(fig)
    learner = [('world_model', 'loss', 'WM loss'), ('world_model', 'term_acc', 'WM terminal accuracy'),
               ('policy', 'critic_loss', 'critic loss'), ('policy', 'imag_objective', 'imagined objective'),
               ('policy', 'risk_lambda', 'risk weight lambda'), ('policy', 'imag_risk_upper_mean', 'imagined risk upper'),
               ('world_model', 'risk_coverage', 'risk-head coverage'), ('world_model', 'risk_event_recall', 'risk-head event recall')]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for ax, (section, key, title) in zip(axes.flat, learner):
        for label, rows_ in updates.items():
            points = [(r['transitions'] / 1e6, r[section][key]) for r in rows_
                      if isinstance(r.get(section), dict) and r[section].get(key) is not None
                      and np.isfinite(r[section][key])]
            if points:
                ax.plot(*zip(*points), label=label, lw=.8)
        ax.set_title(title)
        ax.set_xlabel('physical transitions (M)')
        ax.grid(alpha=.3)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(args.output / 'learner_metrics.png', dpi=130)
    print('wrote', args.output)


if __name__ == '__main__':
    main()
