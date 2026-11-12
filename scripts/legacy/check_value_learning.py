#!/usr/bin/env python3
"""Compare a frozen critic with observed returns from complete recorded episodes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.policy import SliceItCritic
from cutting_world_model.provenance import git_revision
from cutting_world_model.state_spec import CuttingStateSpec
from cutting_world_model.task_heads import pose_error
from cutting_world_model.value_diagnostics import complete_episode_returns, value_return_metrics
from third_party.supermimic.trainer_config import PolicyTrainerCfg


def check(checkpoint, episode_paths, device='cpu', gamma=PolicyTrainerCfg().gamma):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    metadata = saved['metadata']
    layout = metadata.get('prediction_layout')
    if layout not in (None, 'control_history_v2'):
        raise ValueError('Unsupported prediction layout: ' + str(layout))
    spec_class = ControlStateSpec if layout == 'control_history_v2' else CuttingStateSpec
    spec = spec_class(metadata).to(device)
    critic = SliceItCritic(spec.state_dim).to(device)
    critic.load_state_dict(saved['learner']['critic'])
    critic.eval()
    rows, skipped, all_values, all_returns = [], [], [], []
    for path in episode_paths:
        with np.load(path) as data:
            rewards = torch.as_tensor(data['rewards'], dtype=torch.float64, device=device)
            terminated = np.asarray(data['terminated'], dtype=bool)
            truncated = np.asarray(data['truncated'], dtype=bool)
            if not len(rewards) or not terminated[-1] or truncated[-1]:
                skipped.append(str(path))
                continue
            returns = complete_episode_returns(rewards, terminated, truncated, gamma)
            if isinstance(spec, ControlStateSpec):
                states = spec.pack_episode(data, device=device)
            else:
                fields = spec.physical_fields + spec.history_fields
                states = torch.stack([
                    spec.pack({key: data[key][t] for key in fields}, device=device)
                    for t in range(len(data['joint_q']))
                ])
            goals = torch.as_tensor(data['target_pose'], dtype=torch.float32, device=device)
            if len(states) != len(rewards) + 1 or len(goals) != len(states):
                raise ValueError('Expected T actions and T+1 states/goals: ' + str(path))
            with torch.no_grad():
                values = critic(spec.canonical(states), goals,
                                pose_error(goals, spec.field(states, 'previous_ee_pose'))).double()
            rows.append({
                'episode': str(path),
                'return_undiscounted': float(rewards.sum()),
                'return_discounted_start': float(returns[0]),
                'critic_start': float(values[0]),
                'terminal_state_value': float(values[-1]),
                **value_return_metrics(values[:-1], returns),
            })
            all_values.append(values[:-1])
            all_returns.append(returns)
    if not rows:
        raise ValueError('No complete naturally terminated episodes were supplied')
    return {
        'checkpoint': str(checkpoint),
        'checkpoint_git': saved.get('git'),
        'git': git_revision(Path(__file__).resolve().parents[2]),
        'update': saved['learner']['updates'],
        'gamma': gamma,
        'return_type': 'observed_behavior_policy_discounted_return',
        'episode_count': len(rows),
        'skipped_incomplete_episodes': skipped,
        **value_return_metrics(torch.cat(all_values), torch.cat(all_returns)),
        'episodes': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--episodes', required=True, type=Path, nargs='+',
                        help='Episode NPZ files or directories containing episode_*.npz')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gamma', type=float, default=PolicyTrainerCfg().gamma)
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = set()
    for path in args.episodes:
        paths.update(path.glob('episode_*.npz') if path.is_dir() else [path])
    report = check(args.checkpoint, sorted(paths), args.device, args.gamma)
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
