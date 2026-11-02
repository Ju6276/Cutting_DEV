#!/usr/bin/env python3
"""Compare URDF kinematics and analytic state updates with complete SliceIt records."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutting_world_model.kinematics import pose_difference
from cutting_world_model.provenance import git_revision
from cutting_world_model.state_spec import CuttingStateSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    directory = Path(args.dataset)
    task = json.loads((directory / 'task.json').read_text())
    reports = []
    for path in sorted(directory.glob('episode_*.npz')):
        with np.load(path) as data:
            first = {key: value[0] for key, value in data.items() if key not in
                     ('actions', 'clean_actions', 'rewards', 'terminated', 'truncated')}
            metadata = CuttingStateSpec.metadata_from_frame(task, first)
            spec = CuttingStateSpec(metadata).double()
            states = torch.stack([spec.pack({key: data[key][t] for key in spec.layout}).double()
                                  for t in range(len(data['observation']))])
            # Use recorded float64 directly for the numerical equation comparison.
            states = torch.tensor(np.stack([np.concatenate([np.asarray(data[key][t]).reshape(-1)
                                  for key in spec.layout]) for t in range(len(states))]), dtype=torch.float64)
            predicted_pose = spec.fk(spec.field(states, 'joint_q'))
            recorded_pose = torch.tensor(data['ee_pose'], dtype=torch.float64)
            fk_error = pose_difference(predicted_pose, recorded_pose)
            measured = {name: spec.field(states[1:], name) for name in spec.layout}
            goal = torch.tensor(data['target_pose'][1:], dtype=torch.float64)
            action = torch.tensor(data['actions'], dtype=torch.float64)
            derived, reward, done = spec.derive(states[:-1], measured, action, goal)
            observation_error = (spec.field(derived, 'observation') - spec.field(states[1:], 'observation')).abs()
            collision = (spec.field(states[1:], 'peak_wrench') > spec.force_limits).any(-1)
            info = json.loads(path.with_suffix('.json').read_text()) if path.with_suffix('.json').exists() else None
            item = {
                'episode': path.name, 'transitions': len(action),
                'fk_position_max_m': float(fk_error[..., :3].abs().max()),
                'fk_rotation_max_rad': float(fk_error[..., 3:].abs().max()),
                'observation_max_abs_error': float(observation_error.max()),
                'motion_command_max_abs_error': float((spec.field(derived, 'motion_command') - spec.field(states[1:], 'motion_command')).abs().max()),
                'counter_max_abs_error': float((spec.field(derived, 'success_counter') - spec.field(states[1:], 'success_counter')).abs().max()),
                'reward_max_abs_error': float((reward - torch.tensor(data['rewards'])).abs().max()),
                'termination_mismatches': int((done != torch.tensor(data['terminated'])).sum()),
            }
            if info is not None:
                item['collision_mismatches'] = int((collision != torch.tensor([row['collision'] for row in info])).sum())
            reports.append(item)
    result = {'git': git_revision(Path(__file__).resolve().parents[1]), 'episodes': reports}
    (directory / 'state_interface_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
