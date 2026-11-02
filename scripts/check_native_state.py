#!/usr/bin/env python3
"""Check saved native SliceIt controller/filter state and paused observation alignment."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cutting_world_model.kinematics import pose_difference
from cutting_world_model.state_spec import CuttingStateSpec
from cutting_world_model.provenance import git_revision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    directory = Path(args.directory)
    task = json.loads((directory / 'task.json').read_text())
    if task.get('state_layout') != 'native_ros_recurrent_v1':
        raise ValueError('Expected native ROS recurrent state')
    maxima = {}
    minima = {}
    frames = episodes = 0
    def metric(name, value):
        values = np.asarray(value)
        maxima[name] = max(maxima.get(name, 0.), float(np.max(np.abs(values))))
        minima[name] = min(minima.get(name, float('inf')), float(np.min(values)))
    for path in sorted(directory.glob('episode_*.npz')):
        with np.load(path) as data:
            initial = {key: data[key][0] for key in data.files}
            spec = CuttingStateSpec(CuttingStateSpec.metadata_from_frame(task, initial)).double()
            for key in spec.physical_fields + spec.history_fields:
                if not np.isfinite(data[key]).all():
                    raise ValueError('Non-finite field ' + key)
            if not np.all(data['controller_command_valid'] == 1):
                raise ValueError('Invalid startup controller snapshot was recorded')
            for frame in range(len(data['joint_q'])):
                packed = spec.pack({key: data[key][frame] for key in spec.physical_fields + spec.history_fields})
                if packed.shape != (spec.state_dim,):
                    raise ValueError('Native state packing failed')
            q = torch.tensor(data['controller_q'], dtype=torch.float64)
            pose = torch.tensor(data['controller_cached_tip_pose'], dtype=torch.float64)
            metric('controller_fk_position', pose_difference(spec.fk(q), pose).detach().numpy()[:, :3])
            metric('controller_fk_rotation', pose_difference(spec.fk(q), pose).detach().numpy()[:, 3:])
            metric('observation_native_joint_q', data['joint_q'] - data['controller_measured_q'])
            metric('observation_native_joint_qd', data['joint_qd'] - data['controller_measured_qd'])
            metric('internal_measured_velocity_difference', data['controller_velocity'] - data['joint_qd'])
            metric('internal_saturated_command_difference', data['controller_q'] - data['controller_previous_position_command'])
            metric('observation_wrench_queue', data['wrench_history'] - data['wrench_queue'][:, :data['wrench_history'].shape[1]])
            observation_time = data['observation_ros_time'].reshape(-1)
            metric('bridge_age_seconds', observation_time - data['ros_time'].reshape(-1))
            for source in ('controller', 'controller_hardware', 'filter', 'joint', 'wrench'):
                metric(source + '_age_seconds', observation_time - data[source + '_ros_time'].reshape(-1))
            for source in ('filter_raw', 'filter_added'):
                metric(source + '_age_seconds', data[source + '_age'])
            metric('filter_history_count', data['filter_history_count'])
            metric('wrench_queue_count', data['wrench_queue_count'])
            frames += len(q)
            episodes += 1
    if not frames:
        raise ValueError('No saved episodes')
    report = {'git': git_revision(Path(__file__).resolve().parents[1]), 'episodes': episodes,
              'frames': frames, 'state_dim': spec.state_dim, 'delta_dim': spec.delta_dim,
              'max_abs': maxima, 'min': minima}
    (directory / 'native_state_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
