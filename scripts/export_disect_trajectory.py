#!/usr/bin/env python3
"""Export a DiSECt SlicingMotion (optimize_slicing.py parameters) as a reference knife trajectory.

DiSECt evaluates the knife velocity at the start of every material substep and
holds it for that substep. Each exported interval stores the mean of its substep
velocities, so the exported displacement equals DiSECt's free-knife displacement
at every interval boundary.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backends/mujoco'))
from cutting_gpu_backend.reference import KnifeTrajectory


def slicing_velocity(time, frequency, amplitudes, pressing, waypoint_times, kernel_width):
    """DiSECt SlicingMotion.linear_velocity: RBF-weighted pressing (y) and sawing (z)."""
    time = np.asarray(time, dtype=np.float64)[:, None]
    weights = np.exp(-(time - np.asarray(waypoint_times)) ** 2 / kernel_width)
    velocity = np.zeros((len(time), 3))
    velocity[:, 1] = weights @ np.asarray(pressing, dtype=np.float64)
    velocity[:, 2] = np.sin(time[:, 0] * frequency) * (weights @ np.asarray(amplitudes, dtype=np.float64))
    return velocity


def export(frequency, amplitudes, pressing, duration, material_dt, interval, kernel_width,
           waypoint_times=None, source=None):
    substeps = round(duration / material_dt)
    per_interval = round(interval / material_dt)
    if abs(substeps * material_dt - duration) > 1e-12 or abs(per_interval * material_dt - interval) > 1e-12 \
            or substeps % per_interval:
        raise ValueError('Duration and interval must contain whole DiSECt substeps')
    if waypoint_times is None:
        waypoint_times = np.linspace(0., duration, len(amplitudes))
    velocity = slicing_velocity(np.arange(substeps) * material_dt, frequency, amplitudes, pressing,
                                waypoint_times, kernel_width)
    mean = velocity.reshape(-1, per_interval, 3).mean(1)
    times = np.arange(len(mean) + 1) * interval
    return KnifeTrajectory(times, mean, {
        'kind': 'disect_slicing_motion', 'frequency': float(frequency),
        'amplitudes': [float(v) for v in amplitudes], 'pressing_velocities': [float(v) for v in pressing],
        'waypoint_times': [float(v) for v in waypoint_times], 'kernel_width': kernel_width,
        'duration': duration, 'material_dt': material_dt, 'interval': interval, **(source or {})})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True, help='DiSECt-Optimization output directory')
    parser.add_argument('--parameters', choices=('checkpoint', 'initial'), default='checkpoint')
    parser.add_argument('--interval', type=float, default=.001)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    motion, config = manifest['motion'], manifest['config']
    waypoints = motion['waypoints']
    if args.parameters == 'checkpoint':
        saved = torch.load(args.run / 'last.pt', map_location='cpu', weights_only=False)
        frequency, amplitudes, pressing = (p.double().numpy() for p in saved['parameters'])
        source = {'parameters': 'checkpoint', 'next_iteration': saved['next_iteration']}
    else:
        frequency = motion['initial_frequency']
        amplitudes = [motion['initial_amplitudes']] * waypoints
        pressing = [motion['initial_pressing_velocities']] * waypoints
        source = {'parameters': 'initial', 'next_iteration': 0}
    source['run'] = str(args.run.resolve())
    trajectory = export(float(frequency), amplitudes, pressing, config['sim_duration'], config['sim_dt'],
                        args.interval, motion['kernel_width'], source=source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trajectory.save(args.output)
    end = trajectory.displacement(torch.tensor([trajectory.duration], dtype=torch.float64))[0]
    print(json.dumps({'output': str(args.output), 'intervals': len(trajectory.linear_velocity),
                      'duration': trajectory.duration, 'final_displacement_m': end.tolist()}))


if __name__ == '__main__':
    main()
