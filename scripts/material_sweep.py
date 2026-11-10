#!/usr/bin/env python3
"""Reference-trajectory sweep over geometry x material scenes (multi-condition task, S0).

Each exported scene runs the scene's constant-velocity knife reference on the full
UR5e/FDCC/DiSECt environment. The force-limit termination is disabled
(``--measure-only``) so the sweep measures the sensor and material forces a complete cut
needs; it is a measurement for choosing per-object force limits, not a task episode.
Geometry is recorded for rendering with ``render_episode.py``.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from scripts.record_episode import GeometryRecorder
from cutting_world_model.provenance import git_revision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-root', type=Path, default=ROOT / 'backends/mujoco')
    parser.add_argument('--scenes', nargs='+', required=True, help='names under runs/sweep')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--substeps', type=int, default=6)
    parser.add_argument('--reference-duration', type=float,
                        help='keep the scene reference velocity for this long (s); default: scene duration')
    parser.add_argument('--cut-fraction', type=float, default=.9,
                        help='released-spring fraction that counts as cut through')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from cutting_world_model.gpu_env import GPUEnvironment
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.scenes:
        out = args.output / name
        if (out / 'report.json').exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        env = GPUEnvironment(str(args.backend_root), 1, scene=f'runs/sweep/{name}/scene.pt')
        env.physics.force_limits = torch.full_like(env.physics.force_limits, 1e6)
        if args.reference_duration is not None:
            from cutting_gpu_backend.reference import KnifeTrajectory
            physics = env.physics
            physics.set_reference_trajectory(KnifeTrajectory.constant(
                physics.material.reference_linear_velocity.reshape(-1, 3)[0].to(physics.dtype),
                args.reference_duration, kind='extended_scene_velocity'))
        total_springs = int(env.current['cut_spring_ke'].shape[-1])
        cut_step = None
        force_to_cut = None
        recorder = GeometryRecorder(env, args.backend_root)
        frame = env.current
        recorder(frame, None)
        zero = torch.zeros(1, 6, device=env.device)
        sensor_force = sensor_torque = material = 0.
        ending, steps = 'steps', 0
        for step in range(1, args.steps + 1):
            frame, reward, terminated, truncated, info = env.step(zero, reference=True)
            recorder(frame, info)
            peak = frame['peak_wrench'][0].abs()
            sensor_force = max(sensor_force, float(peak[:3].max()))
            sensor_torque = max(sensor_torque, float(peak[3:].max()))
            material = max(material, float(info['peak_force'][0]))
            steps = step
            released_now = int((frame['cut_spring_ke'][0] == 0).sum())
            if cut_step is None and released_now >= args.cut_fraction * total_springs:
                cut_step, force_to_cut = step, sensor_force
            if bool(terminated[0]) or bool(truncated[0]):
                ending = ('success' if bool(info['success'][0]) else
                          'outside' if bool(info['outside'][0]) else 'terminated')
                break
        released = int((frame['cut_spring_ke'][0] == 0).sum())
        total = int(frame['cut_spring_ke'].shape[-1])
        recorder.save(out, args.substeps)
        report = {'scene': name, 'ending': ending, 'control_steps': steps,
                  'sim_seconds': float(frame['sim_time'][0, 0]),
                  'sensor_peak_force_n': sensor_force, 'sensor_peak_torque_nm': sensor_torque,
                  'material_peak_force_n': material,
                  'cut_through_step': cut_step, 'sensor_peak_force_before_cut_n': force_to_cut,
                  'cut_fraction': args.cut_fraction, 'reference_duration_s': args.reference_duration, 'released_springs': released,
                  'total_springs': total, 'wall_seconds': time.monotonic() - started,
                  'force_limit_termination': 'disabled (measurement)', 'git': git_revision(ROOT)}
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ('scene', 'ending', 'control_steps', 'sensor_peak_force_n',
                                                  'cut_through_step', 'sensor_peak_force_before_cut_n',
                                                  'material_peak_force_n', 'released_springs', 'total_springs')}),
              flush=True)
        del env, recorder
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
