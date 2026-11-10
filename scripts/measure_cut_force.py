#!/usr/bin/env python3
"""Force needed to cut each prepared object (multi-condition task, S0).

Each exported scene (runs/sweep/<name>) runs a constant-velocity knife reference on the
full UR5e/FDCC/DiSECt environment, starting ``--clearance`` above the object top, with
the force-limit termination disabled. Reports the largest sensor force before the cut
is through (``--cut-fraction`` of cut springs released). Used to choose the one task
force limit; it is a measurement, not a task episode.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model.provenance import git_revision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-root', type=Path, default=ROOT / 'backends/mujoco')
    parser.add_argument('--scenes', nargs='+', required=True, help='names under runs/sweep')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--reference-duration', type=float,
                        help='keep the scene reference velocity for this long (s); default: scene duration')
    parser.add_argument('--clearance', type=float,
                        help='start with the knife edge this far above the object top (m), by IK')
    parser.add_argument('--reference-speed', type=float,
                        help='downward reference speed (m/s); default: the scene velocity')
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
        lift = None
        if args.clearance is not None:
            from cutting_world_model.multi_env import knife_clearance
            lift = torch.tensor([args.clearance - knife_clearance(args.backend_root, f'runs/sweep/{name}/scene.pt')])
        env = GPUEnvironment(str(args.backend_root), 1, scene=f'runs/sweep/{name}/scene.pt', initial_lift=lift)
        env.physics.force_limits = torch.full_like(env.physics.force_limits, 1e6)
        if args.reference_duration is not None or args.reference_speed is not None:
            from cutting_gpu_backend.reference import KnifeTrajectory
            physics = env.physics
            velocity = physics.material.reference_linear_velocity.reshape(-1, 3)[0].to(physics.dtype)
            if args.reference_speed is not None:
                velocity = velocity / velocity.norm() * args.reference_speed
            duration = args.reference_duration or float(physics.material.reference_duration)
            physics.set_reference_trajectory(KnifeTrajectory.constant(velocity, duration,
                                                                      kind='sweep_constant_velocity'))
        total_springs = int(env.current['cut_spring_ke'].shape[-1])
        cut_step = None
        force_to_cut = None
        frame = env.current
        start = frame['particle_q'][0].double().clone()
        bottom = start[:, 1] < start[:, 1].min() + .003
        zero = torch.zeros(1, 6, device=env.device)
        sensor_force = sensor_torque = material = 0.
        ending, steps = 'steps', 0
        for step in range(1, args.steps + 1):
            frame, reward, terminated, truncated, info = env.step(zero, reference=True)
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
        end = frame['particle_q'][0].double()
        centroid_shift_mm = (1000 * (end.mean(0) - start.mean(0))).tolist()
        bottom_shift_mm = float(1000 * (end[bottom] - start[bottom]).norm(dim=-1).max())
        released = int((frame['cut_spring_ke'][0] == 0).sum())
        total = int(frame['cut_spring_ke'].shape[-1])
        report = {'scene': name, 'ending': ending, 'control_steps': steps,
                  'sim_seconds': float(frame['sim_time'][0, 0]),
                  'sensor_peak_force_n': sensor_force, 'sensor_peak_torque_nm': sensor_torque,
                  'material_peak_force_n': material,
                  'object_centroid_shift_mm': centroid_shift_mm, 'object_bottom_shift_max_mm': bottom_shift_mm,
                  'cut_through_step': cut_step, 'sensor_peak_force_before_cut_n': force_to_cut,
                  'cut_fraction': args.cut_fraction, 'reference_duration_s': args.reference_duration,
                  'reference_speed_mps': args.reference_speed, 'clearance_m': args.clearance, 'released_springs': released,
                  'total_springs': total, 'wall_seconds': time.monotonic() - started,
                  'force_limit_termination': 'disabled (measurement)', 'git': git_revision(ROOT)}
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ('scene', 'ending', 'control_steps', 'sensor_peak_force_n',
                                                  'cut_through_step', 'sensor_peak_force_before_cut_n',
                                                  'material_peak_force_n', 'released_springs', 'total_springs')}),
              flush=True)
        del env
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
