#!/usr/bin/env python3
"""Effect of lateral sawing on the cutting force (measurement, not a task episode).

Each exported scene runs prescribed knife trajectories on the full UR5e/FDCC/DiSECt
environment with the force-limit termination disabled: a constant downward speed, alone
or with a sinusoidal lateral motion along the blade (DiSECt z). Worlds differ only in
the Young's-modulus scale. Reports, per world, the released cut springs, the step of
cut-through, and mean/peak sensor and material force up to cut-through.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model.provenance import git_revision  # noqa: E402


def saw_trajectory(down_speed, distance, amplitude, frequency, dt, extra):
    """Piecewise-constant velocities: -y at ``down_speed`` for ``distance``, z = A sin(2 pi f t)."""
    from cutting_gpu_backend.reference import KnifeTrajectory
    down_time = distance / down_speed
    steps = int(round((down_time + extra) / dt))
    times = [i * dt for i in range(steps + 1)]
    velocity = []
    for i in range(steps):
        t0, t1 = times[i], times[i + 1]
        lateral = 0.
        if amplitude > 0:  # exact mean velocity of A sin(wt) over the interval
            w = 2 * math.pi * frequency
            lateral = amplitude * (math.sin(w * t1) - math.sin(w * t0)) / dt
        down = -down_speed * max(0., min(t1, down_time) - min(t0, down_time)) / dt
        velocity.append([0., down, lateral])
    return KnifeTrajectory(times, velocity, {'kind': 'saw', 'down_speed': down_speed, 'distance': distance,
                                             'amplitude': amplitude, 'frequency': frequency})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-root', type=Path, default=ROOT / 'backends/mujoco')
    parser.add_argument('--scenes', nargs='+', required=True, help='scene paths relative to the backend root')
    parser.add_argument('--young-scales', type=float, nargs='+', default=[1., 2.])
    parser.add_argument('--down-speed', type=float, default=.03)
    parser.add_argument('--saws', nargs='+', default=['0:0', '0.01:1', '0.02:1', '0.02:2'],
                        help='amplitude(m):frequency(Hz) variants; 0:0 is the pure press')
    parser.add_argument('--sim-dt', type=float, help='DiSECt substep override')
    parser.add_argument('--cut-fraction', type=float, default=.98)
    parser.add_argument('--extra-seconds', type=float, default=1.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from cutting_world_model.gpu_env import GPUEnvironment
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    n = len(args.young_scales)
    for scene in args.scenes:
        for saw in args.saws:
            amplitude, frequency = (float(x) for x in saw.split(':'))
            started = time.monotonic()
            env = GPUEnvironment(str(args.backend_root), n, scene=scene, sim_dt=args.sim_dt)
            physics = env.physics
            physics.force_limits = torch.full_like(physics.force_limits, 1e6)
            material = physics.material
            every = torch.arange(n, device=env.device)
            scales = {k: v.clone() for k, v in material.material_scales.items()}
            scales['youngs_modulus'] = torch.tensor(args.young_scales, device=env.device).to(scales['youngs_modulus'])
            material.set_material_scales(every, **scales)
            frame = env.reset(torch.ones(n, dtype=torch.bool, device=env.device))
            heads = env.heads
            distance = float((frame['observation'][0, 1] * heads.max_distance[1]).abs())
            dt = env.metadata['environment']['agent_control_dt']
            physics.set_reference_trajectory(saw_trajectory(args.down_speed, distance, amplitude, frequency,
                                                            dt, args.extra_seconds))
            steps = int(round((distance / args.down_speed + args.extra_seconds) / dt))
            total = int(frame['cut_spring_ke'].shape[-1])
            start = frame['particle_q'].double().mean(1).clone()
            series = [[] for _ in range(n)]
            alive = torch.ones(n, dtype=torch.bool)
            ending = ['steps'] * n
            zero = torch.zeros(n, 6, device=env.device)
            for step in range(1, steps + 1):
                env.needs_reset.zero_()
                frame, reward, terminated, truncated, info = env.step(zero, reference=True)
                peak = frame['peak_wrench'].abs()
                released = (frame['cut_spring_ke'] == 0).sum(-1)
                for w in range(n):
                    if alive[w]:
                        series[w].append([step, float(peak[w, :3].norm()), float(peak[w, :3].max()),
                                          float(peak[w, 3:].max()), float(info['peak_force'][w]), int(released[w])])
                        if bool(terminated[w]) or bool(truncated[w]):
                            ending[w] = ('success' if bool(info['success'][w]) else
                                         'outside' if bool(info['outside'][w]) else 'terminated')
                            alive[w] = False
            shift = (1000 * (frame['particle_q'].double().mean(1) - start)).norm(dim=-1)
            for w in range(n):
                s = series[w]
                cut = next((r[0] for r in s if r[5] >= args.cut_fraction * total), None)
                contact = next((r[0] for r in s if r[4] > .5), None)
                phase = [r for r in s if contact is not None and r[0] >= contact and (cut is None or r[0] <= cut)]
                row = {'scene': scene, 'young_scale': args.young_scales[w], 'amplitude_m': amplitude,
                       'frequency_hz': frequency, 'down_speed_mps': args.down_speed, 'ending': ending[w],
                       'released': s[-1][5], 'total_springs': total, 'cut_through_step': cut, 'contact_step': contact,
                       'mean_sensor_force_n': sum(r[1] for r in phase) / len(phase) if phase else None,
                       'peak_sensor_force_n': max((r[1] for r in phase), default=None),
                       'peak_sensor_component_n': max((r[2] for r in phase), default=None),
                       'peak_sensor_torque_nm': max((r[3] for r in phase), default=None),
                       'mean_material_force_n': sum(r[4] for r in phase) / len(phase) if phase else None,
                       'peak_material_force_n': max((r[4] for r in phase), default=None),
                       'object_centroid_shift_mm': float(shift[w]), 'series': s}
                rows.append(row)
                print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if k != 'series'}),
                      flush=True)
            del env
            torch.cuda.empty_cache()
            print('# wall', round(time.monotonic() - started), 's', flush=True)
            (args.output / 'report.json').write_text(json.dumps(
                {'git': git_revision(ROOT), 'command': sys.argv, 'rows': rows,
                 'series_columns': ['step', 'sensor_force_norm', 'sensor_force_max_component', 'sensor_torque_max',
                                    'material_force', 'released_springs'],
                 'force_limit_termination': 'disabled (measurement)'}) + '\n')


if __name__ == '__main__':
    main()
