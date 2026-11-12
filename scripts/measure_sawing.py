#!/usr/bin/env python3
"""Effect of lateral sawing on the cutting force (measurement, not a task episode).

Each exported scene runs prescribed knife trajectories on the full UR5e/FDCC/DiSECt
environment with the force-limit termination disabled: a constant downward speed, alone
or with a sinusoidal lateral motion along the blade (DiSECt z). Worlds differ only in
the Young's-modulus scale. Reports, per world, the released cut springs, the step of
cut-through, and simultaneous material/sensor forces sampled at 500 Hz up to cut-through.
All component and norm statistics use the same contact-to-release samples. Results
are specific to the measured scenes, parameters and trajectories.
"""
import argparse
import hashlib
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



SAMPLE_COLUMNS = ['time_s', 'step', 'material_fx', 'material_fy', 'material_fz',
                  'sensor_fx', 'sensor_fy', 'sensor_fz', 'sensor_tx', 'sensor_ty', 'sensor_tz',
                  'contact_force_proxy', 'released_springs']


def summarize_samples(series, total, cut_fraction=.98, contact_threshold=.5, sample_dt=.002):
    """Same 500 Hz samples for every mean/peak; release fraction is diagnostic only."""
    contact = next((i for i, r in enumerate(series) if r[11] > contact_threshold), None)
    cut = next((i for i, r in enumerate(series) if r[12] >= cut_fraction * total), None)
    result = {
        'contact_step': None if contact is None else int(series[contact][1]),
        'contact_time_s': None if contact is None else series[contact][0],
        'cut_through_step': None if cut is None else int(series[cut][1]),
        'cut_through_time_s': None if cut is None else series[cut][0],
    }
    phase = [] if contact is None else series[contact:(None if cut is None else cut + 1)]
    result['phase_samples'] = len(phase)
    result['phase_duration_s'] = len(phase) * sample_dt
    if not phase:
        return result
    data = torch.tensor(phase, dtype=torch.float64)
    material_norm = data[:, 2:5].norm(dim=-1)
    sensor_norm = data[:, 5:8].norm(dim=-1)
    result.update(
        mean_material_force_n=float(material_norm.mean()),
        peak_material_force_n=float(material_norm.max()),
        material_impulse_ns=float(material_norm.sum() * sample_dt),
        mean_net_vertical_n=float(data[:, 3].mean()),
        peak_net_vertical_n=float(data[:, 3].max()),
        mean_abs_net_along_blade_n=float(data[:, 4].abs().mean()),
        mean_sensor_force_n=float(sensor_norm.mean()),
        peak_sensor_force_n=float(sensor_norm.max()),
        peak_sensor_component_n=float(data[:, 5:8].abs().max()),
        peak_sensor_torque_nm=float(data[:, 8:11].abs().max()),
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-root', type=Path, default=ROOT / 'backends/mujoco')
    parser.add_argument('--scenes', nargs='+', required=True, help='scene paths relative to the backend root')
    parser.add_argument('--young-scales', type=float, nargs='+', default=[1., 2.])
    parser.add_argument('--down-speed', type=float, default=.03)
    parser.add_argument('--saws', nargs='+', default=['0:0', '0.01:1', '0.02:1', '0.02:2'],
                        help='amplitude(m):frequency(Hz) variants; 0:0 is the pure press')
    parser.add_argument('--sim-dt', type=float, help='DiSECt substep override')
    parser.add_argument('--force-model', choices=('scalar', 'net'), default='scalar')
    parser.add_argument('--friction-velocity', type=float, help='knife friction transition velocity (m/s)')
    parser.add_argument('--cut-fraction', type=float, default=.98)
    parser.add_argument('--contact-force-threshold', type=float, default=.5,
                        help='phase start from sum of contact magnitudes (N); diagnostic, not task success')
    parser.add_argument('--extra-seconds', type=float, default=1.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.cut_fraction <= 1 or args.down_speed <= 0 or args.contact_force_threshold < 0:
        parser.error('invalid release fraction, speed or contact threshold')
    from cutting_world_model.gpu_env import GPUEnvironment
    args.output.mkdir(parents=True, exist_ok=True)
    source_files = ['scripts/measure_sawing.py', 'cutting_world_model/gpu_env.py',
                    'backends/mujoco/cutting_gpu_backend/environment.py',
                    'backends/mujoco/cutting_gpu_backend/coupling.py',
                    'backends/mujoco/cutting_gpu_backend/batched_disect.py',
                    'backends/mujoco/cutting_gpu_backend/metrics.py',
                    'backends/mujoco/cutting_gpu_backend/disect.py',
                    'backends/mujoco/third_party/disect/disect/dflex/sim.py']
    source_hashes = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in source_files}
    scene_hashes = {f: hashlib.sha256((args.backend_root / f).read_bytes()).hexdigest() for f in args.scenes}
    rows = []
    n = len(args.young_scales)
    for scene in args.scenes:
        for saw in args.saws:
            amplitude, frequency = (float(x) for x in saw.split(':'))
            started = time.monotonic()
            env = GPUEnvironment(str(args.backend_root), n, scene=scene, sim_dt=args.sim_dt,
                                 force_model=args.force_model, friction_velocity=args.friction_velocity)
            physics = env.physics
            physics.force_limits = torch.full_like(physics.force_limits, 1e6)
            material = physics.material
            every = torch.ones(n, device=env.device, dtype=torch.bool)
            scales = {k: v.clone() for k, v in material.material_scales.items()}
            scales['youngs_modulus'] = torch.tensor(args.young_scales, device=env.device).to(scales['youngs_modulus'])
            material.set_material_scales(every, **scales)
            frame = env.reset(every)
            distance = float((frame['observation'][0, 1] * env.heads.max_distance[1]).abs())
            dt = env.metadata['environment']['agent_control_dt']
            physics.set_reference_trajectory(saw_trajectory(args.down_speed, distance, amplitude, frequency,
                                                            dt, args.extra_seconds))
            steps = int(round((distance / args.down_speed + args.extra_seconds) / dt))
            total = int(frame['cut_spring_ke'].shape[-1])
            start = frame['particle_q'].double().mean(1).clone()
            shift = torch.zeros(n, device=env.device)
            series = [[] for _ in range(n)]
            alive = torch.ones(n, dtype=torch.bool, device=env.device)
            ending = ['steps'] * n
            zero = torch.zeros(n, 6, device=env.device)

            def sample(p, active):
                # Clone/copy immediately: physics tensors are reused on the next substep.
                net = p.material.knife_wrench()
                sensor = p.feedback['sensor_wrench']
                released = (p.material.state_field('cut_spring_ke') == 0).sum(-1)
                values = torch.cat((p.sim_time[:, None], net[:, :3], sensor,
                                    p.feedback['force_proxy'][:, None], released[:, None]), -1).cpu().tolist()
                for w in torch.nonzero(alive & active).flatten().tolist():
                    series[w].append([values[w][0], step, *values[w][1:]])

            for step in range(1, steps + 1):
                env.needs_reset.zero_()
                frame, reward, terminated, truncated, info = env.step(zero, reference=True, sample_callback=sample)
                displacement = (1000 * (frame['particle_q'].double().mean(1) - start)).norm(dim=-1)
                shift[alive] = displacement[alive].to(shift)
                for w in torch.nonzero(alive & (terminated | truncated)).flatten().tolist():
                    ending[w] = ('success' if bool(info['success'][w]) else
                                 'outside' if bool(info['outside'][w]) else 'terminated')
                alive &= ~(terminated | truncated)
                if not bool(alive.any()):
                    break
            for w in range(n):
                samples = series[w]
                row = {'scene': scene, 'young_scale': args.young_scales[w], 'amplitude_m': amplitude,
                       'frequency_hz': frequency, 'down_speed_mps': args.down_speed, 'ending': ending[w],
                       'released': int(samples[-1][12]) if samples else 0, 'total_springs': total,
                       **summarize_samples(samples, total, args.cut_fraction, args.contact_force_threshold),
                       'force_model': args.force_model, 'friction_velocity': args.friction_velocity,
                       'model_metadata': env.metadata.get('material_contact'),
                       'contact_discretization_revision': env.metadata['contact_discretization_revision'],
                       'object_centroid_shift_mm': float(shift[w]), 'series': samples}
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != 'series'}), flush=True)
            del env
            torch.cuda.empty_cache()
            print('# wall', round(time.monotonic() - started), 's', flush=True)
            (args.output / 'report.json').write_text(json.dumps(
                {'git': git_revision(ROOT), 'command': sys.argv, 'rows': rows,
                 'source_sha256': source_hashes, 'scene_sha256': scene_hashes,
                 'schema': 'simultaneous_coupling_samples_v2', 'sample_dt_s': .002,
                 'series_columns': SAMPLE_COLUMNS,
                 'force_limit_termination': 'disabled (measurement)',
                 'material_frame': 'DiSECt axes (y up, z along blade); reaction on knife',
                 'sensor_frame': 'filtered sensor axes; moment about sensor origin',
                 'force_summary_definition': 'Simultaneous vectors at 500 Hz, common contact-to-release window; '
                                             'arithmetic means of uniformly spaced samples, not means of step peaks. '
                                             '98% release is a diagnostic default, not validated task success.'}) + '\n')


if __name__ == '__main__':
    main()
