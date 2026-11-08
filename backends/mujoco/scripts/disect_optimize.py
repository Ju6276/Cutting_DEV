"""Run the upstream DiSECt/MDMM slicing-trajectory optimization with recorded physical states.

Copied from DiSECt-Optimization/optimize.py (run on GPU9 before 2026-11-07). Changes: headless
DiSECt import, this repository's git revision, and --bottom to override the static include box
(the original adjoint is NaN with a free bottom; RESULTS 2026-11-07). Output (manifest.json,
last.pt, iterations.jsonl) is read by scripts/export_disect_trajectory.py.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import mdmm

from disect_headless import import_disect_cutting


def revision(path):
    return {
        'commit': subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip(),
        'dirty': bool(subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True)),
    }


BOTTOM_MAX_Y = {'free': -0.002, 'fixed': 0.002}
SCRIPT_DIR = Path(__file__).resolve().parent  # resolved before main() changes directory


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disect-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--wall-seconds', type=float, default=14400)
    parser.add_argument('--memory-fraction', type=float, default=0.25)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--bottom', choices=('config', 'free', 'fixed'), default='config',
                        help='config keeps the static include box; free: y < -0.002; fixed: y < 0.002')
    args = parser.parse_args()
    args.disect_root = args.disect_root.resolve()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(args.disect_root)  # configs name meshes relative to the DiSECt root
    cutting = import_disect_cutting(args.disect_root)
    CuttingSim, SlicingMotion, load_settings = cutting.CuttingSim, cutting.SlicingMotion, cutting.load_settings

    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    settings = load_settings(str(args.config))
    settings.sim_duration = 0.6
    settings.sim_dt = 1e-5
    settings.sim_substeps = 500
    if args.bottom != 'config':
        settings.static_vertices.include[0]['max'][1] = BOTTOM_MAX_Y[args.bottom]
    start_time = time.monotonic()
    manifest = {'git': revision(SCRIPT_DIR), 'bottom': args.bottom, 'disect_git': revision(args.disect_root),
                'command': sys.argv, 'config': settings, 'iterations': args.iterations,
                'wall_seconds': args.wall_seconds, 'memory_fraction': args.memory_fraction,
                'optimizer': {'method': 'MDMM', 'lr': 0.01, 'scale': 10000., 'damping': 100.},
                'motion': {'waypoints': 5, 'kernel_width': 0.02, 'initial_frequency': 5.,
                           'initial_amplitudes': 0.02, 'initial_pressing_velocities': -0.05}}
    write_json(args.output / 'manifest.json', manifest)
    sim = CuttingSim(settings, experiment_name='prism_optimization', adapter='cuda',
                     requires_grad=True, allow_nans=False, log_folder=str(args.output))
    amplitudes = torch.full((5,), 0.02, device='cuda', requires_grad=True)
    pressing = torch.full((5,), -0.05, device='cuda', requires_grad=True)
    frequency = torch.tensor(5., device='cuda', requires_grad=True)
    parameters = (frequency, amplitudes, pressing)
    sim.motion = SlicingMotion(initial_pos=torch.tensor([0., settings.initial_y, 0.], device='cuda'),
                               slicing_frequency=frequency, slicing_amplitudes=amplitudes,
                               pressing_velocities=pressing, slicing_times=np.linspace(0., 0.6, 5),
                               slicing_kernel_width=0.02)
    blade_length = sim.knife.depth
    initial_particles = np.asarray(sim.builder.particle_q)
    blade_min = initial_particles[:, 2].max() - blade_length / 2
    blade_max = initial_particles[:, 2].min() + blade_length / 2
    height = mdmm.EqConstraint(lambda: sim.hist_knife_pos[-1, 1] + sim.blade_bottom_center_point_offset[1],
                               0., scale=10000., damping=100.)
    lateral = mdmm.EqConstraint(lambda: (sim.hist_knife_pos[:, 2].clamp(blade_min, blade_max)
                                        - sim.hist_knife_pos[:, 2]).sum(),
                                0., scale=10000., damping=100.)
    sim.cut()
    manifest['static_particles'] = int((np.asarray(sim.builder.particle_mass) == 0).sum())
    write_json(args.output / 'manifest.json', manifest)
    model = sim.model
    np.savez_compressed(args.output / 'topology.npz',
                        **{name: getattr(model, name).detach().cpu().numpy()
                           for name in ('tri_indices', 'cut_tri_indices', 'cut_virtual_tri_indices',
                                        'cut_edge_indices', 'cut_edge_coords')},
                        knife_vertices=sim.knife_vertices, knife_tri_indices=sim.knife_faces)
    constraints = mdmm.MDMM([height, lateral])
    optimizer = constraints.make_optimizer(parameters, lr=0.01)
    begin = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda', weights_only=False)
        with torch.no_grad():
            for current, saved in zip(parameters, checkpoint['parameters']):
                current.copy_(saved)
        constraints.load_state_dict(checkpoint['constraints'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        begin = checkpoint['next_iteration']
    frames = []
    original_step = sim.simulation_step
    iteration = begin

    def recorded_step():
        original_step()
        if sim.sim_step % sim.sim_substeps == 0:
            state = sim.state
            frames.append((sim.sim_time, state.particle_q.detach().cpu().numpy().copy(),
                           state.body_X_sm[sim.model.knife_link_index].detach().cpu().numpy().copy(),
                           state.cut_spring_ke.detach().cpu().numpy().copy()))
            if time.monotonic() - start_time > args.wall_seconds:
                raise TimeoutError('Experiment wall budget reached')
            if sim.sim_step % 5000 == 0:
                write_json(args.output / 'progress.json', {'iteration': iteration, 'phase': 'forward',
                           'substep': sim.sim_step, 'total_substeps': sim.sim_steps,
                           'elapsed_seconds': time.monotonic() - start_time})

    sim.simulation_step = recorded_step
    for iteration in range(begin, args.iterations + 1):
        frames.clear()
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        optimizer.zero_grad()
        forces = sim.simulate(show_progressbar=False)
        mean_force = forces.mean()
        result = constraints(mean_force)
        z = sim.hist_knife_pos[:, 2].detach()
        row = {'iteration': iteration, 'force_mean': float(mean_force.detach()),
               'force_peak': float(forces.detach().max()), 'end_blade_height': float(height.fn().detach()),
               'lower_blade_violation': float((blade_min - z).clamp_min(0).max()),
               'upper_blade_violation': float((z - blade_max).clamp_min(0).max()),
               'released_springs': int((frames[-1][3] == 0).sum()),
               'spring_count': len(frames[-1][3]), 'forward_seconds': time.monotonic() - tick,
               'parameters': [p.detach().cpu().tolist() for p in parameters]}
        if not torch.isfinite(result.value):
            raise FloatingPointError('Nonfinite optimization objective')
        np.savez_compressed(args.output / 'last_trajectory.npz',
                            time=np.array([f[0] for f in frames]), particle_q=np.stack([f[1] for f in frames]),
                            knife_pose=np.stack([f[2] for f in frames]), cut_spring_ke=np.stack([f[3] for f in frames]),
                            force=forces.detach().cpu().numpy())
        if iteration in (0, args.iterations):
            np.savez_compressed(args.output / ('initial_trajectory.npz' if iteration == 0 else 'optimized_trajectory.npz'),
                                **dict(np.load(args.output / 'last_trajectory.npz')))
        if iteration < args.iterations:
            write_json(args.output / 'progress.json', {'iteration': iteration, 'phase': 'backward',
                       'elapsed_seconds': time.monotonic() - start_time})
            tick = time.monotonic()
            result.value.backward()
            row['backward_seconds'] = time.monotonic() - tick
            row['gradients'] = [p.grad.detach().cpu().tolist() for p in parameters]
            if any(not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError('Nonfinite trajectory gradient')
            optimizer.step()
        row['peak_memory_bytes'] = torch.cuda.max_memory_allocated()
        row['elapsed_seconds'] = time.monotonic() - start_time
        with (args.output / 'iterations.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)
        torch.save({'parameters': [p.detach() for p in parameters], 'constraints': constraints.state_dict(),
                    'optimizer': optimizer.state_dict(), 'next_iteration': iteration + 1}, args.output / 'last.pt')
        write_json(args.output / 'progress.json', {'iteration': iteration, 'phase': 'complete' if iteration == args.iterations else 'updated',
                   'elapsed_seconds': time.monotonic() - start_time})


if __name__ == '__main__':
    main()
