"""Check original DiSECt backward gradients against central finite differences.

One object config and one bottom rule per process. The objective is the core of the DiSECt
trajectory optimization: mean knife force over a slicing trajectory, as a function of the
SlicingMotion parameters (frequency, 5 amplitudes, 5 pressing velocities). The finite difference
uses one fixed random direction in relative parameter coordinates at each requested step size.
Run from the repository so the recorded git revision is this repository's.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from disect_headless import import_disect_cutting
from provenance import git_revision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disect-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--bottom', choices=('free', 'fixed'), required=True,
                        help='free: static include box y < -0.002; fixed: y < +0.002 (bottom layer static)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=0.6)
    parser.add_argument('--eps', type=float, nargs='+', default=[1e-2, 1e-3])
    parser.add_argument('--memory-fraction', type=float, default=0.45)
    args = parser.parse_args()
    args.output = args.output.resolve()
    config = args.config.resolve()
    revision = git_revision()
    disect_root = args.disect_root.resolve()
    os.chdir(disect_root)  # configs name meshes relative to the DiSECt root
    cutting = import_disect_cutting(disect_root)
    CuttingSim, SlicingMotion, load_settings = cutting.CuttingSim, cutting.SlicingMotion, cutting.load_settings

    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    settings = load_settings(str(config))
    settings.sim_duration = args.duration
    settings.sim_dt = 1e-5
    settings.sim_substeps = 500
    configured_max_y = settings.static_vertices.include[0]['max'][1]
    settings.static_vertices.include[0]['max'][1] = -0.002 if args.bottom == 'free' else 0.002
    row = {'git_revision': revision, 'command': sys.argv, 'config': str(config), 'bottom': args.bottom, 'configured_include_max_y': configured_max_y,
           'include_max_y': settings.static_vertices.include[0]['max'][1], 'duration': args.duration}

    sim = CuttingSim(settings, experiment_name='gradcheck', adapter='cuda', requires_grad=True,
                     allow_nans=True, log_folder=str(args.output))
    amplitudes = torch.full((5,), 0.02, device='cuda', requires_grad=True)
    pressing = torch.full((5,), -0.05, device='cuda', requires_grad=True)
    frequency = torch.tensor(5., device='cuda', requires_grad=True)
    parameters = (frequency, amplitudes, pressing)
    sim.motion = SlicingMotion(initial_pos=torch.tensor([0., settings.initial_y, 0.], device='cuda'),
                               slicing_frequency=frequency, slicing_amplitudes=amplitudes,
                               pressing_velocities=pressing, slicing_times=np.linspace(0., args.duration, 5),
                               slicing_kernel_width=0.02)
    sim.cut()
    q = np.asarray(sim.builder.particle_q)
    mass = np.asarray(sim.builder.particle_mass)
    static = mass == 0
    row.update(particles=len(q), static=int(static.sum()), y_min=float(q[:, 1].min()), y_max=float(q[:, 1].max()),
               static_y_max=float(q[static, 1].max()) if static.any() else None)

    last = {}
    original_step = sim.simulation_step

    def recorded_step():
        original_step()
        if sim.sim_step % sim.sim_substeps == 0:
            last['ke'] = sim.state.cut_spring_ke.detach()
            last['q_finite'] = bool(torch.isfinite(sim.state.particle_q).all())

    sim.simulation_step = recorded_step

    def objective():
        forces = sim.simulate(show_progressbar=False)
        return forces.mean(), forces

    tick = time.monotonic()
    value, forces = objective()
    row['forward_seconds'] = time.monotonic() - tick
    row.update(force_mean=float(value.detach()), force_peak=float(forces.detach().max()),
               force_finite=bool(torch.isfinite(forces).all()),
               springs=int(last['ke'].numel()), released_springs=int((last['ke'] == 0).sum()),
               particle_q_finite=last['q_finite'])
    tick = time.monotonic()
    value.backward()
    row['backward_seconds'] = time.monotonic() - tick
    grads = [p.grad.detach().clone() for p in parameters]
    row['gradient'] = [g.cpu().tolist() for g in grads]
    row['gradient_finite'] = bool(all(torch.isfinite(g).all() for g in grads))

    # Direction in relative parameter coordinates: delta_i = eps * |p_i| * d_i, d a fixed random unit vector.
    generator = torch.Generator().manual_seed(0)
    flat = torch.cat([p.detach().reshape(-1) for p in parameters]).cpu().double()
    d = torch.randn(flat.numel(), generator=generator, dtype=torch.float64)
    d /= d.norm()
    step = flat.abs() * d
    g = torch.cat([x.reshape(-1) for x in grads]).cpu().double()
    row['directional_analytic'] = float(g @ step)

    def set_flat(values):
        with torch.no_grad():
            offset = 0
            for p in parameters:
                n = p.numel()
                p.copy_(values[offset:offset + n].reshape(p.shape).to(p.dtype))
                offset += n

    row['finite_difference'] = []
    for eps in args.eps:
        out = {}
        for sign in (1, -1):
            set_flat(flat + sign * eps * step)
            with torch.no_grad():
                out[sign] = float(objective()[0])
        set_flat(flat)
        fd = (out[1] - out[-1]) / (2 * eps)
        row['finite_difference'].append({'eps': eps, 'plus': out[1], 'minus': out[-1], 'directional': fd,
                                         'ratio_analytic_over_fd': row['directional_analytic'] / fd if fd else None})
    row['peak_memory_bytes'] = torch.cuda.max_memory_allocated()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'gradcheck.json').write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
