import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from ur5e_model import action_mapping, load_model, targets
from provenance import git_revision


@wp.kernel
def update_controls(clock: wp.array(dtype=float), home: wp.array(dtype=float),
                    ctrl: wp.array2d(dtype=float)):
    world, actuator = wp.tid()
    phase = 3.141592653589793 * clock[world] + float(world) * 0.1
    value = 0.1 * wp.sin(phase + float(actuator) * 0.2)
    ctrl[world, actuator] = value


def gpu_processes():
    return subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'
    ], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worlds', type=int, default=1)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.worlds, args.steps) < 1 or args.warmup < 0:
        parser.error('worlds and steps must be positive; warmup must be nonnegative')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'git_revision': git_revision(), 'config': vars(args) | {'output': str(args.output)},
        'command': ' '.join(__import__('sys').argv), 'pid': os.getpid(),
        'versions': {p: importlib.metadata.version(p) for p in
                     ('mujoco', 'mujoco-warp', 'warp-lang', 'numpy')},
        'gpu_processes_before': gpu_processes(),
    }
    args.output.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2))
    wp.config.kernel_cache_dir = os.environ.get(
        'WARP_CACHE_PATH', str(Path(__file__).resolve().parents[1] / 'runs/cache/warp'))
    wp.init()
    model, cpu = load_model()
    report['model'] = {'nq': model.nq, 'nv': model.nv, 'nu': model.nu,
                       'nbody': model.nbody, 'ngeom': model.ngeom,
                       'timestep': model.opt.timestep,
                       'integrator': int(model.opt.integrator),
                       'action_mapping': action_mapping(model)}
    setup_start = time.perf_counter()
    with wp.ScopedDevice('cuda:0'):
        gpu_model = mjw.put_model(model)
        gpu = mjw.put_data(model, cpu, nworld=args.worlds, nconmax=64, njmax=128)
        home = wp.array(cpu.ctrl, dtype=float)

        def advance():
            wp.launch(update_controls, dim=(args.worlds, model.nu),
                      inputs=[gpu.time, home, gpu.ctrl])
            mjw.step(gpu_model, gpu)

        advance()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            advance()
        graph = capture.graph
        for _ in range(args.warmup):
            wp.capture_launch(graph)
        wp.synchronize()
        report['setup_compile_warmup_seconds'] = time.perf_counter() - setup_start
        initial_q = gpu.qpos.numpy()
        initial_time = gpu.time.numpy()
        start = time.perf_counter()
        for _ in range(args.steps):
            wp.capture_launch(graph)
        wp.synchronize()
        elapsed = time.perf_counter() - start
        final_q = gpu.qpos.numpy()
        final_qv = gpu.qvel.numpy()
        final_time = gpu.time.numpy()
        nefc = gpu.nefc.numpy()
        report.update({
            'elapsed_seconds': elapsed,
            'world_steps_per_second': args.worlds * args.steps / elapsed,
            'batch_steps_per_second': args.steps / elapsed,
            'gpu_processes_after': gpu_processes(),
            'finite': bool(np.isfinite(final_q).all() and np.isfinite(final_qv).all()),
            'max_abs_joint_motion': float(np.max(np.abs(final_q - initial_q))),
            'max_world_joint_spread': float(np.max(np.ptp(final_q, axis=0))),
            'max_constraints_at_end': int(nefc.max()),
            'time_advance_min': float((final_time - initial_time).min()),
            'time_advance_max': float((final_time - initial_time).max()),
            'world0_qpos': final_q[0].tolist(),
        })
        total_steps = int(round(float(final_time[0]) / model.opt.timestep))
        _, reference = load_model()
        for _ in range(total_steps):
            reference.ctrl[:] = targets(model, reference.time, 0)
            mujoco.mj_step(model, reference)
        report['cpu_reference_steps'] = total_steps
        report['cpu_gpu_world0_qpos_max_abs_error'] = float(np.max(np.abs(reference.qpos - final_q[0])))
        report['cpu_gpu_world0_qvel_max_abs_error'] = float(np.max(np.abs(reference.qvel - final_qv[0])))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report['finite']:
        raise RuntimeError('Nonfinite robot state')
    if not np.allclose(final_time - initial_time, args.steps * model.opt.timestep, rtol=1e-4, atol=1e-6):
        raise RuntimeError('Simulation clock did not advance by requested steps')


if __name__ == '__main__':
    main()
