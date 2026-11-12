#!/usr/bin/env python3
"""Record the geometry of a prescribed cut (press, or press plus lateral sawing) for rendering.

Runs one world with the reference branch (no policy), the force-limit termination disabled,
and writes ``episode.npz``, ``report.json`` and ``task.json`` in the format read by
``render_episode.py``. A measurement/illustration tool, not a task episode.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco'), str(ROOT / 'scripts')]
from cutting_world_model.provenance import git_revision  # noqa: E402
from measure_sawing import saw_trajectory  # noqa: E402
from record_episode import GeometryRecorder  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-root', type=Path, default=ROOT / 'backends/mujoco')
    parser.add_argument('--scene', required=True, help='scene path relative to the backend root')
    parser.add_argument('--label', required=True)
    parser.add_argument('--down-speed', type=float, default=.03)
    parser.add_argument('--saw', default='0:0', help='amplitude(m):frequency(Hz); 0:0 is the pure press')
    parser.add_argument('--force-model', choices=('scalar', 'net'), default='scalar')
    parser.add_argument('--friction-velocity', type=float)
    parser.add_argument('--sim-dt', type=float)
    parser.add_argument('--extra-seconds', type=float, default=.5)
    parser.add_argument('--substeps', type=int, default=12, help='display samples per control interval')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from cutting_world_model.gpu_env import GPUEnvironment
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    amplitude, frequency = (float(x) for x in args.saw.split(':'))
    env = GPUEnvironment(str(args.backend_root), 1, scene=args.scene, sim_dt=args.sim_dt,
                         force_model=args.force_model, friction_velocity=args.friction_velocity)
    physics = env.physics
    physics.force_limits = torch.full_like(physics.force_limits, 1e6)
    frame = env.current
    distance = float((frame['observation'][0, 1] * env.heads.max_distance[1]).abs())
    dt = env.metadata['environment']['agent_control_dt']
    physics.set_reference_trajectory(saw_trajectory(args.down_speed, distance, amplitude, frequency, dt,
                                                    args.extra_seconds))
    steps = int(round((distance / args.down_speed + args.extra_seconds) / dt))
    (args.output / 'task.json').write_text(json.dumps(env.metadata) + '\n')
    recorder = GeometryRecorder(env, args.backend_root)
    recorder(frame, None)
    zero = torch.zeros(1, 6, device=env.device)
    total = int(frame['cut_spring_ke'].shape[-1])
    peak, vertical, done = 0., [], 0
    for step in range(1, steps + 1):
        env.needs_reset.zero_()
        frame, reward, terminated, truncated, info = env.step(zero, reference=True)
        recorder(frame, info)
        peak = max(peak, float(info['peak_force'][0]))
        vertical.append(float(physics.material.knife_wrench()[0, 1]))
        done = step
        if bool(terminated[0]) or bool(truncated[0]):
            break
    released = int((frame['cut_spring_ke'][0] == 0).sum())
    report = {'label': args.label, 'outcome': f'{released}/{total} cut springs released',
              'method': 'reference', 'scene': args.scene, 'saw': args.saw, 'down_speed_mps': args.down_speed,
              'force_model': args.force_model, 'friction_velocity': args.friction_velocity,
              'control_steps': done, 'simulated_seconds': done * dt, 'max_material_force_n': peak,
              'max_net_vertical_n': max(vertical), 'released_springs': released, 'total_springs': total,
              'display_frames': recorder.save(args.output, args.substeps),
              'substeps_per_control_step': args.substeps, 'force_limit_termination': 'disabled',
              'git': git_revision(ROOT), 'command': sys.argv, 'total_wall_seconds': time.monotonic() - started}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('git', 'command')}), flush=True)


if __name__ == '__main__':
    main()
