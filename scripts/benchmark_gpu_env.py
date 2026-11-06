#!/usr/bin/env python3
"""Run complete reference cuts with independent completion/reset bookkeeping."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutting_world_model.provenance import git_revision
from train_coupled import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend-root', type=Path,
                   default=Path(__file__).resolve().parents[1] / 'backends/mujoco')
    p.add_argument('--worlds', type=int, required=True)
    p.add_argument('--scene', default='runs/disect_source_v4/scene.pt')
    p.add_argument('--capture', default='runs/source')
    p.add_argument('--record-episode', action='store_true', help='Save world 0 states and transitions')
    p.add_argument('--wall-seconds', type=float, default=600.)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.backend_root.resolve()))
    from cutting_world_model.gpu_env import GPUEnvironment
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    revisions = {'world_model': git_revision(Path(__file__).resolve().parents[1]),
                 'backend': git_revision(args.backend_root)}
    write_json(args.output / 'manifest.json', {
        'git': revisions, 'command': sys.argv, 'worlds': args.worlds, 'pid': os.getpid(),
        'control_mode': 'prescribed_reference_target',
        'wall_budget': args.wall_seconds, 'endpoint': 'first complete task episode per world',
        'timeout': 'write partial report; external timeout adds 60 seconds',
    })
    begin = time.perf_counter()
    env = GPUEnvironment(args.backend_root, args.worlds, scene=args.scene, capture=args.capture)
    torch.cuda.synchronize()
    setup = time.perf_counter() - begin
    done = torch.zeros(args.worlds, dtype=torch.bool, device='cuda')
    peak = torch.zeros(args.worlds, device='cuda')
    returns = torch.zeros(args.worlds, device='cuda')
    steps = torch.zeros(args.worlds, dtype=torch.long, device='cuda')
    results = {}
    physics = 0.
    reset_seconds = 0.
    particle_trace = []
    trajectory = None
    if args.record_episode:
        trajectory = {key: [value[0].cpu().numpy().copy()] for key, value in env.current.items()}
        transitions = {key: [] for key in ('reference_target_pose', 'rewards', 'terminated', 'truncated', 'success', 'material_force')}
    reason = 'budget'
    while time.perf_counter() - begin < args.wall_seconds:
        tick = time.perf_counter()
        command = env.physics.reference_target(env.physics.sim_time + env.environment['agent_control_dt'])
        frame, reward, term, trunc, info = env.step(torch.zeros(args.worlds, 6, device='cuda'), reference=True)
        torch.cuda.synchronize()
        physics += time.perf_counter() - tick
        particle_trace.append(frame['particle_q'][0].cpu())
        if trajectory is not None and not bool(done[0]):
            for key, value in frame.items():
                trajectory[key].append(value[0].cpu().numpy().copy())
            transitions['reference_target_pose'].append(command[0].cpu().numpy().copy())
            for key, value in (('rewards', reward), ('terminated', term), ('truncated', trunc),
                               ('success', info['success']), ('material_force', info['peak_force'])):
                transitions[key].append(value[0].item())
        peak = torch.maximum(peak, info['peak_force'])
        returns += torch.where(done, 0., reward)
        steps += ~done
        ended = term | trunc
        for i in (ended & ~done).nonzero().flatten().tolist():
            results[str(i)] = {
                'success': bool(info['success'][i]), 'collision': bool(info['collision'][i]),
                'truncated': bool(trunc[i]), 'steps': int(steps[i]), 'return': float(returns[i]),
                'released_springs': int(info['released_springs'][i]), 'peak_force': float(peak[i]),
                'vertical_error_m': float((frame['target_pose'][i, 2] - frame['ee_pose'][i, 2]).abs()),
            }
        done |= ended
        status = {'worlds': args.worlds, 'finished': int(done.sum()), 'step': int(steps.max()),
                  'wall_seconds': time.perf_counter() - begin, 'stepping_seconds': physics}
        write_json(args.output / 'progress.json', status)
        print(json.dumps(status), flush=True)
        if done.all():
            reason = 'completed'
            break
        if ended.any():
            tick = time.perf_counter()
            env.reset(ended)
            torch.cuda.synchronize()
            reset_seconds += time.perf_counter() - tick
    torch.save(torch.stack(particle_trace), args.output / 'particles_world0.pt')
    if trajectory is not None:
        np.savez_compressed(args.output / 'episode_world0.npz',
                            **{key: np.asarray(value) for key, value in trajectory.items()},
                            **{key: np.asarray(value) for key, value in transitions.items()})
    write_json(args.output / 'report.json', {
        'git': revisions, 'reason': reason, 'worlds': args.worlds, 'episodes': results,
        'setup_seconds': setup, 'stepping_seconds': physics, 'reset_seconds': reset_seconds,
        'wall_seconds': time.perf_counter() - begin,
        'world_steps_per_second': int(steps.sum()) / physics,
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
    })
    if reason != 'completed' or not all(row['success'] for row in results.values()):
        raise RuntimeError('Complete cutting reference task failed; inspect report.json')


if __name__ == '__main__':
    main()
