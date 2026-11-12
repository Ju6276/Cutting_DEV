#!/usr/bin/env python3
"""Bounded physical reference checks for free_cut_v1; no learner or optimizer.

Writes the exact provisional reward config, per-step returns and terminal causes.
This checks reward/termination plumbing, not algorithm performance or real forces.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-root', type=Path, default=ROOT / 'backends/mujoco')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=float, default=1200.)
    parser.add_argument('--cases', nargs='+', default=['press', 'saw', 'fast', 'stall', 'retract', 'slow', 'force_attempt', 'slow_press', 'hold_in_material'])
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg['task'].get('version') == 'free_cut_v2' and cfg['gamma'] != cfg['task']['gamma']:
        raise ValueError('Evaluation discount must equal task discount')
    if args.output.exists():
        raise ValueError('Use a new output directory')
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.backend_root))
    from cutting_world_model.gpu_env import GPUEnvironment
    from cutting_gpu_backend.reference import KnifeTrajectory

    (args.output / 'working.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT))
    (args.output / 'check_script.py').write_text(Path(__file__).read_text())
    start = time.monotonic()
    results = []
    report = dict(config=cfg, config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  working_diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT)).hexdigest(),
                  purpose='physical task/reward interface check; provisional weights; no training',
                  results=results, incomplete=[], wall_seconds=args.wall_seconds)
    def save():
        report['elapsed_seconds'] = time.monotonic() - start
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    env = GPUEnvironment(str(args.backend_root), 1, scene=cfg['scene'], task=cfg['task'],
                         force_limits=cfg['force_limits'], force_model='net',
                         friction_velocity=cfg['friction_velocity'])
    env.target_speed = torch.tensor([cfg['target_speed']], device=env.device)
    every = torch.ones(1, dtype=torch.bool, device=env.device)
    scales = {k: torch.tensor([cfg['material'].get(k, 1.)], device=env.device)
              for k in env.physics.material.material_scales}
    env.physics.material.set_material_scales(every, **scales)
    frame = env.settle(cfg['settle_seconds'])

    for case in args.cases:
        if case not in ('press', 'saw', 'fast', 'stall', 'retract', 'slow', 'force_attempt', 'slow_press', 'hold_in_material'):
            raise ValueError('Unknown reference case: ' + case)
        if time.monotonic() - start >= args.wall_seconds:
            report['incomplete'].append(case)
            continue
        frame = env.reset()
        dt = env.environment['agent_control_dt']
        distance = float((frame['observation'][0, 1] * env.heads.max_distance[1]).abs())
        nominal_distance = distance
        speed = cfg['target_speed'] * (3 if case == 'fast' else 20 if case == 'force_attempt' else .25 if case == 'slow' else 1)
        if case in ('force_attempt', 'slow_press'):
            distance *= 2  # Continue pressing; workspace/force events still take precedence.
        horizon = cfg['task']['horizon_steps']
        times, velocities = [0.], []
        for i in range(horizon):
            t = i * dt
            down = speed if t < distance / speed else 0.
            if case == 'slow_press' and t >= .5 * nominal_distance / speed:
                down = speed * .25
            if case == 'hold_in_material' and t >= .5 * nominal_distance / speed:
                down = 0.
            if case == 'stall':
                down = 0.
            if case == 'retract' and t > .5 * distance / speed:
                down = -speed if t < distance / speed else 0.
            lateral = (cfg['saw_amplitude'] * (math.sin(2 * math.pi * cfg['saw_frequency'] * (t + dt))
                       - math.sin(2 * math.pi * cfg['saw_frequency'] * t)) / dt
                       if case == 'saw' and down else 0.)
            times.append((i + 1) * dt)
            velocities.append([0., -down, lateral])
        env.physics.set_reference_trajectory(KnifeTrajectory(times, velocities, {'case': case}))
        rows, peak_net = [], 0.
        def sample(physics, active):
            nonlocal peak_net
            if bool(active[0]):
                peak_net = max(peak_net, float(physics.material.knife_wrench()[0, :3].norm()))
        total = 0.
        outcome = 'wall_budget'
        for i in range(horizon):
            if time.monotonic() - start >= args.wall_seconds:
                report['incomplete'].append(case)
                break
            frame, reward, term, trunc, info = env.step(torch.zeros(1, 6, device=env.device),
                                                       reference=True, sample_callback=sample)
            total += cfg['gamma'] ** i * float(reward[0])
            rows.append(dict(step=i + 1, prescribed_velocity_disect=velocities[i], reward=float(reward[0]), progress=float(frame['cut_progress'][0, 0]),
                             peak_wrench=frame['peak_wrench'][0].tolist(),
                             speed_mps=float(env.heads.cutting_speed(frame['observation'])[0]),
                             jerk_normalized=float(env.heads.jerkiness(frame['observation'])[0]),
                             utilization=float((frame['peak_wrench'][0].abs() /
                                                frame['peak_wrench'].new_tensor(cfg['force_limits'])).max()),
                             success=bool(info['success'][0]), collision=bool(info['collision'][0]),
                             outside=bool(info['outside'][0]), timeout=bool(info['timeout'][0]),
                             separated=bool(info['separated'][0])))
            if term[0] or trunc[0]:
                outcome = ('force' if info['collision'][0] else 'workspace' if info['outside'][0]
                           else 'success' if info['success'][0] else 'timeout')
                break
        record = dict(case=case, outcome=outcome, steps=len(rows), discounted_return=total,
                      peak_net_force_n_2ms=peak_net, rows=rows)
        results.append(record)
        save()
        print(json.dumps({k: v for k, v in record.items() if k != 'rows'}), flush=True)
    complete = [r for r in results if r['outcome'] == 'success' and r['case'] in ('press', 'saw')]
    other = [r for r in results if r['case'] not in ('press', 'saw') and r['outcome'] != 'wall_budget']
    report['preference'] = ('not_checked' if not complete or not other else
                           'pass_observed_cases' if min(r['discounted_return'] for r in complete) >
                           max(r['discounted_return'] for r in other) else 'mixed_or_failed')
    report['not_covered'] = ['fitted WM rollout ordering (no new WM has been trained)']
    if not any(r['outcome'] == 'force' for r in results):
        report['not_covered'].append('physical force violation: attempted trajectories did not trigger it')
    slow_rows = [row for r in results if r['case'] == 'slow_press' for row in r['rows']]
    report['slow_press_max_limit_utilization'] = max((row['utilization'] for row in slow_rows), default=None)
    # Do not call a modest measured load a deliberately high-load example.
    report['not_covered'].append('near-limit slow pressing is not certified by a trajectory name')

    save()


if __name__ == '__main__':
    main()

