#!/usr/bin/env python3
"""Functional check of the multi-condition environment with the reference knife trajectory.

Builds object groups (scene, worlds, force-limit scale, clearance and material ranges),
prints each world's sampled conditions, then runs the reference control and reports
each world's first ending, maximum sensor-limit ratio and released springs.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model.multi_env import MultiConditionEnvironment, ObjectGroup, knife_clearance

MATERIAL = {'youngs_modulus': (.5, 2.), 'density': (.8, 1.25), 'cut_spring_stiffness': (.5, 2.)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-root', type=Path, default=ROOT / 'backends/mujoco')
    parser.add_argument('--group', nargs=4, action='append', metavar=('NAME', 'SCENE', 'WORLDS', 'LIMIT_SCALE'),
                        required=True)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    groups = [ObjectGroup(n, s, int(w), float(l), material=MATERIAL) for n, s, w, l in args.group]
    report = {'groups': [{'name': g.name, 'scene': g.scene, 'worlds': g.worlds,
                          'force_limit_scale': g.force_limit_scale,
                          'default_gap_m': knife_clearance(args.backend_root, g.scene)} for g in groups]}
    started = time.monotonic()
    env = MultiConditionEnvironment(args.backend_root, groups, seed=args.seed)
    report['init_seconds'] = time.monotonic() - started
    report['frame_keys'] = sorted(env.current)
    report['conditions'] = env.conditions(range(env.worlds))
    first, margin = {}, torch.zeros(env.worlds)
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        frame, reward, terminated, truncated, info = env.step(torch.zeros(env.worlds, 6), reference=True)
        margin = torch.maximum(margin, info['safety_margin'].float().cpu())
        ended = terminated | truncated
        for w in ended.nonzero().flatten().tolist():
            if w not in first:
                first[w] = {'step': step, 'ending': 'success' if info['success'][w] else 'collision'
                            if info['collision'][w] else 'outside' if info['outside'][w] else 'truncated',
                            'released': int(info['released_springs'][w]), 'max_limit_ratio': float(margin[w])}
        if bool(ended.any()):
            env.reset(ended)
        if len(first) == env.worlds:
            break
    report['seconds_per_step'] = (time.monotonic() - started) / step
    report['endings'] = {w: first.get(w) for w in range(env.worlds)}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for w in range(env.worlds):
        c = report['conditions'][w]
        print(w, c['group'], round(c['start_lift_m'], 4), round(c['youngs_modulus'], 2),
              round(c['cut_spring_stiffness'], 2), round(c['target_speed_mps'], 3), first.get(w), flush=True)


if __name__ == '__main__':
    main()
