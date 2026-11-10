#!/usr/bin/env python3
"""Parallel (process-per-group) against sequential multi-object stepping: same seed, same
actions; reports the largest per-field differences and the seconds per control step."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model.multi_env import MultiConditionEnvironment


def run(config, parallel, steps, seed):
    config = dict(config, parallel=parallel)
    started = time.monotonic()
    env = MultiConditionEnvironment.from_config(ROOT / 'backends/mujoco', config)
    built = time.monotonic() - started
    generator = torch.Generator().manual_seed(seed)
    frames, rewards, ends = [env.current], [], []
    started = time.monotonic()
    for _ in range(steps):
        action = torch.rand(env.worlds, 6, generator=generator) * 2 - 1
        frame, reward, terminated, truncated, info = env.step(action.to(env.device))
        frames.append({k: v.clone() for k, v in frame.items()})
        rewards.append(reward.clone())
        ended = terminated | truncated
        ends.append(ended.clone())
        if bool(ended.any()):
            env.reset(ended)
            frames[-1] = {k: v.clone() for k, v in env.current.items()}
    per_step = (time.monotonic() - started) / steps
    env.close()
    return frames, rewards, ends, built, per_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', nargs=2, choices=('sequential', 'parallel'), default=('sequential', 'parallel'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    a = run(config, args.modes[0] == 'parallel', args.steps, args.seed)
    b = run(config, args.modes[1] == 'parallel', args.steps, args.seed)
    diff, per_frame = {}, []
    for fa, fb in zip(a[0], b[0]):
        row = {}
        for k in ('joint_q', 'joint_qd', 'peak_wrench', 'wrench_history', 'observation', 'target_speed'):
            row[k] = float((fa[k] - fb[k]).abs().max())
            diff[k] = max(diff.get(k, 0.), row[k])
        per_frame.append({k: round(v, 6) for k, v in row.items() if k in ('joint_q', 'peak_wrench')})
    diff['reward'] = max(float((ra - rb).abs().max()) for ra, rb in zip(a[1], b[1]))
    diff['ending_mismatches'] = sum(int((ea != eb).sum()) for ea, eb in zip(a[2], b[2]))
    report = {'config': str(args.config), 'steps': args.steps, 'modes': list(args.modes),
              'first': {'build_s': a[3], 'step_s': a[4]}, 'second': {'build_s': b[3], 'step_s': b[4]},
              'max_abs_difference': diff, 'per_frame': per_frame}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
