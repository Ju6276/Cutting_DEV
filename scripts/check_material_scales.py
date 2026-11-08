#!/usr/bin/env python3
"""P0: per-world material scales in the batched benchmark match separate single-world runs.

The coupled UR5e/FDCC/DiSECt environment runs the scene reference trajectory in one batch whose
worlds carry different material scales, and separately in one single-world environment per scale.
Both run the same number of control steps: until the first world of the batch terminates (or
--steps). Reported per world: maximum batch-minus-single difference of peak force, observation and
released springs, and the same quantities against the unscaled world (the size of the material
effect that the batch must resolve).
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cutting_world_model.provenance import git_revision

# World 0 is the exported material; each other world changes one parameter.
DEFAULT_WORLDS = [
    {},
    {'youngs_modulus': 2.},
    {'youngs_modulus': .5},
    {'density': 1.5},
    {'cut_spring_stiffness': .5},
    {'cut_spring_stiffness': 2.},
]


def make_env(root, worlds):
    from cutting_world_model.gpu_env import GPUEnvironment
    env = GPUEnvironment(root, len(worlds))
    every = torch.ones(len(worlds), dtype=torch.bool, device=env.device)
    for key in sorted({k for w in worlds for k in w}):
        env.physics.material.set_material_scales(
            every, **{key: torch.tensor([w.get(key, 1.) for w in worlds], device=env.device)})
    env.reset()
    return env


def rollout(env, steps):
    zero = torch.zeros(env.worlds, 6, device=env.device)
    rows = []
    for _ in range(steps):
        frame, reward, terminated, truncated, info = env.step(zero, reference=True)
        rows.append({'peak_force': info['peak_force'].double().cpu(),
                     'released': info['released_springs'].cpu(),
                     'observation': frame['observation'].double().cpu(),
                     'reward': reward.double().cpu(),
                     'done': (terminated | truncated).cpu()})
        if bool((terminated | truncated).any()):
            break
    return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}


def compare_export(root, export, scales):
    """Scaled source arrays against a scene exported from a config with the scaled settings."""
    from cutting_gpu_backend.batched_disect import MATERIAL_SCALES
    source = torch.load(Path(root) / 'runs/disect_source_v4/scene.pt', weights_only=False, map_location='cpu')
    other = torch.load(export, weights_only=False, map_location='cpu')
    result = {}
    for key, value in scales.items():
        for name in MATERIAL_SCALES[key]['scaled'] + MATERIAL_SCALES[key]['inverse']:
            base, target = getattr(source['model'], name), getattr(other['model'], name)
            scaled = base * value if name in MATERIAL_SCALES[key]['scaled'] else base / value
            scaled = torch.where(base == 0, base, scaled)
            result[name] = float(((scaled - target).abs() / target.abs().clamp_min(1e-30)).max())
    base, target = source['state'].cut_spring_ke, other['state'].cut_spring_ke
    if 'cut_spring_stiffness' in scales:
        result['state.cut_spring_ke'] = float(((base * scales['cut_spring_stiffness'] - target).abs()
                                               / target.abs()).max())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scaled-export', type=Path,
                        help='scene exported from the source config with --export-scales applied')
    parser.add_argument('--export-scales', type=json.loads, default=None)
    args = parser.parse_args()
    root = str(args.backend_root.resolve())
    sys.path.insert(0, root)
    worlds = DEFAULT_WORLDS
    batch = rollout(make_env(root, worlds), args.steps)
    steps = len(batch['peak_force'])
    report = {'git': git_revision(ROOT), 'command': sys.argv, 'worlds': worlds, 'steps': steps, 'per_world': []}
    for i, scales in enumerate(worlds):
        single = rollout(make_env(root, [scales]), steps)
        if len(single['peak_force']) != steps:
            raise RuntimeError(f'World {i} single run ended after {len(single["peak_force"])} of {steps} steps')
        diff = lambda key, ref, j: float((batch[key][:, i] - ref[key][:, j]).abs().max())
        report['per_world'].append({
            'scales': scales,
            'batch_vs_single': {k: diff(k, single, 0) for k in ('peak_force', 'observation', 'reward', 'released')},
            'vs_unscaled_world': {k: diff(k, batch, 0) for k in ('peak_force', 'observation', 'reward', 'released')},
            'peak_force_max': float(batch['peak_force'][:, i].max()),
            'released_end': int(batch['released'][-1, i]),
            'done_end': bool(batch['done'][-1, i]),
        })
        print(json.dumps(report['per_world'][-1]), flush=True)
    if args.scaled_export:
        report['export_max_relative_error'] = compare_export(root, args.scaled_export, args.export_scales)
        print(json.dumps(report['export_max_relative_error']), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
