#!/usr/bin/env python3
"""Run the scene reference knife trajectory on an exported scene and measure object motion.

Pre-check for new objects (multi-object plan, P3): with a free bottom, does a sphere or cylinder
slide or roll under the knife? The bottom layer (particles within --bottom-band of the lowest
particle) is tracked: its horizontal displacement (sliding, DiSECt y is up) and its best-fit
rotation angle against the initial layout (rolling; Kabsch). Also reported per control step:
peak material force, released springs and the task termination flags. Task heads (goal, workspace)
are the prism benchmark's; for other objects they only bound the episode.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cutting_world_model.provenance import git_revision


def rotation_angle(before, after):
    """Angle (rad) of the best-fit rotation taking centred ``before`` points to ``after``."""
    a = before - before.mean(0)
    b = after - after.mean(0)
    u, _, vt = torch.linalg.svd(a.T @ b)
    d = torch.sign(torch.det(vt.T @ u.T))
    r = vt.T @ torch.diag(torch.tensor([1., 1., float(d)], dtype=a.dtype, device=a.device)) @ u.T
    return math.acos(max(-1., min(1., (float(torch.trace(r)) - 1) / 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-root', type=Path, required=True)
    parser.add_argument('--scene', required=True, help='exported scene relative to the backend root')
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--bottom-band', type=float, default=.002)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = str(args.backend_root.resolve())
    sys.path.insert(0, root)
    from cutting_world_model.gpu_env import GPUEnvironment
    env = GPUEnvironment(root, 1, scene=args.scene)
    material = env.physics.material
    q0 = material.state_field('particle_q')[0].double().clone()
    static = (material.model.particle_inv_mass == 0).cpu()
    bottom = q0[:, 1] < q0[:, 1].min() + args.bottom_band
    zero = torch.zeros(1, 6, device=env.device)
    rows, end = [], 'steps'
    for step in range(1, args.steps + 1):
        frame, reward, terminated, truncated, info = env.step(zero, reference=True)
        q = material.state_field('particle_q')[0].double()
        shift = q[bottom] - q0[bottom]
        horizontal = shift[:, [0, 2]].norm(dim=-1)
        rows.append({'step': step, 'peak_force': float(info['peak_force'][0]),
                     'released': int(info['released_springs'][0]),
                     'bottom_horizontal_max_mm': 1e3 * float(horizontal.max()),
                     'bottom_horizontal_mean_mm': 1e3 * float(horizontal.mean()),
                     'bottom_vertical_mean_mm': 1e3 * float(shift[:, 1].mean()),
                     'bottom_rotation_deg': math.degrees(rotation_angle(q0[bottom], q[bottom])),
                     'centroid_shift_mm': (1e3 * (q.mean(0) - q0.mean(0))).tolist()})
        if bool(terminated[0] | truncated[0]):
            end = ('success' if bool(info['success'][0]) else 'collision' if bool(info['collision'][0])
                   else 'outside' if bool(info['outside'][0]) else 'truncated')
            break
    summary = {'git': git_revision(ROOT), 'command': sys.argv, 'scene': args.scene,
               'particles': int(q0.shape[0]), 'static_particles': int(static.sum()),
               'bottom_particles': int(bottom.sum()), 'object_height_m': float(q0[:, 1].max() - q0[:, 1].min()),
               'object_top_m': float(q0[:, 1].max()), 'cut_springs': int(material.springs),
               'steps': len(rows), 'end': end,
               'peak_force_max': max(r['peak_force'] for r in rows), 'released_end': rows[-1]['released'],
               'bottom_horizontal_max_mm': max(r['bottom_horizontal_max_mm'] for r in rows),
               'bottom_rotation_max_deg': max(r['bottom_rotation_deg'] for r in rows)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'report.json').write_text(json.dumps({**summary, 'per_step': rows}, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
