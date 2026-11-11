#!/usr/bin/env python3
"""Material-scale stability grid of the DiSECt substep for each S0 object.

One group per object; each world gets one (Young's modulus, density, cut-spring) scale
combination. The knife is held with zero action (step 1 failures in the stress test happened
with zero knife velocity), then optionally driven down with the reference trajectory.
Nonfinite worlds are recorded instead of stopping the run.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model.multi_env import MultiConditionEnvironment, ObjectGroup  # noqa: E402
from cutting_gpu_backend.batched_disect import BatchedDiSECt  # noqa: E402

BAD = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--idle-steps', type=int, default=20)
    parser.add_argument('--reference-steps', type=int, default=200)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dt', type=float)
    parser.add_argument('--only')
    parser.add_argument('--E', type=float, nargs='+', default=[1., 1.25, 1.5, 1.75, 2.])
    args = parser.parse_args()
    advance = BatchedDiSECt.advance

    def tolerant(self, duration, active=None):
        try:
            return advance(self, duration, active=active)
        except FloatingPointError:
            q = self.state.particle_q.reshape(self.worlds, -1)
            ke = self.state.cut_spring_ke.reshape(self.worlds, -1)
            bad = ((~torch.isfinite(q).all(-1)) | (~torch.isfinite(ke).all(-1))).cpu()
            key = id(self)
            BAD[key] = BAD.get(key, torch.zeros_like(bad)) | bad
            return torch.nan_to_num(torch.linalg.vector_norm(self.state_field('knife_f'), dim=-1).sum(-1))

    BatchedDiSECt.advance = tolerant
    cfg = json.loads((args.root / 'configs/multi_condition/s0_train.json').read_text())
    grid = list(itertools.product(args.E, (.8, 1., 1.25), (.5, 1., 2.)))
    groups = [ObjectGroup(name=g['name'], scene=g['scene'], worlds=len(grid), clearance=(.01, .01))
              for g in cfg['groups'] if args.only in (None, g['name'])]
    env = MultiConditionEnvironment(args.root / 'backends/mujoco', groups, seed=0,
                                    force_limits=cfg['force_limits'], parallel=False)
    E, rho, ks = (torch.tensor(c) for c in zip(*grid))
    if args.dt:
        for member in env.members:
            member.env.physics.material.dt = args.dt
    peak = [torch.zeros(len(grid)) for _ in env.members]
    for member in env.members:
        member.reset(torch.arange(len(grid)), {'youngs_modulus': E, 'density': rho, 'cut_spring_stiffness': ks},
                     torch.full((len(grid),), .03))
    env.current = env._gather() if hasattr(env, '_gather') else env.current
    report = {}
    first_bad = {}
    phases = [('idle', args.idle_steps, False), ('reference', args.reference_steps, True)]
    step = 0
    for phase, count, reference in phases:
        for _ in range(count):
            step += 1
            for member in env.members:
                member.env.needs_reset.zero_()
                _, _, _, _, info = member.step(torch.zeros(member.env.worlds, 6), reference=reference)
                k = env.members.index(member)
                peak[k] = torch.maximum(peak[k], torch.nan_to_num(info['peak_force'].float().cpu()))
            for i, member in enumerate(env.members):
                bad = BAD.get(id(member.env.physics.material), torch.zeros(len(grid), dtype=torch.bool))
                for w in bad.nonzero().flatten().tolist():
                    first_bad.setdefault((i, w), (phase, step))
    for i, (group, member) in enumerate(zip(groups, env.members)):
        rows = []
        for w, (e, d, k) in enumerate(grid):
            rows.append({'E': e, 'density': d, 'cut_spring': k, 'first_nonfinite': first_bad.get((i, w)), 'peak_force_n': float(peak[i][w])})
        report[group.name] = rows
        print(group.name, 'dt', member.env.physics.material.dt, flush=True)
        for r in rows:
            print(f"  E x{r['E']:<4} rho x{r['density']:<4} ks x{r['cut_spring']:<3} -> {r['first_nonfinite']} peak {r['peak_force_n']:.1f} N", flush=True)
    args.output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
