#!/usr/bin/env python3
"""Bounded eager/graph reset comparison, exact same integrator/materials/duration."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_gpu_backend.batched_disect import BatchedDiSECt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=.75)
    p.add_argument('--worlds', type=int, nargs='+', default=[1, 256])
    p.add_argument('--wall-seconds', type=float, default=1200.)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((ROOT / 'configs/free_cut/review_v2.json').read_text())
    results = []
    started = time.monotonic()
    def write():
        (args.output / 'report.json').write_text(json.dumps(dict(
            duration=args.seconds, results=results, elapsed=time.monotonic()-started), indent=2))
    def timed(fn):
        torch.cuda.synchronize(); t=time.monotonic(); fn(); torch.cuda.synchronize()
        return time.monotonic()-t
    for n in args.worlds:
        if time.monotonic()-started > args.wall_seconds:
            break
        m = BatchedDiSECt(ROOT / 'backends/mujoco' / cfg['scene'], n, friction_velocity=.01)
        all_rows = torch.ones(n, dtype=torch.bool, device='cuda')
        m.set_motion(torch.zeros(n, 6, device='cuda'))
        m.set_material_scales(all_rows, youngs_modulus=torch.linspace(1.,2.,n,device='cuda'))
        m.restore_export(all_rows)
        saved = m.capture_rows(all_rows)
        for selected in sorted(set([1,n])):
            if time.monotonic()-started > args.wall_seconds: break
            mask = torch.arange(n,device='cuda') < selected
            m.restore_rows(all_rows, saved)
            eager = timed(lambda:m.advance(args.seconds,active=mask,track_peak=True))
            reference = m.capture_rows(all_rows)
            m.restore_rows(all_rows,saved)
            first = timed(lambda:m.settle_selected(args.seconds,mask))
            computed = m.capture_rows(all_rows)
            m.restore_rows(all_rows,saved)
            warm = timed(lambda:m.settle_selected(args.seconds,mask))
            state_err = {}
            for name in ('particle_q','particle_qd','cut_spring_ke','knife_contact_q'):
                state_err[name] = float((reference[0][name]-computed[0][name]).abs().max())
            unchanged = all(torch.equal(saved[i][k][~mask],computed[i][k][~mask])
                            for i in range(3) for k in saved[i])
            row = dict(worlds=n,selected=selected,eager_s=eager,first_graph_s=first,
                       warm_graph_s=warm,speedup=eager/warm,errors=state_err,
                       inactive_bitwise_unchanged=unchanged)
            results.append(row); print(json.dumps(row),flush=True);write()
        del m
    write()

if __name__=='__main__': main()
