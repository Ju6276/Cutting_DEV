"""Validate half-potato randomized production resets and distribution corners."""
import argparse
import itertools
import json
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'backends/mujoco')]
from cutting_world_model.multi_env import MultiConditionEnvironment, knife_clearance
from cutting_world_model.gpu_env import GPUEnvironment
from cutting_world_model.provenance import git_revision
from scripts.compare_potato_preparation import metrics, cpu_row, stable_windows


def follow(material, deadline):
    if bool((material.model.particle_inv_mass == 0).any()):
        raise ValueError('Expected a freely placed object')
    rows = [dict(time_s=0., **cpu_row(metrics(material)))]
    peak = torch.zeros(material.worlds, device='cuda')
    for step in range(10):
        if time.monotonic() > deadline:
            raise TimeoutError('Reset verification wall budget')
        peak = torch.maximum(peak, material.advance_stationary(.05))
        rows.append(dict(time_s=(step+1)*.05, **cpu_row(metrics(material))))
    accepted = np.asarray(stable_windows(rows))[5:].all(0)
    undamaged = np.asarray([r['released'] for r in rows]).max(0) == 0
    finite = np.isfinite(np.array([r['com_m'] for r in rows])).all((0, 2))
    ok = accepted & undamaged & finite & (peak.cpu().numpy() == 0)
    return dict(passed=ok.tolist(), stable=accepted.tolist(),
                max_knife_contact=peak.tolist(), rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/free_cut/pose_potato.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wall-seconds', type=int, default=900)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    def stop(*_): raise TimeoutError('Reset verification wall budget')
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(a.wall_seconds)
    torch.set_num_threads(2)
    cfg = json.loads(a.config.read_text())
    cfg['parallel'] = False
    cfg['groups'][0]['worlds'] = 64
    cfg['seed'] = 11314
    deadline = time.monotonic()+a.wall_seconds
    report = dict(git=git_revision(ROOT), config=cfg, batches=[],
                  criteria=dict(window_s=.25, com_speed_mm_s=.2, particle_rms_mm_s=1.,
                                displacement_mm=.01, followup_s=.5))
    (a.output/'manifest.json').write_text(json.dumps(report, indent=2))
    multi = None
    try:
        multi = MultiConditionEnvironment.from_config(ROOT/'backends/mujoco', cfg)
        for batch in range(2):
            if batch:
                multi.reset()
            result = follow(multi.members[0].env.physics.material, deadline)
            result.update(kind='log_uniform_training_distribution', batch=batch,
                          conditions=multi.conditions(range(multi.worlds)))
            report['batches'].append(result)
            print('Random reset', batch, sum(result['passed']), '/', len(result['passed']), flush=True)
        multi.close()
        multi = None
        group = cfg['groups'][0]
        keys = ['youngs_modulus', 'density', 'cut_spring_stiffness']
        corners = list(itertools.product(*(group['material'][k] for k in keys), group['clearance']))
        lift = torch.tensor([x[-1] for x in corners])-knife_clearance(ROOT/'backends/mujoco', group['scene'])
        env = GPUEnvironment(ROOT/'backends/mujoco', len(corners), scene=group['scene'],
                             initial_lift=lift, sim_dt=group['sim_dt'], force_model='net',
                             friction_velocity=group['friction_velocity'], task=cfg['task'],
                             force_limits=cfg['force_limits'])
        material = env.physics.material
        mask = torch.ones(len(corners), device='cuda', dtype=torch.bool)
        scales = {k: torch.tensor([x[i] for x in corners], device='cuda') for i,k in enumerate(keys)}
        material.set_material_scales(mask, **scales)
        env.target_speed = torch.full((len(corners),), .03, device='cuda')
        env.settle(group['settle_seconds'])
        result = follow(material, deadline)
        result.update(kind='all_material_and_clearance_corners', conditions=corners,
                      columns=keys+['clearance'])
        report['batches'].append(result)
        report['passed'] = all(all(r['passed']) for r in report['batches'])
        report['total_conditions'] = sum(len(r['passed']) for r in report['batches'])
        report['passed_conditions'] = sum(sum(r['passed']) for r in report['batches'])
        (a.output/'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k: report[k] for k in ('passed','total_conditions','passed_conditions')}), flush=True)
        if not report['passed']:
            raise RuntimeError('Stability check failed; inspect recorded cases before choosing settling time')
    finally:
        if multi is not None:
            multi.close()
        signal.alarm(0)


if __name__ == '__main__':
    main()
