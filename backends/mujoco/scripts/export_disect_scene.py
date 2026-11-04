"""Export an original DiSECt scene and a short exact-motion reference without ROS."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from provenance import git_revision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disect-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--warmup-steps', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.disect_root.resolve()))
    from disect.cutting import CuttingSim, load_settings
    settings = load_settings(str(args.config.resolve()))
    settings.sim_dt = 1e-5
    settings.sim_substeps = 500
    with torch.no_grad():
        sim = CuttingSim(settings, adapter='cuda', requires_grad=False, allow_nans=False,
                         log_folder=str(args.output / 'log'))
        sim.cut()
        sim.state = sim.model.state()
        sim.disable_gradients()
        for _ in range(args.warmup_steps):
            sim.simulation_step()
        torch.save({'model': sim.model, 'state': sim.state,
                    'initial_pos': sim.motion.current_pos,
                    'initial_rot': sim.motion.current_rot,
                    'linear_velocity': sim.motion.lin_vel,
                    'angular_velocity': sim.motion.ang_vel,
                    'spine_height': sim.knife.spine_height,
                    'settings': dict(settings)}, args.output / 'scene.pt')
        for _ in range(args.steps):
            sim.simulation_step()
        torch.save({key: getattr(sim.state, key) for key in
                    ('particle_q', 'particle_qd', 'cut_spring_ke', 'knife_f', 'joint_q', 'joint_qd')},
                   args.output / 'reference.pt')
    (args.output / 'manifest.json').write_text(json.dumps({
        'git_revision': git_revision(), 'disect_revision': __import__('subprocess').check_output(
            ['git', '-C', str(args.disect_root), 'rev-parse', 'HEAD'], text=True).strip(),
        'command': sys.argv, 'settings': dict(settings), 'reference_steps': args.steps,
        'warmup_steps': args.warmup_steps,
        'vertices': sim.model.particle_count, 'springs': sim.model.cut_spring_count,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
