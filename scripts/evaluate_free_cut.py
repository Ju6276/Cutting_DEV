#!/usr/bin/env python3
"""Frozen WM-MPPI / policy evaluation on the explicit free_cut_v3 distribution."""
import argparse
import importlib.metadata
import platform
import json
from pathlib import Path
import signal
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cutting_world_model.benchmark import evaluate, file_digest, write_json
from cutting_world_model.free_cut_evaluation import FreeCutEvaluationEnvironment, selected_config
from cutting_world_model.provenance import git_revision
from evaluate_benchmark import make_controller


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=('wm-mppi', 'supermimic', 'constant-zero'), required=True)
    p.add_argument('--conditions', type=Path, required=True,
                   help='Exact conditions file recorded by the training checkpoint')
    p.add_argument('--group', required=True)
    p.add_argument('--transfer-scene',
                   help='Explicit unseen object .pt path relative to backend; keeps the training task and solver')
    p.add_argument('--backend-root', type=Path, default=ROOT / 'backends/mujoco')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--paired-checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--wall-seconds', type=float, default=900.)
    p.add_argument('--mpc-horizon', type=int, default=8)
    p.add_argument('--mpc-samples', type=int, default=512)
    p.add_argument('--mpc-refinements', type=int, default=1)
    p.add_argument('--mpc-temperature', type=float, default=1.)
    p.add_argument('--mpc-noise-std', type=float, default=.5)
    p.add_argument('--mpc-discount', type=float, default=None,
                   help='Defaults to the task gamma; a mismatch is rejected')
    p.add_argument('--mpc-terminal-value', choices=('none', 'critic'), default='none')
    return p


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if args.episodes < 1 or not 0 < args.wall_seconds < float('inf'):
        parser.error('Episodes and finite wall budget must be positive')
    if args.method != 'constant-zero' and args.checkpoint is None:
        parser.error('Frozen learned controllers require --checkpoint')
    if args.paired_checkpoint is not None and args.method != 'wm-mppi':
        parser.error('--paired-checkpoint applies only to MPPI')
    conditions = json.loads(args.conditions.read_text())
    selected = selected_config(conditions, args.group, args.seed, args.transfer_scene)
    gamma = conditions['task']['gamma']
    args.mpc_discount = gamma if args.mpc_discount is None else args.mpc_discount
    if args.mpc_discount != gamma:
        parser.error('MPPI discount must equal the explicit task gamma')
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        from cutting_world_model.benchmark import check_contract
        check_contract(conditions, saved['metadata'].get('task_conditions'), 'training_conditions')
        del saved
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def terminate(signum, stack):
        raise InterruptedError('Free-cut evaluation received termination signal')
    signal.signal(signal.SIGTERM, terminate)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(args.backend_root.resolve()))
    manifest = {
        'git': git_revision(ROOT), 'command': sys.argv, 'method': args.method,
        'training_conditions': conditions, 'runtime_conditions': selected,
        'scene_transfer': {
            'source_group': args.group,
            'source_scene': next(g['scene'] for g in conditions['groups'] if g['name'] == args.group),
            'evaluation_scene': selected['groups'][0]['scene'],
            'unseen_scene': args.transfer_scene is not None,
            'scope': 'Native geometry/material parameters change with the asset; no model updates.',
        },
        'conditions_sha256': file_digest(args.conditions),
        'checkpoint': str(args.checkpoint.resolve()) if args.checkpoint else None,
        'checkpoint_sha256': file_digest(args.checkpoint) if args.checkpoint else None,
        'worlds': 1, 'requested_episodes': args.episodes, 'wall_budget_seconds': args.wall_seconds,
        'reset_mode': 'randomized_material_and_speed_per_episode_clearance_per_world',
        'physics_assets': {str(path): file_digest(args.backend_root / path) for path in (
            Path(selected['groups'][0]['scene']),
            Path(next(g['scene'] for g in conditions['groups'] if g['name'] == args.group)),
            Path('runs/model_coupled/robot.mjb'),
            Path('runs/controller_source/controller_params.json'))},
        'gpu': torch.cuda.get_device_name(0),
        'runtime': {'python': platform.python_version(), 'packages': {
            name: importlib.metadata.version(name) for name in
            ('torch', 'numpy', 'mujoco', 'mujoco-warp', 'warp-lang')}},
        'implementation_sha256': {str(path.relative_to(ROOT)): file_digest(path)
                                  for path in sorted((ROOT / 'cutting_world_model').glob('*.py'))
                                  + [Path(__file__).resolve(), ROOT / 'scripts/evaluate_benchmark.py']},
        'scope': 'Frozen physical evaluation; no optimizer updates. Early WM results are not algorithm comparisons.',
    }
    write_json(args.output / 'manifest.json', manifest)
    env = policy = None
    try:
        env = FreeCutEvaluationEnvironment(args.backend_root, conditions, args.group, args.seed,
                                           args.transfer_scene)
        write_json(args.output / 'task.json', env.metadata)
        policy = make_controller(args, env, manifest)
        manifest['setup_seconds'] = time.monotonic() - started
        write_json(args.output / 'manifest.json', manifest)
        report = evaluate(env, policy, args.output, args.episodes, started + args.wall_seconds)
        if report['state'] != 'completed':
            raise TimeoutError('Evaluation reached its wall budget; see report.json')
    except BaseException as error:
        if not (args.output / 'report.json').exists():
            write_json(args.output / 'report.json', {'state': 'failed', 'error': repr(error)})
        raise
    finally:
        if policy is not None and hasattr(policy, 'plans'):
            write_json(args.output / 'planning.json', {'planner': policy.manifest, 'steps': policy.plans})
        if policy is not None and hasattr(policy, 'close'):
            policy.close()
        if env is not None:
            env.close()
        report_path = args.output / 'report.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            report['total_wall_seconds'] = time.monotonic() - started
            write_json(report_path, report)


if __name__ == '__main__':
    main()
