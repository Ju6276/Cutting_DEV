"""Run the pinned manipulation SuperMimic runner on the UR5e cutting task."""
import argparse
from dataclasses import asdict, replace
import json
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cutting_world_model.manipulation import CuttingEnvironment, CuttingMetricsLogger, SOURCE_REVISION
from cutting_world_model.provenance import git_revision
from cutting_world_model.benchmark import write_json

def restore(runner, path):
    from clab.algorithms.supermimic.networks import ACTOR_INPUT_NORMALIZATION
    from clab.algorithms.supermimic.runner import finite_tensors

    environment = json.loads((path.parent / 'params/environment.json').read_text())
    if environment != asdict(runner.env.cfg):
        raise ValueError('Checkpoint cutting environment differs from the current task')
    checkpoint = torch.load(path, map_location=runner.device, weights_only=True)
    checks = {'algorithm': 'supermimic', 'task': runner.task,
              'actor_input_normalization': ACTOR_INPUT_NORMALIZATION,
              'mdp_contract': runner.mdp.contract(), 'config': asdict(runner.config)}
    for key, expected in checks.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f'Checkpoint differs at {key}')
    if not finite_tensors(checkpoint):
        raise ValueError('Checkpoint contains non-finite tensors')
    runner.agent.load_state_dict(checkpoint['agent'])
    for key in ('iteration', 'sim_steps', 'real_transitions', 'collection_fraction', 'world_model_fraction'):
        setattr(runner, key, checkpoint[key])
    rng = checkpoint['rng']
    torch.set_rng_state(rng['torch'].cpu())
    if len(rng['cuda']) != torch.cuda.device_count():
        raise ValueError('Checkpoint visible CUDA device count differs')
    torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])
    random.setstate(rng['python'])
    n = rng['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--physics-python', type=Path, required=True)
    parser.add_argument('--backend-root', type=Path, default=ROOT / 'backends/mujoco')
    parser.add_argument('--worlds', type=int, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--reset-bank', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int,
                        help='Iterations executed by this invocation, including after resume')
    parser.add_argument('--max-hours', type=float, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online')
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.source_root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(args.source_root), 'status', '--porcelain'], text=True)
    if revision != SOURCE_REVISION or dirty:
        raise ValueError('Use the clean, pinned supermimic_manipulation checkout')
    if args.worlds < 1 or args.max_hours <= 0:
        raise ValueError('World count and wall budget must be positive')
    sys.path.insert(0, str(args.source_root / 'src'))
    from clab.algorithms.supermimic.agent import SuperMimicAgent
    from clab.algorithms.supermimic.config import SuperMimicConfig
    from clab.algorithms.supermimic.runner import RunConfig, SuperMimicRunner
    from clab.algorithms.supermimic.train import smoke_profile
    import wandb

    algorithm, run = SuperMimicConfig(), RunConfig()
    if args.smoke:
        algorithm, run = smoke_profile(algorithm, run)
    if args.iterations is not None:
        run = replace(run, iterations=args.iterations)
    run.validate(algorithm)
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = algorithm.tf32
    torch.backends.cudnn.allow_tf32 = algorithm.tf32
    args.output.mkdir(parents=True, exist_ok=False)
    params = args.output / 'params'
    params.mkdir()
    manifest = {'git': git_revision(ROOT), 'source_revision': revision, 'command': sys.argv,
                'wall_budget_seconds': args.max_hours * 3600, 'seed': args.seed,
                'algorithm': asdict(algorithm), 'run': asdict(run), 'worlds': args.worlds,
                'reset_mode': 'stage_bank' if args.reset_bank else 'full_task',
                'endpoint': 'configured upstream iterations per invocation',
                'shutdown': 'save at the next completed iteration, then close the physical worker'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stop = False
    checkpoint_requested = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    def request_checkpoint(signum, frame):
        nonlocal checkpoint_requested
        checkpoint_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_checkpoint)
    begin = time.monotonic()
    environment = runner = tracking = None
    exit_code = 0
    def iteration_boundary():
        nonlocal checkpoint_requested
        if checkpoint_requested:
            runner.save()
            checkpoint_requested = False
        write_json(args.output / 'progress.json', {
            'state': 'running', 'iteration': runner.iteration,
            'real_transitions': runner.real_transitions,
            'wm_updates': runner.agent.world_model_updates,
            'policy_updates': runner.agent.policy_updates,
            'completed_episodes': environment.completed_episodes,
            'successful_episodes': environment.successful_episodes,
            'wall_seconds': time.monotonic() - begin})
        return stop or time.monotonic() - begin >= args.max_hours * 3600
    try:
        with (args.output / 'physics.log').open('w') as physics_log:
            environment = CuttingEnvironment(
                args.physics_python, args.backend_root, worlds=args.worlds,
                scene=args.scene, capture=args.capture, log=physics_log,
                reset_bank=args.reset_bank, seed=args.seed)
            (params / 'environment.json').write_text(json.dumps(asdict(environment.cfg), indent=2) + '\n')
            (params / 'benchmark.json').write_text(json.dumps(
                environment.cfg.metadata['benchmark_contract'], indent=2) + '\n')
            tracking = wandb.init(project='cutting-manipulation', name=args.output.name,
                                  dir=str(args.output), mode=args.wandb_mode, config=manifest)
            (args.output / 'wandb.json').write_text(json.dumps(
                {key: getattr(tracking, key) for key in ('id', 'url', 'entity', 'project')}, indent=2) + '\n')
            runner = SuperMimicRunner(env=environment, adapter=environment, config=algorithm,
                                     run_config=run, output=args.output, task='ur5e_apple', logger='both')
            runner.logger = CuttingMetricsLogger(runner.logger, environment)
            if args.resume:
                restore(runner, args.resume.resolve())
            checkpoint, completed = runner.run(stop_requested=iteration_boundary)
            report = {'completed': completed, 'checkpoint': str(checkpoint),
                      'iterations': runner.iteration, 'real_transitions': runner.real_transitions,
                      'wm_updates': runner.agent.world_model_updates,
                      'policy_updates': runner.agent.policy_updates,
                      'completed_episodes': environment.completed_episodes,
                      'successful_episodes': environment.successful_episodes,
                      'physics_overflow_world_steps': runner.physics_overflow_world_steps,
                      'diverged_transitions': int(runner.diverged_transitions.item()),
                      'reward_observation_checks': runner.reward_comparison_metrics(),
                      'wall_seconds': time.monotonic() - begin}
            if args.smoke:
                reloaded = SuperMimicAgent(environment.mdp, algorithm, environment.device)
                saved = torch.load(checkpoint, map_location=environment.device, weights_only=True)
                reloaded.load_state_dict(saved['agent'])
                obs = runner.observations['actor']
                with torch.no_grad():
                    torch.testing.assert_close(reloaded.policy(obs), runner.agent.policy(obs), atol=0, rtol=0)
                report['reload_action_max_error'] = 0.0
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            write_json(args.output / 'progress.json', dict(report, state='completed' if completed else 'stopped'))
            print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        exit_code = 1
        (args.output / 'report.json').write_text(json.dumps({
            'completed': False, 'error': repr(error),
            'wall_seconds': time.monotonic() - begin}, indent=2) + '\n')
        write_json(args.output / 'progress.json', {
            'state': 'failed', 'error': repr(error), 'wall_seconds': time.monotonic() - begin})
        raise
    finally:
        try:
            if runner is not None:
                runner.close()
        finally:
            try:
                if environment is not None:
                    environment.close()
            finally:
                if tracking is not None:
                    tracking.finish(exit_code=exit_code)


if __name__ == '__main__':
    main()
