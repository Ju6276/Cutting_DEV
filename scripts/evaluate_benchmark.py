#!/usr/bin/env python3
"""Run frozen controllers through one fixed physical cutting environment."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import signal
import platform

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cutting_world_model.benchmark import (
    ConstantPolicy, ReferencePolicy, WorldModelPolicy, benchmark_scene, check_contract, evaluate, file_digest,
    physical_contract, write_json,
)
from cutting_world_model.provenance import git_revision


class SACPolicy:
    def __init__(self, args, env):
        from cutting_world_model.sac_gpu_bridge import SACPolicyProcess
        self.process = SACPolicyProcess(
            python=args.sac_python, sliceit_root=args.sliceit_root,
            config=args.sac_config, model_dir=args.checkpoint)
        try:
            self.process.validate_metadata(env.metadata)
        except BaseException:
            self.process.close()
            raise

    def reset(self, frame):
        pass

    def act(self, frame):
        action = self.process.act(frame['observation'].cpu().numpy(), deterministic=True)
        return torch.as_tensor(action, device=frame['observation'].device, dtype=torch.float32)

    def observe(self, frame):
        pass

    def unchanged(self):
        return self.process.unchanged()

    def close(self):
        self.process.close()


def argument_parser(default_method=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=('constant-zero', 'reference', 'supermimic', 'sac', 'wm-mppi'),
                        required=default_method is None, default=default_method)
    parser.add_argument('--backend-root', type=Path, default=ROOT / 'backends/mujoco')
    parser.add_argument('--contract', type=Path, default=ROOT / 'configs/public_prism/benchmark.json')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--wall-seconds', type=float, default=600.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sac-python', type=Path)
    parser.add_argument('--sliceit-root', type=Path)
    parser.add_argument('--sac-config', type=Path)
    parser.add_argument('--trajectory', type=Path,
                        help='Reference knife trajectory (cutting-knife-velocity-v1 JSON); '
                             'default is the scene constant-velocity reference')
    parser.add_argument('--mpc-horizon', type=int, default=8)
    parser.add_argument('--mpc-samples', type=int, default=512)
    parser.add_argument('--mpc-refinements', type=int, default=1)
    parser.add_argument('--mpc-temperature', type=float, default=1.)
    parser.add_argument('--mpc-noise-std', type=float, default=.5)
    parser.add_argument('--mpc-discount', type=float, default=.99)
    parser.add_argument('--mpc-terminal-value', choices=('none', 'critic'), default='none')
    return parser


def make_controller(args, env, manifest):
    if args.method == 'supermimic':
        policy = WorldModelPolicy(args.checkpoint, env)
        manifest['policy_updates'] = policy.updates
        manifest['policy_training_reward'] = policy.training_reward
    elif args.method == 'wm-mppi':
        from cutting_world_model.mpc import MPPIConfig, MPPIPolicy
        config = MPPIConfig(args.mpc_horizon, args.mpc_samples, args.mpc_refinements,
                            args.mpc_temperature, args.mpc_noise_std, args.mpc_discount,
                            args.mpc_terminal_value)
        policy = MPPIPolicy(args.checkpoint, env, config)
        manifest['planner'] = policy.manifest
    elif args.method == 'sac':
        policy = SACPolicy(args, env)
        manifest['sac_actor'] = policy.process.metadata
    elif args.method == 'reference':
        from cutting_gpu_backend.reference import KnifeTrajectory
        trajectory = KnifeTrajectory.load(args.trajectory) if args.trajectory else None
        env.physics.set_reference_trajectory(trajectory)
        trajectory = env.physics.reference_trajectory
        manifest['trajectory'] = {
            'path': str(args.trajectory.resolve()) if args.trajectory else None,
            'sha256': file_digest(args.trajectory) if args.trajectory else None,
            'values_sha256': trajectory.digest(), 'intervals': len(trajectory.linear_velocity),
            'duration_seconds': trajectory.duration, 'source': trajectory.metadata}
        policy = ReferencePolicy()
    else:
        policy = ConstantPolicy()
    return policy


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if args.episodes < 1 or not 0 < args.wall_seconds < float('inf'):
        parser.error('Episodes and finite wall budget must be positive')
    if args.trajectory is not None and args.method != 'reference':
        parser.error('--trajectory applies to --method reference')
    if args.method not in ('constant-zero', 'reference') and args.checkpoint is None:
        parser.error('Learned policies require --checkpoint')
    if args.method == 'sac' and any(value is None for value in
                                  (args.sac_python, args.sliceit_root, args.sac_config)):
        parser.error('SAC requires --sac-python, --sliceit-root and --sac-config')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def terminate(signum, stack):
        raise InterruptedError('Benchmark received termination signal')
    signal.signal(signal.SIGTERM, terminate)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    manifest = {'git': git_revision(ROOT), 'command': sys.argv, 'method': args.method,
                'contract': json.loads(args.contract.read_text()), 'worlds': 1,
                'requested_episodes': args.episodes, 'action_noise_std': 0.,
                'wall_budget_seconds': args.wall_seconds,
                'checkpoint': str(args.checkpoint.resolve()) if args.checkpoint else None,
                'control_branch': 'knife_reference' if args.method == 'reference' else 'policy_action',
                'runtime': {'python': platform.python_version(),
                            'packages': {name: importlib.metadata.version(name) for name in
                                         ('torch', 'numpy', 'mujoco', 'mujoco-warp', 'warp-lang')},
                            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
                'endpoint': 'Complete frozen episodes or wall budget'}
    write_json(args.output / 'manifest.json', manifest)
    policy = None
    try:
        sys.path.insert(0, str(args.backend_root.resolve()))
        from cutting_world_model.gpu_env import GPUEnvironment
        manifest['scene'] = benchmark_scene(manifest['contract'])
        write_json(args.output / 'manifest.json', manifest)
        env = GPUEnvironment(args.backend_root, worlds=1, scene=manifest['scene'])
        actual = physical_contract(args.backend_root, env)
        check_contract(manifest['contract'], actual)
        write_json(args.output / 'task.json', env.metadata)
        policy = make_controller(args, env, manifest)
        manifest['setup_seconds'] = time.monotonic() - started
        write_json(args.output / 'manifest.json', manifest)
        report = evaluate(env, policy, args.output, args.episodes, started + args.wall_seconds)
        if report['state'] != 'completed':
            raise TimeoutError('Benchmark reached its wall budget; see report.json')
    except BaseException as error:
        if not (args.output / 'report.json').exists():
            write_json(args.output / 'report.json', {'state': 'failed', 'error': repr(error),
                                                    'wall_seconds': time.monotonic() - started})
        raise
    finally:
        if policy is not None and hasattr(policy, 'plans'):
            write_json(args.output / 'planning.json', {'planner': policy.manifest, 'steps': policy.plans})
        report_path = args.output / 'report.json'
        if report_path.exists():
            final_report = json.loads(report_path.read_text())
            final_report['total_wall_seconds'] = time.monotonic() - started
            write_json(report_path, final_report)
        if policy is not None and hasattr(policy, 'close'):
            policy.close()


if __name__ == '__main__':
    main()
