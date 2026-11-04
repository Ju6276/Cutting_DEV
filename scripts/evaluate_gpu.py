#!/usr/bin/env python3
"""Evaluate a frozen policy in the physical UR5e/DiSECt task."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.policy import SliceItPolicy
from cutting_world_model.provenance import git_revision
from train_coupled import write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-root', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'backends/mujoco')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=1800.)
    args = parser.parse_args()
    if not 0 < args.max_seconds < float('inf'):
        parser.error('--max-seconds must be finite and positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    sys.path.insert(0, str(args.backend_root.resolve()))
    from cutting_world_model.gpu_env import GPUEnvironment

    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    env = GPUEnvironment(args.backend_root, 1)
    frame = env.current
    metadata = ControlStateSpec.metadata_from_frame(
        env.metadata, {k: v[0].cpu().numpy() for k, v in frame.items()})
    if metadata != saved['metadata']:
        raise ValueError('Checkpoint and physical task metadata differ')
    spec = ControlStateSpec(metadata).to(env.device)
    policy = SliceItPolicy().to(env.device)
    policy.load_state_dict(saved['learner']['policy'])
    policy.eval()
    history = frame['observation'].float()[None].repeat(policy.history, 1, 1)
    write_json(output / 'manifest.json', {
        'git': git_revision(Path(__file__).resolve().parents[1]),
        'backend_git': git_revision(args.backend_root),
        'command': sys.argv, 'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_iteration': saved['iteration'],
        'policy_updates': saved['learner']['updates'],
        'worlds': 1, 'action_noise_std': 0.,
        'wall_budget_seconds': args.max_seconds,
        'endpoint': 'One deterministic physical episode or wall budget',
    })
    total_return = 0.
    peak_force = 0.
    report = {'state': 'running'}
    try:
        for step in range(env.environment['steps_per_episode']):
            if time.monotonic() - started >= args.max_seconds:
                report['state'] = 'budget_complete'
                break
            action = spec.execute(policy(history, frame['target_pose'].float()))
            frame, reward, terminated, truncated, info = env.step(action)
            history = torch.cat((history[1:], frame['observation'].float()[None]), 0)
            total_return += float(reward[0])
            peak_force = max(peak_force, float(info['peak_force'][0]))
            report.update(steps=step + 1, return_value=total_return,
                          simulation_seconds=(step + 1) * env.environment['agent_control_dt'],
                          peak_force=peak_force, success=bool(info['success'][0]),
                          collision=bool(info['collision'][0]), truncated=bool(truncated[0]),
                          released_springs=int(info['released_springs'][0]),
                          wall_seconds=time.monotonic() - started)
            if bool(terminated[0] | truncated[0]):
                report['state'] = 'completed'
            write_json(output / 'progress.json', report)
            if step % 20 == 0 or report['state'] == 'completed':
                print(json.dumps(report), flush=True)
            if report['state'] == 'completed':
                break
        unchanged = all(torch.equal(value.cpu(), saved['learner']['policy'][name])
                        for name, value in policy.state_dict().items())
        report.update(policy_unchanged=unchanged, optimizer_updates=0)
        if not unchanged:
            raise RuntimeError('Evaluation changed policy weights')
    except BaseException as error:
        report.update(state='failed', error=repr(error))
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - started
        write_json(output / 'report.json', report)
        write_json(output / 'progress.json', report)


if __name__ == '__main__':
    main()
