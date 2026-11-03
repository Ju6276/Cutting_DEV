#!/usr/bin/env python3
"""Online source SuperMimic updates with packed ROS-free cutting environments."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutting_world_model.online_replay import BatchedOnlineReplay
from cutting_world_model.online_schedule import OnlineSchedule
from cutting_world_model.policy import SliceItPolicy, SliceItCritic
from cutting_world_model.provenance import git_revision
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.lifecycle import finalize_training
from cutting_world_model.trainer import OnlineLearner
from cutting_world_model.tracking import ExperimentTracker
from cutting_world_model.world_model import CuttingWorldModel
from third_party.supermimic.schedule import cosine_lr
from train_coupled import write_json, append_json


def finalize_gpu_run(checkpoint, progress, tracker, output, status, iteration, updates):
    return finalize_training(
        save=lambda: checkpoint('last.pt'),
        record=lambda: progress(iteration=iteration, updates=updates),
        finish=lambda: tracker.finish(output / 'last.pt', status),
        close=lambda: None,
        state=status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-root', type=Path, required=True)
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--microbatch', type=int, default=64)
    parser.add_argument('--max-hours', type=float, default=6.)
    parser.add_argument('--max-iterations', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--wandb-project', default='sliceit-supermimic')
    parser.add_argument('--wandb-entity')
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'state': 'initializing', 'pid': os.getpid(), 'worlds': args.worlds}
    write_json(output / 'progress.json', status)
    try:
        run(args, output, status, started)
    except BaseException as error:
        if status['state'] == 'initializing':
            status.update(state='failed', error=repr(error),
                          wall_seconds=time.monotonic() - started)
            write_json(output / 'progress.json', status)
        raise


def run(args, output, status, started):
    sys.path.insert(0, str(args.backend_root.resolve()))
    from cutting_world_model.gpu_env import GPUEnvironment
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = 'cuda:0'
    env = GPUEnvironment(args.backend_root, args.worlds)
    frame = env.current
    one = {k: v[0].cpu().numpy() for k, v in frame.items()}
    metadata = ControlStateSpec.metadata_from_frame(env.metadata, one)
    write_json(output / 'task.json', metadata)
    wm = CuttingWorldModel(metadata).to(device)
    spec = wm.spec
    policy = SliceItPolicy().to(device)
    critic = SliceItCritic(spec.state_dim).to(device)
    learner = OnlineLearner(wm, policy, critic, args.microbatch)
    schedule = OnlineSchedule(worlds=args.worlds)
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved['metadata'] != metadata:
            raise ValueError('Checkpoint and physical task metadata differ')
        learner.restore(saved['learner'])
        schedule = OnlineSchedule.resume(saved.get('schedule', {}), worlds=args.worlds)
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    replay = BatchedOnlineReplay(args.worlds, spec.state_dim, device,
                                 capacity=schedule.replay_capacity)
    history = frame['observation'].float()[None].repeat(policy.history, 1, 1)
    state = spec.pack_batch(frame).float()
    returns = torch.zeros(args.worlds, device=device)
    lengths = torch.zeros(args.worlds, device=device, dtype=torch.long)
    episodes = torch.zeros_like(lengths)
    revisions = {'world_model': git_revision(Path(__file__).resolve().parents[1]),
                 'backend': git_revision(args.backend_root)}
    manifest = {
        'git': revisions, 'command': sys.argv,
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'world_model': asdict(learner.wcfg), 'policy': asdict(learner.pcfg),
        'schedule': asdict(schedule.config), 'endpoint': 'wall budget or iteration budget',
        'timeout': 'SIGTERM saves last.pt and progress.json',
        'state_dim': spec.state_dim, 'delta_dim': spec.delta_dim,
        'replay_capacity_per_world': schedule.replay_capacity,
    }
    write_json(output / 'manifest.json', manifest)
    tracker = ExperimentTracker(output, manifest, args.wandb_project, args.wandb_entity, args.wandb_mode)
    stop_requested = False

    def stop(signum, context):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def progress(**values):
        status.update(values, source_iteration=schedule.source_iteration,
                      transitions=schedule.transitions, wall_seconds=time.monotonic() - started)
        write_json(output / 'progress.json', status)

    def learner_progress(**values):
        if stop_requested:
            raise InterruptedError('Training interrupted')
        if time.monotonic() - started >= args.max_hours * 3600:
            raise TimeoutError('Training time budget reached')
        progress(**values)

    learner.progress = learner_progress

    def checkpoint(name):
        temp = output / (name + '.tmp')
        torch.save({'learner': learner.checkpoint(), 'metadata': metadata,
                    'iteration': schedule.iteration, 'schedule': schedule.state_dict(),
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state(), 'git': revisions}, temp)
        temp.replace(output / name)

    physical_limit = (None if args.max_iterations is None else
                      schedule.iteration + args.max_iterations)
    status['state'] = 'running'
    timing = {'physics': 0., 'world_model': 0., 'policy': 0.}
    error = None
    try:
        while schedule.source_iteration < schedule.config.total_iterations:
            if physical_limit is not None and schedule.iteration >= physical_limit:
                status['state'] = 'completed'
                break
            if stop_requested or time.monotonic() - started >= args.max_hours * 3600:
                status['state'] = 'interrupted' if stop_requested else 'budget_complete'
                break
            with torch.no_grad():
                goal = frame['target_pose'].float()
                clean = policy(history, goal)
                executed = spec.execute(clean + .1 * torch.randn_like(clean))
            progress(phase='physics', iteration=schedule.iteration, updates=learner.updates)
            tick = time.monotonic()
            frame, reward, terminated, truncated, info = env.step(executed)
            timing['physics'] += time.monotonic() - tick
            with torch.no_grad():
                next_state = spec.pack_batch_transition(frame, state)
                replay.add(state, clean, executed, goal, reward, next_state, terminated, truncated)
                state = next_state
                history = torch.cat((history[1:], frame['observation'].float()[None]), 0)
                returns += reward
                lengths += 1
            ended = terminated | truncated
            metrics = {'iteration': schedule.iteration + 1, 'physics/reward_mean': float(reward.mean())}
            if ended.any():
                metrics.update({'episode/return_mean': float(returns[ended].mean()),
                                'episode/length_mean': float(lengths[ended].float().mean()),
                                'episode/success_rate': float(info['success'][ended].float().mean()),
                                'episode/collision_rate': float(info['collision'][ended].float().mean())})
            for e in ended.nonzero().flatten().tolist():
                append_json(output / 'episodes.jsonl', {
                    'env': e, 'episode': int(episodes[e]), 'iteration': schedule.iteration + 1,
                    'return': float(returns[e]), 'steps': int(lengths[e]),
                    'success': bool(info['success'][e]), 'collision': bool(info['collision'][e]),
                    'truncated': bool(truncated[e]), 'released_springs': int(info['released_springs'][e]),
                })
            schedule.after_environment_step()
            if ended.any():
                frame = env.reset(ended)
                state = spec.reset_batch_history(frame, state, ended)
                history[:, ended] = frame['observation'][ended].float()[None]
                returns[ended] = 0
                lengths[ended] = 0
                episodes[ended] += 1
            progress(iteration=schedule.iteration,
                     episodes=int(episodes.sum()), replay_rows=replay.buffer.size)
            for source_step in schedule.update_source_iterations:
                lr = cosine_lr(source_step, schedule.config.total_iterations,
                               learner.wcfg.lr, schedule.config.lr_final_ratio, schedule.config.lr_decay_fraction)
                learner.set_lr(lr)
                tick = time.monotonic()
                wm_metrics = learner.fit_world_model(replay.buffer)
                timing['world_model'] += time.monotonic() - tick
                tick = time.monotonic()
                policy_metrics = learner.improve_policy(replay.buffer)
                timing['policy'] += time.monotonic() - tick
                metrics.update({'wm/' + k: v for k, v in wm_metrics.items()})
                metrics.update({'policy/' + k: v for k, v in policy_metrics.items()})
                metrics['lr'] = lr
                append_json(output / 'updates.jsonl', {
                    'iteration': schedule.iteration, 'source_iteration': source_step,
                    'transitions': schedule.transitions, 'update': learner.updates, 'lr': lr,
                    'world_model': wm_metrics, 'policy': policy_metrics, 'timing': dict(timing),
                    'peak_gpu_mib': torch.cuda.max_memory_allocated() / 2**20,
                })
            progress(iteration=schedule.iteration, updates=learner.updates,
                     episodes=int(episodes.sum()),
                     replay_rows=replay.buffer.size, timing=dict(timing))
            metrics.update({'updates': learner.updates, 'source_iteration': schedule.source_iteration,
                            'physics/transitions': schedule.transitions,
                            'episodes': int(episodes.sum()), 'replay/rows': replay.buffer.size,
                            'gpu/torch_peak_mib': torch.cuda.max_memory_allocated() / 2**20,
                            'wall_seconds': time.monotonic() - started})
            metrics.update({'time/' + k: v for k, v in timing.items()})
            tracker.log(metrics)
            append_json(output / 'steps.jsonl', dict(status))
            print(json.dumps(status), flush=True)
            if schedule.checkpoint_due:
                checkpoint('checkpoint_%08d.pt' % schedule.source_iteration)
        else:
            status['state'] = 'completed'
    except TimeoutError:
        status['state'] = 'budget_complete'
    except InterruptedError:
        status['state'] = 'interrupted'
    except BaseException as caught:
        error = caught
        status.update(state='failed', error=repr(caught))
        raise
    finally:
        errors = finalize_gpu_run(checkpoint, progress, tracker, output, status,
                                  schedule.iteration, learner.updates)
        if errors and error is None:
            raise RuntimeError('Training finalization failed: ' + repr(errors))


if __name__ == '__main__':
    main()
