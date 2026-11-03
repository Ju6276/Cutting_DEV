#!/usr/bin/env python3
"""Online SuperMimic-style learning in the unchanged SliceIt Gazebo/DiSECt task."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutting_world_model.coupled_env import CoupledEnvironment
from cutting_world_model.kinematics import pose_difference
from cutting_world_model.online_replay import OnlineReplay
from cutting_world_model.online_schedule import OnlineSchedule
from cutting_world_model.policy import SliceItPolicy, SliceItCritic
from cutting_world_model.provenance import git_revision
from cutting_world_model.state_spec import CuttingStateSpec
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.trainer import OnlineLearner
from cutting_world_model.world_model import CuttingWorldModel
from third_party.supermimic.schedule import cosine_lr


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def append_json(path, data):
    with path.open('a') as output:
        output.write(json.dumps(data) + '\n')


def save_episode(path, frames, actions, clean_actions, rewards, terminated, truncated):
    data = {key: np.asarray([frame[key] for frame in frames]) for key in frames[0]}
    data.update(actions=np.asarray(actions), clean_actions=np.asarray(clean_actions),
                rewards=np.asarray(rewards), terminated=np.asarray(terminated), truncated=np.asarray(truncated))
    np.savez_compressed(path, **data)


def main():
    import rospy
    parser = argparse.ArgumentParser()
    parser.add_argument('--sliceit-root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--microbatch', type=int, default=64)
    parser.add_argument('--max-hours', type=float, default=6.)
    parser.add_argument('--max-iterations', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume')
    parser.add_argument('--model-state', choices=('native', 'control-history'), default='control-history')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval-episodes', type=int, default=5)
    args = parser.parse_args(rospy.myargv()[1:])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.evaluate and not args.resume:
        parser.error('--evaluate requires --resume')
    rospy.init_node('world_model_online', anonymous=True, disable_signals=True)

    def interrupt_setup(signum, frame):
        raise InterruptedError('Training setup interrupted')

    signal.signal(signal.SIGTERM, interrupt_setup)
    signal.signal(signal.SIGINT, interrupt_setup)
    env = CoupledEnvironment(args.config)
    try:
        run(args, env, output)
    finally:
        env.close()


def run(args, env, output):
    frame = env.reset()
    spec_class = ControlStateSpec if args.model_state == 'control-history' else CuttingStateSpec
    metadata = spec_class.metadata_from_frame(env.metadata, frame)
    write_json(output / 'task.json', metadata)
    wm = CuttingWorldModel(metadata).to(args.device)
    spec = wm.spec
    policy = SliceItPolicy().to(args.device)
    critic = SliceItCritic(spec.state_dim).to(args.device)
    learner = OnlineLearner(wm, policy, critic, args.microbatch)
    schedule = OnlineSchedule(worlds=1)
    if args.resume:
        saved = torch.load(args.resume, map_location=args.device, weights_only=False)
        if saved['metadata'] != metadata:
            raise ValueError('Resume task/geometry differs from checkpoint')
        learner.restore(saved['learner'])
        if not args.evaluate:
            schedule = OnlineSchedule.resume(saved.get('schedule', {}), worlds=1)
        torch.set_rng_state(saved['torch_rng'].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    replay = (None if args.evaluate else
              OnlineReplay(spec.state_dim, args.device, capacity=schedule.replay_capacity))
    start = time.monotonic()
    stop_requested = False

    def stop(signum, context):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    manifest = {'git': git_revision(Path(__file__).resolve().parents[1]),
                'sliceit_git': git_revision(args.sliceit_root), 'command': sys.argv,
                'arguments': vars(args), 'world_model': asdict(learner.wcfg),
                'policy': asdict(learner.pcfg), 'schedule': asdict(schedule.config),
                'schedule_units': schedule.state_dict(),
                'environment': env.environment, 'controller': env.controller,
                'feedback_graph': env.feedback_graph}
    write_json(output / 'manifest.json', manifest)
    status = {'state': 'running', 'iteration': schedule.iteration, 'updates': learner.updates}

    def checkpoint_model(name):
        path = output / name
        temporary = path.with_suffix('.tmp')
        torch.save({'learner': learner.checkpoint(), 'metadata': metadata,
                    'iteration': schedule.iteration, 'schedule': schedule.state_dict(),
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state() if torch.cuda.is_available() else torch.get_rng_state(),
                    'git': git_revision(Path(__file__).resolve().parents[1])}, temporary)
        temporary.replace(path)

    def learner_progress(**values):
        if stop_requested:
            raise InterruptedError('Training stop requested')
        if time.monotonic() - start >= args.max_hours * 3600:
            raise TimeoutError('Training time budget reached')
        status.update(values, wall_seconds=time.monotonic() - start)
        write_json(output / 'progress.json', status)

    learner.progress = learner_progress
    episode = 0
    frames, actions, clean_actions, rewards, terms, truncs = [frame], [], [], [], [], []
    history = [frame['observation'].copy() for _ in range(policy.history)]
    state = spec.pack(frame, args.device)
    episode_start = time.monotonic()
    wm_seconds, policy_seconds, physics_seconds = 0., 0., 0.
    executed_steps = 0
    try:
        while schedule.iteration < schedule.config.total_iterations:
            if args.max_iterations is not None and executed_steps >= args.max_iterations:
                status['state'] = 'completed'
                break
            if stop_requested or time.monotonic() - start >= args.max_hours * 3600:
                status['state'] = 'budget_complete' if not stop_requested else 'interrupted'
                break
            with torch.no_grad():
                goal = torch.as_tensor(frame['target_pose'], dtype=torch.float32, device=args.device)
                obs_history = torch.as_tensor(np.asarray(history), dtype=torch.float32, device=args.device)[:, None]
                clean = policy(obs_history, goal[None])[0]
                executed = spec.execute(clean if args.evaluate else clean + .1 * torch.randn_like(clean))
            tick = time.monotonic()
            next_frame, reward, terminated, truncated, info = env.step(executed.cpu().numpy())
            executed_steps += 1
            physics_seconds += time.monotonic() - tick
            with torch.no_grad():
                next_state = spec.pack_transition(next_frame, state)
            if not args.evaluate:
                replay.add(state, clean, executed, goal, reward, next_state, terminated, truncated)
            frames.append(next_frame)
            actions.append(executed.cpu().numpy())
            clean_actions.append(clean.cpu().numpy())
            rewards.append(reward)
            terms.append(terminated)
            truncs.append(truncated)
            history = history[1:] + [next_frame['observation'].copy()]
            frame = next_frame
            state = next_state
            updates = schedule.after_environment_step() if not args.evaluate else 0
            status.update(iteration=schedule.iteration, transitions=schedule.transitions,
                          episode=episode, episode_step=len(actions),
                          last_reward=reward, episode_return=float(sum(rewards)),
                          wall_seconds=time.monotonic() - start, phase='physics',
                          updates=learner.updates)
            write_json(output / 'progress.json', status)
            append_json(output / 'steps.jsonl', dict(status))
            print(json.dumps(status), flush=True)
            if terminated or truncated:
                save_episode(output / ('episode_%05d.npz' % episode), frames, actions, clean_actions, rewards, terms, truncs)
                append_json(output / 'episodes.jsonl', {
                    'episode': episode, 'iteration': schedule.iteration, 'return': float(sum(rewards)),
                    'steps': len(actions), 'success': bool(info.get('success', False)),
                    'collision': bool(info.get('collision', False)), 'truncated': truncated,
                    'wall_seconds': time.monotonic() - episode_start,
                    'released_springs': int(np.count_nonzero(frame['cut_spring_ke'] == 0)),
                })
                episode += 1
                if args.evaluate and episode >= args.eval_episodes:
                    status['state'] = 'completed'
                    break
                frame = env.reset()
                state = spec.pack(frame, args.device)
                history = [frame['observation'].copy() for _ in range(policy.history)]
                frames, actions, clean_actions, rewards, terms, truncs = [frame], [], [], [], [], []
                episode_start = time.monotonic()
            for update_step in schedule.update_iterations if not args.evaluate else []:
                lr = cosine_lr(update_step, schedule.config.total_iterations,
                               learner.wcfg.lr, schedule.config.lr_final_ratio, schedule.config.lr_decay_fraction)
                learner.set_lr(lr)
                status.update(phase='world_model', wall_seconds=time.monotonic() - start)
                write_json(output / 'progress.json', status)
                tick = time.monotonic()
                wm_metrics = learner.fit_world_model(replay.buffer)
                wm_seconds += time.monotonic() - tick
                status.update(phase='policy', wall_seconds=time.monotonic() - start)
                write_json(output / 'progress.json', status)
                tick = time.monotonic()
                pi_metrics = learner.improve_policy(replay.buffer)
                policy_seconds += time.monotonic() - tick
                record = {'iteration': schedule.iteration, 'update_iteration': update_step,
                          'transitions': schedule.transitions, 'update': learner.updates,
                          'world_model': wm_metrics, 'policy': pi_metrics, 'lr': lr,
                          'wall_seconds': time.monotonic() - start,
                          'timing': {'physics': physics_seconds, 'world_model': wm_seconds, 'policy': policy_seconds},
                          'peak_gpu_mib': torch.cuda.max_memory_allocated() / 2**20}
                append_json(output / 'updates.jsonl', record)
                print(json.dumps(record), flush=True)
            if not args.evaluate and schedule.checkpoint_due:
                checkpoint_model('checkpoint_%08d.pt' % schedule.iteration)
        else:
            status['state'] = 'completed'
    except TimeoutError:
        status['state'] = 'budget_complete'
    except InterruptedError:
        status['state'] = 'interrupted'
    except BaseException as error:
        status.update(state='failed', error=repr(error))
        raise
    finally:
        if not args.evaluate:
            checkpoint_model('last.pt')
        status.update(iteration=schedule.iteration, transitions=schedule.transitions, updates=learner.updates,
                      wall_seconds=time.monotonic() - start)
        write_json(output / 'progress.json', status)


if __name__ == '__main__':
    main()
