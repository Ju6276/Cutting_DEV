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
from cutting_world_model.episode_log import ended_metrics, episode_row
from cutting_world_model.policy import SliceItPolicy, SliceItCritic
from cutting_world_model.provenance import git_revision
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.lifecycle import finalize_training
from cutting_world_model.task_heads import FORCE_PENALTIES, reward_settings, without_reward_settings
from cutting_world_model.trainer import ACTOR_OBJECTIVES, CRITIC_TARGETS, OnlineLearner
from cutting_world_model.tracking import ExperimentTracker
from cutting_world_model.world_model import TERMINAL_GRADIENTS, CuttingWorldModel
from third_party.supermimic.schedule import cosine_lr
from train_coupled import write_json, append_json, learning_status


def finalize_gpu_run(checkpoint, progress, tracker, output, status, iteration, updates):
    return finalize_training(
        save=lambda: checkpoint('last.pt'),
        record=lambda: progress(iteration=iteration, updates=updates),
        finish=lambda: tracker.finish(output / 'last.pt', status),
        close=lambda: None,
        state=status)


def without_substeps(metadata):
    """Task metadata with the object groups' DiSECt substeps removed: a numerical-stability
    setting of the material integrator, not a change of the task."""
    conditions = metadata.get('task_conditions')
    if conditions is None:
        return metadata
    conditions = {k: v for k, v in conditions.items() if k != 'sim_dt_note'}
    conditions['groups'] = [{k: v for k, v in g.items() if k != 'sim_dt'} for g in conditions['groups']]
    return {**metadata, 'task_conditions': conditions}


def restore_replay(replay, checkpoint_path, saved):
    """Load the replay snapshot written with ``checkpoint_path``. Episodes in progress at the
    checkpoint are closed: the environment restarts every world, so no training window may
    join a pre-resume step to a post-resume step."""
    path = Path(checkpoint_path).parent / 'replay.pt'
    if not path.exists():
        raise FileNotFoundError(f'{path} missing; use --resume-without-replay to start empty')
    snapshot = torch.load(path, map_location='cpu', weights_only=False)
    if snapshot.get('checkpoint') != Path(checkpoint_path).name or snapshot.get('iteration') != saved['iteration']:
        raise ValueError(f"{path} belongs to {snapshot.get('checkpoint')} at iteration "
                         f"{snapshot.get('iteration')}, not {Path(checkpoint_path).name} at {saved['iteration']}")
    replay.load_state_dict(snapshot)
    replay.buffer.end_episode(replay.ids)
    replay.frame.zero_()
    replay.episode += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-root', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'backends/mujoco')
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--microbatch', type=int, default=64)
    parser.add_argument('--wm-only-updates', type=int, default=500,
                        help='Completed world-model rounds before policy/critic updates')
    parser.add_argument('--max-hours', type=float, default=6.)
    parser.add_argument('--max-iterations', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--allow-substep-change', action='store_true',
                        help='With --resume: accept task conditions that differ from the checkpoint '
                             'only in object-group DiSECt substeps (sim_dt); recorded in the manifest')
    parser.add_argument('--resume-without-replay', action='store_true',
                        help='With --resume: start from an empty replay instead of the replay.pt '
                             'snapshot written with the checkpoint')
    parser.add_argument('--wandb-project', default='sliceit-supermimic')
    parser.add_argument('--wandb-entity')
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online')
    parser.add_argument('--critic-target', choices=CRITIC_TARGETS, default='imagined',
                        help="'imagined': source GAE targets on imagined rollouts; "
                             "'real': GAE targets on recorded replay windows; "
                             "'real_wm': also on WM open-loop predictions of those states")
    parser.add_argument('--actor-horizon', type=int,
                        help='Imagination/BPTT steps of the actor objective (default: source 32)')
    parser.add_argument('--actor-bc-alpha', type=float,
                        help='TD3+BC alpha for actor regularization toward recorded actions (default: off)')
    parser.add_argument('--critic-reset', action='store_true',
                        help="With --resume and --critic-target real: discard the saved critic")
    parser.add_argument('--critic-prefit-rounds', type=int, default=0,
                        help='Real-target critic updates on refilled replay before the next policy update')
    parser.add_argument('--critic-microbatch', type=int,
                        help='Storage chunk for real-target critic minibatches (default: --microbatch)')
    parser.add_argument('--actor-objective', choices=ACTOR_OBJECTIVES, default='model',
                        help="'model': imagined-objective gradient (+ optional BC); "
                             "'bc': the BC term alone at the same scale (needs --actor-bc-alpha)")
    parser.add_argument('--wm-terminal-gradient', choices=TERMINAL_GRADIENTS, default='dynamics',
                        help="'dynamics': source, the terminal BCE also trains the dynamics through the "
                             "predicted state; 'head': in WM fitting the BCE trains only the terminal head "
                             "(policy BPTT unchanged). A resume must keep the checkpoint's route")
    parser.add_argument('--change-wm-terminal-gradient', action='store_true',
                        help='With --resume: start --wm-terminal-gradient from a checkpoint written '
                             'with the other route (e.g. variant A from policy62/WM562)')
    parser.add_argument('--risk-constraint', action='store_true',
                        help='route B: quantile force-risk head and Lagrangian risk penalty in BPTT')
    parser.add_argument('--risk-estimator', choices=('quantile', 'ensemble'), default='quantile',
                        help='quantile: proposed risk head; ensemble: precedent comparator (mean + 2 std of 5 WMs)')
    parser.add_argument('--risk-threshold', type=float, default=.9)
    parser.add_argument('--risk-target-rate', type=float, default=.02,
                        help='target fraction of ended episodes that hit the force limit')
    parser.add_argument('--risk-lambda-init', type=float, default=10.)
    parser.add_argument('--risk-lambda-lr', type=float, default=50.)
    parser.add_argument('--checkpoint-every', type=int, default=0,
                        help='also save checkpoint_<iteration>.pt (and the replay) every N iterations; 0: off')
    parser.add_argument('--conditions', type=Path,
                        help='multi-condition task JSON (object groups, materials, start clearance, '
                             'target speeds, w_speed); worlds = sum of group worlds')
    parser.add_argument('--force-penalty', choices=FORCE_PENALTIES, default='sliceit',
                        help="'sliceit': source -sigmoid(s/2-3); 'linear': -s/--force-penalty-scale")
    parser.add_argument('--force-penalty-scale', type=float,
                        help='Force (N) at which the linear penalty reaches -1 (required for linear)')
    args = parser.parse_args()
    if args.change_wm_terminal_gradient and not args.resume:
        parser.error('--change-wm-terminal-gradient requires --resume')
    if args.actor_objective == 'bc' and args.actor_bc_alpha is None:
        parser.error("--actor-objective bc requires --actor-bc-alpha")
    if (args.force_penalty == 'linear') != (args.force_penalty_scale is not None):
        parser.error('--force-penalty-scale is required for, and only for, the linear penalty')
    if args.wm_only_updates < 0:
        parser.error("--wm-only-updates must be nonnegative")
    if args.critic_reset and (args.critic_target == 'imagined' or not args.resume):
        parser.error('--critic-reset requires --resume and a real-data --critic-target')
    if args.critic_prefit_rounds < 0 or (args.critic_reset and args.critic_prefit_rounds < 1):
        parser.error('--critic-reset requires a positive --critic-prefit-rounds')
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
    reward = ({'force_penalty': 'linear', 'force_penalty_scale': args.force_penalty_scale}
              if args.force_penalty == 'linear' else None)
    conditions = None
    if args.conditions is not None:
        from cutting_world_model.multi_env import MultiConditionEnvironment
        conditions = json.loads(args.conditions.read_text())
        if conditions.get('w_speed'):
            reward = {**(reward or {}), 'w_speed': float(conditions['w_speed'])}
        env = MultiConditionEnvironment.from_config(args.backend_root, conditions, reward=reward)
        args.worlds = env.worlds
        status['worlds'] = env.worlds
        task_goal = MultiConditionEnvironment.goal
    else:
        env = GPUEnvironment(args.backend_root, args.worlds, reward=reward)
        task_goal = lambda f: f['target_pose'].float()  # noqa: E731
    try:
        return run_with_environment(args, output, status, started, env, conditions, task_goal)
    finally:
        if hasattr(env, 'close'):
            env.close()


def run_with_environment(args, output, status, started, env, conditions, task_goal):
    device = 'cuda:0'
    frame = env.current
    one = {k: v[0].cpu().numpy() for k, v in frame.items()}
    metadata = ControlStateSpec.metadata_from_frame(env.metadata, one)
    if conditions is not None:
        metadata['goal_dim'] = 8
        metadata['task_conditions'] = conditions
    goal_dim = int(metadata.get('goal_dim', 7))
    write_json(output / 'task.json', metadata)
    wm = CuttingWorldModel(metadata).to(device)
    spec = wm.spec
    policy = SliceItPolicy(goal_dim).to(device)
    critic = SliceItCritic(spec.state_dim, goal_dim).to(device)
    learner = OnlineLearner(wm, policy, critic, args.microbatch,
                            wm_only_updates=args.wm_only_updates,
                            critic_target=args.critic_target,
                            critic_microbatch=args.critic_microbatch,
                            actor_horizon=args.actor_horizon,
                            actor_bc_alpha=args.actor_bc_alpha,
                            actor_objective=args.actor_objective,
                            wm_terminal_gradient=args.wm_terminal_gradient,
                            risk_constraint=({'estimator': args.risk_estimator,
                                              'threshold': args.risk_threshold,
                                              'target_rate': args.risk_target_rate,
                                              'lambda_init': args.risk_lambda_init,
                                              'lambda_lr': args.risk_lambda_lr}
                                             if args.risk_constraint else None))
    schedule = OnlineSchedule(worlds=args.worlds)
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        compare = without_substeps if args.allow_substep_change else (lambda m: m)
        if compare(without_reward_settings(saved['metadata'])) != compare(without_reward_settings(metadata)):
            raise ValueError('Checkpoint and physical task metadata differ')
        # The critic estimates returns of the reward it was fitted on; a reward change
        # keeps the WM and policy but requires a fresh critic.
        if (reward_settings(saved['metadata']['environment'])
                != reward_settings(metadata['environment']) and not args.critic_reset):
            raise ValueError('Checkpoint reward differs; a reward change requires --critic-reset')
        learner.restore(saved['learner'], reset_critic=args.critic_reset,
                        change_wm_terminal_gradient=args.change_wm_terminal_gradient)
        if learner.critic_prefit_pending and args.critic_prefit_rounds < 1:
            raise ValueError('Checkpoint has a pending critic pre-fit; set --critic-prefit-rounds')
        schedule = OnlineSchedule.resume(saved.get('schedule', {}), worlds=args.worlds)
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    replay = BatchedOnlineReplay(args.worlds, spec.state_dim, device,
                                 capacity=schedule.replay_capacity, goal_dim=goal_dim)
    if args.resume and not args.resume_without_replay:
        restore_replay(replay, args.resume, saved)
    history = frame['observation'].float()[None].repeat(policy.history, 1, 1)
    state = spec.pack_batch(frame).float()
    returns = torch.zeros(args.worlds, device=device)
    lengths = torch.zeros(args.worlds, device=device, dtype=torch.long)
    peak_material = torch.zeros(args.worlds, device=device)
    peak_ratio = torch.zeros(args.worlds, device=device)
    episodes = torch.zeros_like(lengths)
    revisions = {'world_model': git_revision(Path(__file__).resolve().parents[1]),
                 'backend': git_revision(args.backend_root)}
    manifest = {
        'git': revisions, 'command': sys.argv,
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'world_model': asdict(learner.wcfg), 'policy': asdict(learner.pcfg),
        'schedule': asdict(schedule.config), 'training_stage': learning_status(learner),
        'endpoint': 'wall budget or iteration budget',
        'timeout': 'SIGTERM saves last.pt and progress.json',
        'state_dim': spec.state_dim, 'delta_dim': spec.delta_dim,
        'replay_capacity_per_world': schedule.replay_capacity,
        'critic_variant': learner.critic_variant(),
        'critic_microbatch': learner.critic_microbatch, 'actor_horizon': learner.actor_horizon,
        'actor_bc_alpha': learner.actor_bc_alpha, 'actor_objective': learner.actor_objective,
        'wm_terminal_gradient': learner.wm.terminal_gradient,
        'wm_terminal_gradient_changed_at_resume': args.change_wm_terminal_gradient,
        'reward': reward_settings(metadata['environment']) or 'sliceit',
        'replay_snapshot': 'replay.pt rewritten with every checkpoint',
    }
    write_json(output / 'manifest.json', manifest)
    tracker = ExperimentTracker(output, manifest, args.wandb_project, args.wandb_entity, args.wandb_mode)
    stop_requested = False
    save_requested = False

    def stop(signum, context):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def request_checkpoint(signum, context):
        nonlocal save_requested
        save_requested = True

    signal.signal(signal.SIGUSR1, request_checkpoint)

    def progress(**values):
        status.update(values, **learning_status(learner), transitions=schedule.transitions,
                      wall_seconds=time.monotonic() - started)
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
        replay.save(output / 'replay.pt', metadata=metadata, iteration=schedule.iteration,
                    transitions=schedule.transitions, checkpoint=name)

    risk_window = [0, 0]  # force-limit and all ended episodes since the last dual update
    physical_limit = (None if args.max_iterations is None else
                      schedule.iteration + args.max_iterations)
    status['state'] = 'running'
    timing = {'physics': 0., 'world_model': 0., 'policy': 0.}
    error = None
    try:
        while schedule.iteration < schedule.config.total_iterations:
            if physical_limit is not None and schedule.iteration >= physical_limit:
                status['state'] = 'completed'
                break
            if stop_requested or time.monotonic() - started >= args.max_hours * 3600:
                status['state'] = 'interrupted' if stop_requested else 'budget_complete'
                break
            with torch.no_grad():
                goal = task_goal(frame)
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
                peak_material = torch.maximum(peak_material, info['peak_force'].float())
                peak_ratio = torch.maximum(peak_ratio, info['safety_margin'].float())
            ended = terminated | truncated
            risk_window[0] += int((ended & info['collision']).sum())
            risk_window[1] += int(ended.sum())
            metrics = {'iteration': schedule.iteration + 1, 'physics/reward_mean': float(reward.mean())}
            metrics.update(ended_metrics(ended, returns, lengths, info, truncated, peak_material, peak_ratio,
                                         env.groups if conditions is not None else None,
                                         env.group_index if conditions is not None else None))
            for e in ended.nonzero().flatten().tolist():
                append_json(output / 'episodes.jsonl', episode_row(
                    e, int(episodes[e]), schedule.iteration + 1, returns, lengths, info, truncated,
                    peak_material, peak_ratio, env.conditions([e])[0] if conditions is not None else None))
            schedule.after_environment_step()
            if ended.any():
                frame = env.reset(ended)
                state = spec.reset_batch_history(frame, state, ended)
                history[:, ended] = frame['observation'][ended].float()[None]
                peak_material[ended] = 0
                peak_ratio[ended] = 0
                returns[ended] = 0
                lengths[ended] = 0
                episodes[ended] += 1
            progress(iteration=schedule.iteration,
                     episodes=int(episodes.sum()), replay_rows=replay.buffer.size)
            for update_step in schedule.update_iterations:
                lr = cosine_lr(update_step, schedule.config.total_iterations,
                               learner.wcfg.lr, schedule.config.lr_final_ratio, schedule.config.lr_decay_fraction)
                learner.set_lr(lr)
                tick = time.monotonic()
                wm_metrics = learner.fit_world_model(replay.buffer)
                if learner.risk_head is not None:
                    wm_metrics.update(learner.fit_risk_head(replay.buffer))
                    learner.update_risk_lambda(*risk_window)
                    risk_window = [0, 0]
                timing['world_model'] += time.monotonic() - tick
                policy_metrics = {}
                if learner.should_update_policy and learner.critic_prefit_pending:
                    tick = time.monotonic()
                    for index, row in enumerate(learner.prefit_critic(replay.buffer,
                                                                      args.critic_prefit_rounds)):
                        append_json(output / 'critic_prefit.jsonl', {'round': index + 1, **row})
                        tracker.log({'critic_prefit/' + k: v for k, v in row.items()})
                    timing['critic_prefit'] = time.monotonic() - tick
                    checkpoint('critic_prefit_complete.pt')
                if learner.should_update_policy:
                    tick = time.monotonic()
                    policy_metrics = learner.improve_policy(replay.buffer)
                    timing['policy'] += time.monotonic() - tick
                metrics.update({'wm/' + k: v for k, v in wm_metrics.items()})
                metrics.update({'policy/' + k: v for k, v in policy_metrics.items()})
                metrics['lr'] = lr
                append_json(output / 'updates.jsonl', {
                    'iteration': schedule.iteration, 'update_iteration': update_step,
                    'transitions': schedule.transitions, 'update': learner.updates, 'lr': lr,
                    **learning_status(learner),
                    'world_model': wm_metrics, 'policy': policy_metrics, 'timing': dict(timing),
                    'peak_gpu_mib': torch.cuda.max_memory_allocated() / 2**20,
                })
                if learner.wm_only_updates and learner.wm_updates == learner.wm_only_updates:
                    checkpoint('wm_only_complete.pt')
                elif learner.wm_only_updates and learner.updates == 1:
                    checkpoint('first_joint_update.pt')
            progress(iteration=schedule.iteration, updates=learner.updates,
                     episodes=int(episodes.sum()),
                     replay_rows=replay.buffer.size, timing=dict(timing))
            metrics.update({'updates': learner.updates,
                            'learning/wm_updates': learner.wm_updates,
                            'learning/policy_updates': learner.updates,
                            'learning/stage': learner.stage,
                            'learning/wm_only_updates': learner.wm_only_updates,
                            'physics/transitions': schedule.transitions,
                            'episodes': int(episodes.sum()), 'replay/rows': replay.buffer.size,
                            'gpu/torch_peak_mib': torch.cuda.max_memory_allocated() / 2**20,
                            'wall_seconds': time.monotonic() - started})
            metrics.update({'time/' + k: v for k, v in timing.items()})
            metrics.update({'optimizer/' + k + '_steps': v for k, v in learner.optimizer_steps.items()})
            tracker.log(metrics)
            append_json(output / 'steps.jsonl', dict(status))
            print(json.dumps(status), flush=True)
            if args.checkpoint_every and schedule.iteration % args.checkpoint_every == 0:
                save_requested = True
            if save_requested:
                checkpoint('checkpoint_%08d.pt' % schedule.iteration)
                save_requested = False
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
