#!/usr/bin/env python3
"""Official SuperMimic training loop (runners/supermimic_runner.SuperMimicRunner @ 5efbf11) on the
fixed cutting prism.

Each iteration: the actor (plus N(0, noise) exploration) acts in every world; the pre-step frame
(state, raw action, goal, terminated, truncated) goes into the replay; the state/action/actor
normalizers are updated from the collected frames; then ``Ratio(num_envs / num_batches /
batch_size / horizon * replay_ratio)`` times: one WM update (10 windows of 8 frames x 2048) and
one actor-critic policy update (4096 imagined starts, horizon 32). No WM-only pre-training stage.
Components and every adaptation: cutting_world_model/official/components.py and
docs/OFFICIAL_SUPERMIMIC.md.
"""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import signal
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco'), str(ROOT / 'scripts')]
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.episode_log import ended_metrics, episode_row
from cutting_world_model.official.components import (
    OFFICIAL_COMMIT, OFFICIAL_REPOSITORY, ActorCriticPolicyLearner, ActorCriticPolicyLearnerCfg, Actor,
    Dynamics, Ratio, ReplayBuffer, RunnerCfg, WorldModelEnv, WorldModelLearner, WorldModelLearnerCfg, build_mlp)
from cutting_world_model.provenance import git_revision
from cutting_world_model.tracking import ExperimentTracker
from train_coupled import append_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-root', type=Path, default=ROOT / 'backends/mujoco')
    parser.add_argument('--worlds', type=int, default=1024)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-iterations', type=int, required=True)
    parser.add_argument('--max-hours', type=float, default=48.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint-every', type=int, default=RunnerCfg.save_interval)
    parser.add_argument('--policy-chunk', type=int, default=512, help='memory only: gradient accumulation chunk')
    parser.add_argument('--wandb-project', default='sliceit-supermimic')
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = 'cuda:0'
    sys.path.insert(0, str(args.backend_root.resolve()))
    from cutting_world_model.gpu_env import GPUEnvironment
    env = GPUEnvironment(args.backend_root, args.worlds)          # fixed prism, task limits
    frame = env.current
    metadata = ControlStateSpec.metadata_from_frame(env.metadata, {k: v[0].cpu().numpy() for k, v in frame.items()})
    write_json(args.output / 'task.json', metadata)
    runner_cfg, wm_cfg, policy_cfg = RunnerCfg(), WorldModelLearnerCfg(), ActorCriticPolicyLearnerCfg()
    spec = ControlStateSpec(metadata).to(device)
    dynamics = Dynamics(spec, 6, runner_cfg.dynamics_hidden_dims).to(device)
    actor = Actor(spec.state_dim, 7, 6, runner_cfg.latent_dim, runner_cfg.actor_hidden_dims,
                  runner_cfg.history_length).to(device)
    critic = build_mlp(spec.state_dim + 7, runner_cfg.critic_hidden_dims, 1).to(device)
    schema = {'state': ((spec.state_dim,), torch.float32), 'action': ((6,), torch.float32),
              'goal': ((7,), torch.float32), 'terminated': ((), torch.bool), 'truncated': ((), torch.bool)}
    replay = ReplayBuffer(schema, runner_cfg.buffer_size, args.worlds, device)
    wm_learner = WorldModelLearner(dynamics, wm_cfg)
    wm_env = WorldModelEnv(dynamics, replay, policy_cfg.batch_size, runner_cfg.termination_softness)
    policy_learner = ActorCriticPolicyLearner(actor, critic, wm_env, policy_cfg, chunk=args.policy_chunk)
    ratio = args.worlds / wm_cfg.num_batches / wm_cfg.batch_size / wm_cfg.horizon * runner_cfg.replay_ratio
    should_train = Ratio(ratio)
    manifest = {'git': git_revision(ROOT), 'command': sys.argv, 'official': {'repository': OFFICIAL_REPOSITORY,
                'commit': OFFICIAL_COMMIT}, 'runner': asdict(runner_cfg), 'world_model': asdict(wm_cfg),
                'policy': asdict(policy_cfg), 'worlds': args.worlds, 'updates_per_env_step': ratio,
                'task': 'fixed cutting prism (GPUEnvironment default scene and task limits)',
                'adaptations': 'docs/OFFICIAL_SUPERMIMIC.md'}
    write_json(args.output / 'manifest.json', manifest)
    tracker = ExperimentTracker(args.output, manifest, args.wandb_project, None, args.wandb_mode,
                                job_type='official-supermimic')
    status = {'state': 'running', 'pid': os.getpid(), 'worlds': args.worlds}
    stop = {'requested': False}
    signal.signal(signal.SIGTERM, lambda *a: stop.update(requested=True))
    signal.signal(signal.SIGINT, lambda *a: stop.update(requested=True))

    def checkpoint(name, iteration):
        torch.save({'dynamics': dynamics.state_dict(), 'actor': actor.state_dict(), 'critic': critic.state_dict(),
                    'wm_optimizers': [o.state_dict() for o in wm_learner.optimizers],
                    'actor_optimizers': [o.state_dict() for o in policy_learner.actor_optimizers],
                    'critic_optimizers': [o.state_dict() for o in policy_learner.critic_optimizers],
                    'metadata': metadata, 'manifest': manifest, 'iteration': iteration}, args.output / (name + '.tmp'))
        (args.output / (name + '.tmp')).replace(args.output / name)

    state = spec.pack_batch(frame).float()
    returns = torch.zeros(args.worlds, device=device)
    lengths = torch.zeros(args.worlds, device=device, dtype=torch.long)
    episodes = torch.zeros_like(lengths)
    peak_material = torch.zeros(args.worlds, device=device)
    peak_ratio = torch.zeros(args.worlds, device=device)
    timing = {'physics': 0., 'world_model': 0., 'policy': 0.}
    updates, losses = 0, {}
    try:
        for step in range(args.max_iterations):
            if stop['requested'] or time.monotonic() - started > args.max_hours * 3600:
                status['state'] = 'interrupted' if stop['requested'] else 'budget_complete'
                break
            goal = frame['target_pose'].float()
            with torch.no_grad():
                actor.eval()
                action = actor(state[:, None], goal)
                action = action + torch.randn_like(action) * policy_cfg.noise
            tick = time.monotonic()
            frame, reward, terminated, truncated, info = env.step(spec.execute(action))
            timing['physics'] += time.monotonic() - tick
            replay.insert({'state': state, 'action': action, 'goal': goal,
                           'terminated': terminated.bool(), 'truncated': truncated.bool()})
            wm_learner.update_normalization(state, action)
            actor.update_normalization(state, goal)
            with torch.no_grad():
                next_state = spec.pack_batch_transition(frame, state)
                returns += reward
                lengths += 1
                peak_material = torch.maximum(peak_material, info['peak_force'].float())
                peak_ratio = torch.maximum(peak_ratio, info['safety_margin'].float())
            ended = terminated | truncated
            metrics = {'iteration': step + 1, 'physics/transitions': (step + 1) * args.worlds,
                       **ended_metrics(ended, returns, lengths, info, truncated, peak_material, peak_ratio)}
            for e in ended.nonzero().flatten().tolist():
                append_json(args.output / 'episodes.jsonl', episode_row(
                    e, int(episodes[e]), step + 1, returns, lengths, info, truncated, peak_material, peak_ratio))
            if bool(ended.any()):
                frame = env.reset(ended)
                next_state = spec.reset_batch_history(frame, next_state, ended)
                for t in (returns, lengths, peak_material, peak_ratio):
                    t[ended] = 0
                episodes[ended] += 1
            state = next_state
            for _ in range(should_train(step)):
                if replay.num_transitions < wm_cfg.horizon:
                    continue
                tick = time.monotonic()
                losses.update(wm_learner.update([replay.sample(wm_cfg.horizon, wm_cfg.batch_size)
                                                 for _ in range(wm_cfg.num_batches)]))
                timing['world_model'] += time.monotonic() - tick
                tick = time.monotonic()
                losses.update(policy_learner.update())
                timing['policy'] += time.monotonic() - tick
                updates += 1
            metrics.update({'updates': updates, **losses, **{'time/' + k: v for k, v in timing.items()}})
            if step % runner_cfg.log_interval == 0 or bool(ended.any()):
                tracker.log(metrics)
            status.update(iteration=step + 1, updates=updates, episodes=int(episodes.sum()),
                          transitions=(step + 1) * args.worlds, wall_seconds=time.monotonic() - started,
                          timing=dict(timing), losses={k: v for k, v in losses.items()})
            write_json(args.output / 'progress.json', status)
            if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0:
                checkpoint('checkpoint_%08d.pt' % (step + 1), step + 1)
        else:
            status['state'] = 'completed'
    except BaseException as error:
        status.update(state='failed', error=repr(error))
        raise
    finally:
        checkpoint('last.pt', status.get('iteration', 0))
        write_json(args.output / 'progress.json', status)
        tracker.finish(args.output / 'last.pt', status)


if __name__ == '__main__':
    main()
