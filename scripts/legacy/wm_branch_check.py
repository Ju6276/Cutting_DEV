#!/usr/bin/env python3
"""Step 2 of the WM reliability check: do the WM and physics rank the same actions alike?

For each unperturbed episode of a policy under test (ablation-(a) checkpoints), at every
recorded state two branches run for ``--horizon`` steps, each closed loop with its own
policy: A = the episode's policy, B = the reference policy (policy62). The WM side uses the
WM of the checkpoint under test with the rule of ``OnlineLearner.imagine`` (task terminals
absorb). The physical side restores the recorded state from a replay of the episode's
action prefix and executes the same rule. Reported separately per branch: the sum of
rewards, the terminal critic value (critic of the checkpoint under test; zero once
absorbed), first terminal step and reason, peak sensor force and, physically, peak material
force. The replay is checked against the recorded states and against a second replay with
the same inputs (FP32 atomic accumulation makes repeats differ slightly).
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model.gpu_env import GPUEnvironment
from cutting_world_model.policy import SliceItCritic, SliceItPolicy
from cutting_world_model.provenance import git_revision
from cutting_world_model.reliability import force_summary, imagine_closed_loop, load_episodes, read_index
from cutting_world_model.trainer import OnlineLearner
from cutting_world_model.world_model import CuttingWorldModel

REASON_NAMES = {-1: None, 0: 'success', 1: 'collision', 2: 'outside'}


def learner_from(path, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    data = saved['learner']
    wm = CuttingWorldModel(saved['metadata']).to(device)
    learner = OnlineLearner(wm, SliceItPolicy().to(device), SliceItCritic(wm.spec.state_dim).to(device),
                            microbatch=64, critic_target=data.get('critic_variant', {}).get('target', 'imagined'),
                            actor_horizon=data.get('actor_horizon'), actor_bc_alpha=data.get('actor_bc_alpha'),
                            actor_objective=data.get('actor_objective', 'model'),
                            wm_terminal_gradient=data.get('wm_terminal_gradient', 'dynamics'))
    wm.load_state_dict(data['world_model'])
    learner.policy.load_state_dict(data['policy'])
    learner.critic.load_state_dict(data['critic'])
    for module in (wm, learner.policy, learner.critic):
        module.eval()
    return learner


def copy_frame(frame):
    return {k: v.clone() for k, v in frame.items()}


@torch.no_grad()
def replay(env, spec, ep):
    """Execute the recorded actions from reset; frames before each step and packed states."""
    frame = env.reset()
    state = spec.pack_batch(frame).float()
    frames, states = [copy_frame(frame)], [state[0]]
    for action in ep['actions']:
        frame, *_ = env.step(action[None])
        state = spec.pack_batch_transition(frame, state).float()
        frames.append(copy_frame(frame))
        states.append(state[0])
        if env.needs_reset.any():
            break
    return frames, torch.stack(states)


@torch.no_grad()
def physical_branch(env, learner, policy, frame, state, goal, horizon):
    spec = learner.wm.spec
    env.restore_frame(copy_frame(frame))
    history = learner.observation_history(state[None])
    total, first, reason, sensor, material = 0., 0, None, 0., 0.
    for step in range(1, horizon + 1):
        action = spec.execute(policy(history, goal[None]))
        frame, reward, term, trunc, info = env.step(action)
        state = spec.pack_batch_transition(frame, state[None]).float()[0]
        total += float(reward[0])
        sensor = max(sensor, float(force_summary(spec, state)[0]))
        material = max(material, float(info['peak_force'][0]))
        history = torch.cat((history[1:], frame['observation'].float()[None]), 0)
        if bool(term[0]) or bool(trunc[0]):
            first = step
            reason = ('success' if bool(info['success'][0]) else 'collision' if bool(info['collision'][0])
                      else 'outside' if bool(info['outside'][0]) else 'truncated')
            break
    value = 0. if first else float(learner.critic_values(state[None], goal[None])[0])
    return {'reward_sum': total, 'terminal_value': value, 'first_terminal_step': first or None,
            'terminal_reason': reason, 'peak_sensor_force_n': sensor, 'peak_material_force_n': material}


@torch.no_grad()
def model_branch(learner, policy, state, goal, horizon):
    spec = learner.wm.spec
    history = learner.observation_history(state[None])
    rewards, states, first, reason = imagine_closed_loop(learner.wm, policy, state[None], history,
                                                         goal[None], horizon)
    end = states[-1]
    value = 0. if int(first[0]) else float(learner.critic_values(end, goal[None])[0])
    sensor = float(force_summary(spec, states[1:, 0])[0].max())
    return {'reward_sum': float(rewards.sum()), 'terminal_value': value,
            'first_terminal_step': int(first[0]) or None, 'terminal_reason': REASON_NAMES[int(reason[0])],
            'peak_sensor_force_n': sensor}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=Path, required=True, help='supplement directory')
    parser.add_argument('--split', default='dev')
    parser.add_argument('--checkpoint', nargs=2, action='append', metavar=('LABEL', 'PATH'), required=True,
                        help='policies under test; their unperturbed episodes are branched')
    parser.add_argument('--reference', type=Path, required=True, help='reference policy checkpoint (policy62)')
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--wall-seconds', type=float, default=3000.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    device = 'cuda:0'
    env = GPUEnvironment(str((ROOT / 'backends/mujoco').resolve()), 1)
    reference = learner_from(args.reference, device).policy
    index = {item['path']: item for item in read_index(args.episodes)}
    episodes = load_episodes(args.episodes, args.split, device)
    report = {'git': git_revision(ROOT), 'command': sys.argv, 'horizon': args.horizon, 'episodes': []}
    stop_reason = None
    for label, path in args.checkpoint:
        learner = learner_from(path, device)
        spec = learner.wm.spec
        mine = [ep for ep in episodes if index[ep['id']]['policy'] == label and index[ep['id']]['sigma'] == 0]
        for ep in mine:
            frames, states = replay(env, spec, ep)
            frames2, states2 = replay(env, spec, ep)
            n = len(states) - 1
            check = {'replayed_steps': n, 'recorded_steps': len(ep['actions']),
                     'max_abs_vs_recorded': float((states - ep['states'][:n + 1].float()).abs().max()),
                     'max_abs_repeat_spread': float((states - states2).abs().max())}
            rows = []
            for t in range(n):
                if time.monotonic() - started > args.wall_seconds:
                    stop_reason = 'wall budget'
                    break
                goal = ep['goals'][t].float()
                row = {'t': t}
                for name, policy in (('A', learner.policy), ('B', reference)):
                    row[name] = {'model': model_branch(learner, policy, ep['states'][t].float(), goal, args.horizon),
                                 'physics': physical_branch(env, learner, policy, frames[t], states[t], goal,
                                                            args.horizon)}
                rows.append(row)
            report['episodes'].append({'label': label, 'id': ep['id'], 'replay_check': check, 'branches': rows})
            print(label, ep['id'], json.dumps(check), len(rows), 'states', flush=True)
            if stop_reason:
                break
        if stop_reason:
            break
    report['stop_reason'] = stop_reason
    report['seconds'] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
