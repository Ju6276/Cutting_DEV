#!/usr/bin/env python3
"""Fit a control-history model on whole episodes and evaluate separate episodes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.kinematics import pose_difference
from cutting_world_model.online_replay import OnlineReplay
from cutting_world_model.policy import SliceItPolicy, SliceItCritic
from cutting_world_model.provenance import git_revision
from cutting_world_model.trainer import OnlineLearner
from cutting_world_model.world_model import CuttingWorldModel
from third_party.supermimic.trainer_config import WMTrainerCfg


def write_json(path, result):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load_episode(path, metadata, device):
    task = json.loads((path.parent / 'task.json').read_text())
    for key in ('robot_xml', 'robot_base', 'robot_tip', 'joint_names', 'environment',
                'controller', 'action_bounds', 'task_interface_revision', 'step_interface_revision'):
        if task.get(key) != metadata.get(key):
            raise ValueError('Episode task mismatch: ' + str(path) + ' / ' + key)
    spec = ControlStateSpec(metadata).to(device)
    with np.load(path) as archive:
        data = {key: archive[key] for key in spec.base_fields}
        data.update({key: archive[key] for key in
                     ('actions', 'rewards', 'terminated', 'truncated', 'target_pose')})
    count = len(data['actions'])
    if not count or any(len(data[key]) != count + 1 for key in spec.base_fields):
        raise ValueError('Expected T actions and T+1 states: ' + str(path))
    if any(len(data[key]) != count for key in ('rewards', 'terminated', 'truncated')):
        raise ValueError('Transition label count differs: ' + str(path))
    ended = data['terminated'] | data['truncated']
    if ended[:-1].any() or not ended[-1]:
        raise ValueError('Expected one complete episode: ' + str(path))
    if not np.all(data['target_pose'] == data['target_pose'][0]):
        raise ValueError('Target changes within episode: ' + str(path))
    if not all(np.isfinite(value).all() for value in data.values()):
        raise ValueError('Non-finite recorded data: ' + str(path))
    return {'path': str(path), 'data': data, 'states': spec.pack_episode(data, device),
            'actions': torch.as_tensor(data['actions'], dtype=torch.float32, device=device),
            'goal': torch.as_tensor(data['target_pose'][0], dtype=torch.float32, device=device),
            'rewards': torch.as_tensor(data['rewards'], dtype=torch.float32, device=device),
            'done': torch.as_tensor(data['terminated'], device=device)}


@torch.no_grad()
def interface_check(metadata, episode):
    spec = ControlStateSpec(metadata).double()
    data = episode['data']
    states = spec.pack_episode(data, dtype=torch.float64)
    fields = {key: spec.field(states[1:], key) for key in spec.layout}
    derived, reward, done = spec.derive(states[:-1], fields,
        torch.as_tensor(data['actions'], dtype=torch.float64),
        torch.as_tensor(data['target_pose'][1:], dtype=torch.float64))
    result = {'path': episode['path'], 'transitions': len(data['actions']),
              'reward_max_abs': float((reward - torch.as_tensor(data['rewards'])).abs().max()),
              'done_mismatches': int((done != torch.as_tensor(data['terminated'])).sum())}
    for field in ('observation', 'motion_command', 'success_counter', 'control_history'):
        result[field + '_max_abs'] = float((spec.field(derived, field) - spec.field(states[1:], field)).abs().max())
    return result


@torch.no_grad()
def evaluate(model, episodes, horizons=(1, 8, 32)):
    spec = model.spec
    result = []
    for ep in episodes:
        for horizon in horizons:
            n = len(ep['actions']) - horizon + 1
            if n <= 0:
                continue
            starts = torch.arange(n, device=ep['states'].device)
            state = ep['states'][starts]
            rewards = torch.zeros(n, device=state.device)
            alive = torch.ones(n, dtype=torch.bool, device=state.device)
            for step in range(horizon):
                previous = state
                state, reward, logits = model.step(state, ep['actions'][starts + step], ep['goal'].expand(n, -1))
                collision = (spec.field(state, 'peak_wrench') > spec.force_limits).any(-1)
                task_done, _, _, _ = spec.heads.termination(
                    spec.field(state, 'observation'),
                    spec.field(previous, 'success_counter'), collision)
                # State errors audit the raw predictor; episodic returns end at
                # the first predicted task terminal, including its reward once.
                rewards += torch.where(alive, reward, torch.zeros_like(reward))
                alive &= ~task_done
                if not torch.isfinite(state).all() or not torch.isfinite(logits).all() or not torch.isfinite(reward).all():
                    result.append({'episode': ep['path'], 'horizon': horizon, 'finite': False, 'failed_step': step + 1})
                    break
            else:
                target = ep['states'][starts + horizon]
                error = spec.error_vector(state.double(), target.double())
                item = {'episode': ep['path'], 'horizon': horizon, 'windows': n, 'finite': True,
                        'normalized_rmse': float(error.square().mean().sqrt()),
                        'reset_start_normalized_rmse': float(error[0].square().mean().sqrt()),
                        'state_abs_max_normalized': float(spec.canonical(state).abs().max())}
                for key in spec.physical_fields:
                    diff = spec.field(state.double(), key) - spec.field(target.double(), key)
                    item[key + '_rmse'] = float(diff.square().mean().sqrt())
                    if 'wrench' in key:
                        item[key + '_force_rmse_N'] = float(diff[..., :3].square().mean().sqrt())
                        item[key + '_torque_rmse_Nm'] = float(diff[..., 3:].square().mean().sqrt())
                pose_delta = pose_difference(spec.fk(spec.field(state.double(), 'joint_q')),
                                             spec.fk(spec.field(target.double(), 'joint_q')))
                item['fk_translation_rmse_m'] = float(pose_delta[..., :3].square().mean().sqrt())
                item['fk_rotation_rmse_rad'] = float(pose_delta[..., 3:].square().mean().sqrt())
                truth_return = torch.stack([ep['rewards'][starts + j] for j in range(horizon)]).sum(0)
                item['return_rmse'] = float((rewards.double() - truth_return.double()).square().mean().sqrt())
                labels = ep['done'][starts + horizon - 1]
                probability = logits.sigmoid()
                actual_collision = (spec.field(target, 'peak_wrench') > spec.force_limits).any(-1)
                predicted_collision = (spec.field(state, 'peak_wrench') > spec.force_limits).any(-1)
                item.update(terminal_brier=float((probability - labels.float()).square().mean()),
                            terminal_positives=int(labels.sum()),
                            terminal_false_negatives=int((labels & (probability < .5)).sum()),
                            terminal_false_positives=int((~labels & (probability >= .5)).sum()),
                            collision_positives=int(actual_collision.sum()),
                            collision_false_negatives=int((actual_collision & ~predicted_collision).sum()),
                            collision_false_positives=int((~actual_collision & predicted_collision).sum()))
                result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, nargs='+', required=True)
    parser.add_argument('--held-out', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=50)
    parser.add_argument('--max-seconds', type=float, default=600.)
    parser.add_argument('--microbatch', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    start = time.monotonic()
    if set(p.resolve() for p in args.train) & set(p.resolve() for p in args.held_out):
        parser.error('Training and held-out episode paths overlap')
    if args.updates < 0 or args.max_seconds <= 0:
        parser.error('Require nonnegative updates and a positive time budget')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    task = json.loads((args.train[0].parent / 'task.json').read_text())
    with np.load(args.train[0]) as data:
        first = {name: data[name][0] for name in ControlStateSpec.physical_fields + ControlStateSpec.history_fields}
    metadata = ControlStateSpec.metadata_from_frame(task, first)
    model = CuttingWorldModel(metadata).to(args.device)
    train = [load_episode(path, metadata, args.device) for path in args.train]
    held_out = [load_episode(path, metadata, args.device) for path in args.held_out]
    replay = OnlineReplay(model.spec.state_dim, args.device)
    for ep in train:
        for t, action in enumerate(ep['actions']):
            replay.add(ep['states'][t], action, action, ep['goal'], float(ep['rewards'][t]),
                       ep['states'][t + 1], bool(ep['data']['terminated'][t]), bool(ep['data']['truncated'][t]))
    # Offline fitting uses the same batches, loss and Adam settings; eager execution avoids compile startup.
    cfg = WMTrainerCfg(compile=None)
    learner = OnlineLearner(model, SliceItPolicy().to(args.device),
                            SliceItCritic(model.spec.state_dim).to(args.device), args.microbatch, wm_cfg=cfg)
    write_json(args.output / 'manifest.json', {
        'git': git_revision(Path(__file__).resolve().parents[1]), 'command': sys.argv,
        'world_model': asdict(cfg), 'arguments': {key: str(value) if isinstance(value, Path) else
        [str(p) for p in value] if key in ('train', 'held_out') else value for key, value in vars(args).items()},
        'state_dim': model.spec.state_dim, 'predicted_dim': model.spec.delta_dim,
        'dynamics_input_dim': model.spec.dynamics_dim,
        'parameters': sum(p.numel() for p in model.parameters()),
        'train_transitions': sum(len(ep['actions']) for ep in train),
        'held_out_transitions': sum(len(ep['actions']) for ep in held_out)})
    write_json(args.output / 'task.json', metadata)
    report = {'interface': [interface_check(metadata, ep) for ep in train + held_out],
              'initial_train': evaluate(model, train), 'initial_held_out': evaluate(model, held_out)}
    write_json(args.output / 'report.json', report)
    if any(row['done_mismatches'] or any(value > 1e-8 for key, value in row.items() if key.endswith('_max_abs'))
           for row in report['interface']):
        raise RuntimeError('Recorded task equations do not match compact state')
    completed = 0
    def progress(**values):
        write_json(args.output / 'progress.json', dict(update=completed, wall_seconds=time.monotonic() - start, **values))
        if time.monotonic() - start > args.max_seconds:
            raise TimeoutError('Offline fitting time budget reached')
    learner.progress = progress
    try:
        for update in range(1, args.updates + 1):
            metrics = learner.fit_world_model(replay.buffer)
            completed = update
            row = {'update': update, 'world_model': metrics, 'wall_seconds': time.monotonic() - start}
            if update in (1, 5, 10, 20, 50) or update == args.updates:
                row['held_out'] = evaluate(model, held_out)
            with (args.output / 'metrics.jsonl').open('a') as output:
                output.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps({'update': update, 'loss': metrics['loss'], 'seconds': row['wall_seconds']}), flush=True)
        report['status'] = 'completed'
    except TimeoutError:
        report['status'] = 'budget_complete'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        adam_steps = int(next(iter(learner.wopt.state.values()))['step']) if learner.wopt.state else 0
        temporary = args.output / 'model.tmp'
        torch.save({'metadata': metadata, 'world_model': model.state_dict(),
                    'optimizer': learner.wopt.state_dict(), 'updates': completed, 'adam_steps': adam_steps}, temporary)
        temporary.replace(args.output / 'model.pt')
        report.update(updates=completed, adam_steps=adam_steps, wall_seconds=time.monotonic() - start,
                      train=evaluate(model, train), held_out=evaluate(model, held_out))
        write_json(args.output / 'report.json', report)
        write_json(args.output / 'progress.json', {'state': report['status'], 'updates': completed,
                                                  'wall_seconds': report['wall_seconds']})


if __name__ == '__main__':
    main()
