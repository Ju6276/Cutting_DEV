"""Frozen policies share physical resets, task heads and episode accounting."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .task_heads import reward_settings, without_reward_settings
from .checkpoint import checkpoint_identity


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def check_contract(expected, actual, path='benchmark'):
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            raise ValueError(f'{path}: configuration keys differ')
        for key in expected:
            check_contract(expected[key], actual[key], f'{path}.{key}')
    elif expected != actual:
        raise ValueError(f'{path}: expected {expected!r}, got {actual!r}')


# Benchmark name -> exported DiSECt scene (relative to the backend root).
BENCHMARK_SCENES = {
    'public-prism-ur5e-v1': 'runs/disect_source_v4/scene.pt',
    # Same prism with the bottom layer static (include box y < 0.002 instead of y < -0.002).
    'public-prism-fixed-bottom-ur5e-v1': 'runs/disect_source_v4_fixed_bottom/scene.pt',
}


def benchmark_scene(contract):
    if contract.get('name') not in BENCHMARK_SCENES:
        raise ValueError(f"Unknown benchmark {contract.get('name')!r}")
    return BENCHMARK_SCENES[contract['name']]


def physical_contract(backend_root, env):
    root = Path(backend_root)
    names = [name for name, path in BENCHMARK_SCENES.items() if path == env.physics.scene]
    if len(names) != 1:
        raise ValueError(f'Scene {env.physics.scene!r} belongs to no benchmark')
    scene = torch.load(root / env.physics.scene, weights_only=False, map_location='cpu')
    with np.load(root / 'runs/source/episode_000.npz') as capture:
        reset = {key: capture[key][0].tolist()
                 for key in ('joint_q', 'joint_qd', 'target_pose')}
    return {
        'name': names[0],
        'reset_mode': 'fixed_export',
        'task': {key: value for key, value in env.physics.task.items() if key != 'robot_xml'},
        'controller_parameters': json.loads(
            (root / 'runs/controller_source/controller_params.json').read_text()),
        'material_settings': scene['settings'],
        'mesh': {'particles': env.physics.material.particles,
                 'cut_springs': env.physics.material.springs},
        'reset': reset,
        'asset_content': {
            'scene_sha256': file_digest(root / env.physics.scene),
            'robot_model_sha256': file_digest(root / 'runs/model_coupled/robot.mjb'),
            'robot_xml_sha256': hashlib.sha256(env.physics.task['robot_xml'].encode()).hexdigest(),
        },
    }


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def copy_frame(frame):
    return {key: value[0].detach().cpu().numpy().copy() for key, value in frame.items()}


def save_episode(output, row, states, transitions):
    arrays = {key: np.stack([state[key] for state in states]) for key in states[0]}
    for key in transitions[0] if transitions else ():
        arrays[key] = np.stack([transition[key] for transition in transitions])
    if not transitions:
        arrays.update(action=np.empty((0, 6)), reward=np.empty(0),
                      terminated=np.empty(0, dtype=bool), truncated=np.empty(0, dtype=bool))
    np.savez_compressed(output / f"episode_{row['episode']:03d}.npz", **arrays)


def episode_metrics(row, totals):
    """Derived per-episode metrics from per-step integrals (see docs/BENCHMARK.md)."""
    seconds = row['simulation_seconds']
    return {
        'mean_material_force_n': totals['material_impulse'] / seconds if seconds > 0 else 0.,
        'material_impulse_ns': totals['material_impulse'],
        'knife_work_j': totals['knife_work'],
        'knife_gross_work_j': totals['knife_gross_work'],
        'safety_margin': totals['safety_margin'],
        'cut_fraction': row['released_springs'] / row['total_cut_springs'],
        'mean_jerk': (None if totals['jerkiness'] is None
                      else totals['jerkiness'] / row['steps'] if row['steps'] else 0.),
    }


SUMMARY_EXCLUDED = ('episode',)


def summarize(episodes):
    """Mean and population std of numeric row fields over completed episodes."""
    completed = [row for row in episodes if row['completed']]
    summary = {'episodes': len(completed),
               'success_rate': (sum(row['success'] for row in completed) / len(completed)
                                if completed else None),
               'collision_rate': (sum(row['collision'] for row in completed) / len(completed)
                                  if completed else None)}
    if completed:
        for key, value in completed[0].items():
            if key in SUMMARY_EXCLUDED or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values = np.array([row[key] for row in completed], dtype=np.float64)
            summary[key] = {'mean': float(values.mean()), 'std': float(values.std())}
    return summary


class ConstantPolicy:
    def reset(self, frame):
        pass

    def act(self, frame):
        return torch.zeros((1, 6), device=frame['observation'].device)

    def observe(self, frame):
        pass

    def unchanged(self):
        return True


class ReferencePolicy(ConstantPolicy):
    """Prescribed knife trajectory: the backend reference branch sets the controller target.

    The recorded six-component action is zero; the 6-D action mapper is bypassed.
    """
    reference = True


class WorldModelPolicy:
    def __init__(self, checkpoint, env):
        from .control_state import ControlStateSpec
        from .policy import SliceItPolicy

        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        metadata = ControlStateSpec.metadata_from_frame(env.metadata, copy_frame(env.current))
        # The benchmark scores every method with its own reward; a policy trained with
        # another force penalty is still evaluated on the same physical task.
        if without_reward_settings(metadata) != without_reward_settings(saved['metadata']):
            raise ValueError('Checkpoint and physical task metadata differ')
        self.training_reward = reward_settings(saved['metadata']['environment']) or 'sliceit'
        self.spec = ControlStateSpec(metadata).to(env.device)
        self.policy = SliceItPolicy().to(env.device).eval()
        self.weights = saved['learner']['policy']
        self.policy.load_state_dict(self.weights)
        self.updates = saved['learner']['updates']
        self.checkpoint_identity = checkpoint_identity(saved)
        self.reset(env.current)

    def reset(self, frame):
        self.history = frame['observation'].float()[None].repeat(self.policy.history, 1, 1)

    def act(self, frame):
        return self.spec.execute(self.policy(self.history, frame['target_pose'].float()))

    def observe(self, frame):
        self.history = torch.cat((self.history[1:], frame['observation'].float()[None]), 0)

    def unchanged(self):
        return all(torch.equal(value.cpu(), self.weights[name])
                   for name, value in self.policy.state_dict().items())


@torch.no_grad()
def evaluate(env, policy, output, episodes, deadline, clock=time.monotonic, on_frame=None):
    """Each recorded transition ends at the physical terminal frame, before reset."""
    if env.worlds != 1 or episodes < 1:
        raise ValueError('Frozen benchmark uses one world and at least one episode')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = clock()
    report = {'state': 'running', 'requested_episodes': episodes, 'episodes': [],
              'optimizer_updates': 0, 'inference_seconds': 0., 'physics_seconds': 0.,
              'reset_seconds': 0.}
    write_json(output / 'progress.json', report)
    reference = bool(getattr(policy, 'reference', False))
    report['control_branch'] = 'knife_reference' if reference else 'policy_action'
    initial = copy_frame(env.current)
    row = None
    states = []
    transitions = []
    try:
        for episode in range(episodes):
            if clock() >= deadline:
                report['state'] = 'budget_complete'
                break
            tick = clock()
            frame = env.reset()
            reset = copy_frame(frame)
            if reset.keys() != initial.keys():
                raise ValueError('Physical reset fields changed')
            for key in initial:
                if not np.array_equal(reset[key], initial[key]):
                    raise ValueError(f'Physical reset changed {key}')
            policy.reset(frame)
            synchronize(env.device)
            report['reset_seconds'] += clock() - tick
            states = [reset]
            transitions = []
            start_time = float(frame['sim_time'][0, 0])
            initial_released = int((frame['cut_spring_ke'][0] == 0).sum())
            totals = dict.fromkeys(('material_impulse', 'knife_work', 'knife_gross_work',
                                    'safety_margin', 'jerkiness'), 0.)
            row = {'episode': episode, 'completed': False, 'steps': 0, 'return': 0.,
                   'simulation_seconds': 0., 'material_peak_force_n': 0.,
                   'initial_released_springs': initial_released,
                   'released_springs': initial_released,
                   'success': False, 'collision': False, 'outside': False,
                   'terminated': False, 'truncated': False,
                   'total_cut_springs': int(frame['cut_spring_ke'].shape[-1])}
            row.update(episode_metrics(row, totals))
            if on_frame is not None:
                on_frame(frame, None)
            while clock() < deadline:
                tick = clock()
                action = policy.act(frame)
                synchronize(env.device)
                report['inference_seconds'] += clock() - tick
                if action.shape != (1, 6) or not bool(torch.isfinite(action).all()):
                    raise ValueError('Policy must return a finite [1, 6] action')
                low, high = action.new_tensor(env.metadata['action_bounds'])
                if bool(((action < low) | (action > high)).any()):
                    raise ValueError('Policy action is outside the task bounds')
                if clock() >= deadline:
                    break
                tick = clock()
                frame, reward, terminated, truncated, info = (
                    env.step(action, reference=True) if reference else env.step(action))
                synchronize(env.device)
                report['physics_seconds'] += clock() - tick
                states.append(copy_frame(frame))
                transitions.append({
                    'action': action[0].cpu().numpy().copy(),
                    'reward': float(reward[0]), 'terminated': bool(terminated[0]),
                    'truncated': bool(truncated[0]),
                    **{key: value[0].cpu().numpy().copy() for key, value in info.items()},
                })
                row['return'] += float(reward[0])
                row.update(steps=row['steps'] + 1,
                           simulation_seconds=float(frame['sim_time'][0, 0]) - start_time,
                           material_peak_force_n=max(row['material_peak_force_n'], float(info['peak_force'][0])),
                           released_springs=int(info['released_springs'][0]),
                           success=bool(info['success'][0]), collision=bool(info['collision'][0]),
                           outside=bool(info['outside'][0]), terminated=bool(terminated[0]),
                           truncated=bool(truncated[0]))
                for key in ('material_impulse', 'knife_work', 'knife_gross_work'):
                    totals[key] += float(info[key][0])
                # Jerk is observed only with jerkiness_as_obs; otherwise mean_jerk is None.
                totals['jerkiness'] = (None if 'jerkiness' not in info or totals['jerkiness'] is None
                                       else totals['jerkiness'] + float(info['jerkiness'][0]))
                totals['safety_margin'] = max(totals['safety_margin'], float(info['safety_margin'][0]))
                row.update(episode_metrics(row, totals))
                row['completed'] = row['terminated'] or row['truncated']
                if on_frame is not None:
                    on_frame(frame, info)
                policy.observe(frame)
                write_json(output / 'progress.json', {**report, 'current_episode': row,
                                                      'wall_seconds': clock() - started})
                if row['completed']:
                    break
            row['newly_released_springs'] = row['released_springs'] - initial_released
            report['episodes'].append(row)
            save_episode(output, row, states, transitions)
            print(json.dumps(row), flush=True)
            if not row['completed']:
                report['state'] = 'budget_complete'
                break
        else:
            report['state'] = 'completed'
        report['policy_unchanged'] = policy.unchanged()
        if not report['policy_unchanged']:
            raise RuntimeError('Frozen evaluation changed policy parameters')
    except BaseException as error:
        report.update(state='failed', error=repr(error))
        if row is not None and all(item is not row for item in report['episodes']):
            row['newly_released_springs'] = row['released_springs'] - row['initial_released_springs']
            report['episodes'].append(row)
            save_episode(output, row, states, transitions)
        raise
    finally:
        completed = [row for row in report['episodes'] if row['completed']]
        report.update(completed_episodes=len(completed),
                      successes=sum(row['success'] for row in completed),
                      summary=summarize(report['episodes']),
                      wall_seconds=clock() - started)
        write_json(output / 'report.json', report)
        write_json(output / 'progress.json', report)
    return report
