"""Run the source SAC trainer with external tracking and bounded-job checkpoint cleanup."""
import argparse
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time

import numpy as np
import rospy
from tf2rl.algos.sac import SAC
from tf2rl.experiments.cb_trainer import Trainer
from tf2rl.misc.get_replay_buffer import save_replay_buffer
from ur3e_openai.robot_env import RobotGazeboEnv

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--sliceit-root', required=True)
parser.add_argument('--output', required=True)
args, source_arguments = parser.parse_known_args()
ROOT = Path(args.sliceit_root).resolve()
OUT = Path(args.output).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutting_world_model.lifecycle import close_environment, finalize_training
from cutting_world_model.sidecar_tracking import SidecarTracker
sys.argv = [str(ROOT / 'underlay_ws/src/ur_python_utilities/ur3e_rl/scripts/tf2rl_sac.py')] + source_arguments

config = json.loads((OUT / 'run_config.json').read_text())
evaluating = '--evaluate' in source_arguments
tracker = SidecarTracker(OUT, config, job_type='sac-evaluation' if evaluating else 'sac-training')
(OUT / 'wandb_link.json').write_text(json.dumps(tracker.info, indent=2) + '\n')
state = dict(state='running', steps=0, updates=0, episodes=0, episode_steps=0, episode_return=0.)
started = time.monotonic()
objects = {}
stop_requested = False
episode_results = []
def request_stop(signum, frame):
    global stop_requested
    if signum == signal.SIGTERM:
        raise RuntimeError('Training terminated by supervisor')
    stop_requested = True
original_ros_init = rospy.init_node
def initialize_ros(*args, **kwargs):
    kwargs['disable_signals'] = True
    result = original_ros_init(*args, **kwargs)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    return result
rospy.init_node = initialize_ros
events = (OUT / 'metrics.jsonl').open('a', buffering=1)

def emit(values, upload=True):
    values = dict(values, iteration=state['steps'], updates=state['updates'], wall_seconds=time.monotonic() - started)
    events.write(json.dumps(values) + '\n')
    (OUT / 'progress.json').write_text(json.dumps(dict(state, wall_seconds=time.monotonic() - started), indent=2) + '\n')
    if upload:
        tracker.log(values)

original_init = Trainer.__init__
def initialize(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    objects['trainer'] = self
    objects['environment'] = self._env.unwrapped
    (OUT / 'learner_path.txt').write_text(self._output_dir + '\n')
Trainer.__init__ = initialize
original_update = Trainer.update_policy
def update(self, replay_buffer, *args, **kwargs):
    objects['replay'] = replay_buffer
    result = original_update(self, replay_buffer, *args, **kwargs)
    if stop_requested:
        raise SystemExit(0)
    return result
Trainer.update_policy = update
original_train = SAC.train
def train(self, *args, **kwargs):
    if evaluating:
        raise RuntimeError('SAC optimizer was called during frozen evaluation')
    result = original_train(self, *args, **kwargs)
    if not np.isfinite(np.asarray(result)).all():
        raise FloatingPointError('Non-finite SAC TD errors')
    state['updates'] += 1
    if state['updates'] == 1 or state['updates'] % 100 == 0:
        iterations = {name: int(getattr(self, name + '_optimizer').iterations.numpy())
                      for name in ('actor', 'qf1', 'qf2', 'alpha')}
        emit(dict(kind='optimizer', **{'optimizer/' + k: v for k, v in iterations.items()}))
    return result
SAC.train = train
original_step = RobotGazeboEnv.step
def step(self, action):
    objects['environment'] = self
    result = original_step(self, action)
    observation, reward, done, info = result
    if not np.isfinite(observation).all() or not np.isfinite(reward):
        raise FloatingPointError('Non-finite physical observation or reward')
    state['steps'] += 1
    state['episode_steps'] += 1
    state['episode_return'] += float(reward)
    emit(dict(kind='transition', reward=float(reward), done=bool(done),
              **{'reward/step': float(reward), 'physics/reward': float(reward)}))
    objects['last_info'] = info
    if evaluating and done:
        state['episodes'] += 1
        emit(dict(kind='episode', episode=state['episodes'],
                  episode_return=state['episode_return'], episode_steps=state['episode_steps'],
                  success=bool(info.get('success', False)), collision=bool(info.get('collision', False)),
                  **{'reward/episode': state['episode_return']}))
        state['episode_steps'] = 0
        state['episode_return'] = 0.
    return result
RobotGazeboEnv.step = step
if evaluating:
    def evaluate_once(self):
        before = [variable.numpy().copy() for variable in self._policy.actor.weights]
        names = ('actor', 'qf1', 'qf2', 'alpha')
        iterations = {name: int(getattr(self._policy, name + '_optimizer').iterations.numpy())
                      for name in names}
        result = self.evaluate_policy(total_steps=0)
        unchanged = all(np.array_equal(old, variable.numpy())
                        for old, variable in zip(before, self._policy.actor.weights))
        after = {name: int(getattr(self._policy, name + '_optimizer').iterations.numpy())
                 for name in names}
        if not unchanged or iterations != after or state['updates']:
            raise RuntimeError('Evaluation changed policy or optimizer state')
        state.update(evaluation_episodes=self._test_episodes,
                     evaluation_return=float(result[0]), evaluation_success_rate=float(result[2]),
                     actor_unchanged=unchanged, optimizer_iterations=after)
        emit(dict(kind='evaluation', **state), upload=True)
    Trainer.evaluate_policy_continuously = evaluate_once
original_detailed_log = Trainer._detailed_log
def detailed_log(self, n_episode, total_steps, episode_steps, episode_return):
    result = original_detailed_log(self, n_episode, total_steps, episode_steps, episode_return)
    state['episodes'] = int(n_episode)
    info = objects.get('last_info', {})
    episode_results.append((float(episode_return), bool(info.get('success', False))))
    emit(dict(kind='episode', episode=int(n_episode), episode_return=float(episode_return),
              episode_steps=int(episode_steps), success=bool(info.get('success', False)),
              collision=bool(info.get('collision', False)),
              success_rate=float(np.mean([item[1] for item in episode_results])),
              **{'reward/episode': float(episode_return),
                 'reward/episode_mean_20': float(np.mean([item[0] for item in episode_results[-20:]]))}))
    state['episode_steps'] = 0
    state['episode_return'] = 0.
    return result
Trainer._detailed_log = detailed_log
primary_error = None
try:
    runpy.run_path(str(ROOT / 'underlay_ws/src/ur_python_utilities/ur3e_rl/scripts/tf2rl_sac.py'), run_name='__main__')
    state['state'] = 'completed'
except SystemExit as error:
    primary_error = error if error.code not in (None, 0) else None
    state['state'] = 'stopped' if error.code in (None, 0) else 'failed'
    state['exit_reason'] = repr(error)
    raise
except BaseException as error:
    primary_error = error
    state['state'] = 'failed'
    state['exit_reason'] = repr(error)
    raise
finally:
    def save():
        if evaluating:
            return
        trainer = objects.get('trainer')
        if trainer is None:
            return
        failures = []
        try:
            state['final_checkpoint'] = trainer.checkpoint_manager.save()
        except BaseException as error:
            failures.append('checkpoint: ' + repr(error))
        if 'replay' in objects:
            try:
                save_replay_buffer(objects['replay'], trainer.replay_buffer_path)
                state['replay_rows'] = int(objects['replay'].get_stored_size())
                state['replay_path'] = str(trainer.replay_buffer_path)
            except BaseException as error:
                failures.append('replay: ' + repr(error))
        if failures:
            raise RuntimeError('; '.join(failures))

    def record():
        state['wall_seconds'] = time.monotonic() - started
        path = OUT / 'progress.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    def close():
        environment = objects.get('environment')
        if environment is not None:
            try:
                state['shutdown'] = close_environment(environment)
            except Exception as error:
                state['shutdown'] = getattr(error, 'results', repr(error))
                raise

    def finish():
        failures = []
        for name, callback in (
                ('event_write', lambda: events.write(json.dumps(dict(kind='final', **state)) + '\n')),
                ('event_close', events.close)):
            try:
                callback()
            except BaseException as error:
                failures.append({'stage': name, 'error': repr(error)})
        if failures:
            state['state'] = 'failed'
            state.setdefault('cleanup_errors', []).extend(failures)
        try:
            tracker.finish(status=state)
        except BaseException as error:
            failures.append({'stage': 'tracking_finish', 'error': repr(error)})
        if failures:
            raise RuntimeError('Final event/tracking failure: ' + repr(failures))

    cleanup_errors = finalize_training(save, record, finish, close, state)
    if cleanup_errors and primary_error is None:
        raise RuntimeError('Training finalization failed: ' + repr(cleanup_errors))
