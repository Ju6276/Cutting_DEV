"""Execute the SAC entry's dispatch and finalization without starting ROS."""
import ast
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cutting_world_model.lifecycle import finalize_training


def entry_fixture(tmp_path, evaluating=True, failure=None, mutate_actor=False):
    calls = []
    weights = np.array([1., 2.])
    policy = SimpleNamespace(actor=SimpleNamespace(
        weights=[SimpleNamespace(numpy=lambda: weights)]))
    for name in ('actor', 'qf1', 'qf2', 'alpha'):
        setattr(policy, name + '_optimizer', SimpleNamespace(
            iterations=SimpleNamespace(numpy=lambda: np.array(123))))

    class Trainer:
        _policy = policy
        _test_episodes = 10
        checkpoint_manager = SimpleNamespace(save=lambda: calls.append('save'))

        def evaluate_policy_continuously(self):
            raise AssertionError('The infinite evaluator must be bypassed')

        def __call__(self):
            # The source cb_trainer.__call__ continues into training if this returns.
            if evaluating:
                self.evaluate_policy_continuously()
            calls.append('training')

        def evaluate_policy(self, total_steps):
            assert total_steps == 0
            if failure:
                raise failure
            calls.extend(['evaluation_episode'] * self._test_episodes)
            if mutate_actor:
                weights[0] = 3.
            return 91., 19., 1.

    state = dict(state='running', updates=0)
    finished = []

    def close(environment):
        calls.append('close')
        return {'close': {'state': 'ok'}}

    scope = dict(
        Trainer=Trainer, evaluating=evaluating, np=np, state=state,
        emit=lambda *args, **kwargs: calls.append('emit'),
        runpy=SimpleNamespace(run_path=lambda *args, **kwargs: Trainer()()),
        ROOT=tmp_path, OUT=tmp_path, objects={'trainer': Trainer(), 'environment': object()},
        time=time, started=time.monotonic(), json=json, events=io.StringIO(),
        close_environment=close, finalize_training=finalize_training,
        tracker=SimpleNamespace(finish=lambda status: finished.append(dict(status))))
    path = Path(__file__).resolve().parents[1] / 'scripts/train_sac.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    dispatch = next(node for node in tree.body if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Name) and node.test.id == 'evaluating')
    lifecycle = next(node for node in tree.body if isinstance(node, ast.Try))
    initialization = ast.Assign(targets=[ast.Name(id='primary_error', ctx=ast.Store())],
                                value=ast.Constant(value=None))
    executable = ast.fix_missing_locations(ast.Module(
        body=[dispatch, initialization, lifecycle], type_ignores=[]))
    return compile(executable, str(path), 'exec'), scope, calls, finished


def test_frozen_evaluation_finishes_ten_episodes_without_entering_training(tmp_path):
    code, scope, calls, finished = entry_fixture(tmp_path)
    exec(code, scope)
    assert calls.count('evaluation_episode') == 10
    assert 'training' not in calls
    assert 'save' not in calls
    assert calls.count('close') == 1
    assert scope['state']['updates'] == 0
    assert scope['state']['actor_unchanged'] is True
    assert scope['state']['optimizer_iterations'] == dict.fromkeys(
        ('actor', 'qf1', 'qf2', 'alpha'), 123)
    assert finished[-1]['state'] == 'completed'
    assert json.loads((tmp_path / 'progress.json').read_text())['state'] == 'completed'


@pytest.mark.parametrize('failure, mutate_actor, message', [
    (RuntimeError('physical evaluation failed'), False, 'physical evaluation failed'),
    (None, True, 'Evaluation changed policy or optimizer state'),
])
def test_evaluation_failure_propagates_and_finalizes(tmp_path, failure, mutate_actor, message):
    code, scope, calls, finished = entry_fixture(tmp_path, failure=failure,
                                                 mutate_actor=mutate_actor)
    with pytest.raises(RuntimeError, match=message):
        exec(code, scope)
    assert 'training' not in calls
    assert 'save' not in calls
    assert calls.count('close') == 1
    assert finished[-1]['state'] == 'failed'
    assert message in finished[-1]['exit_reason']


def test_training_still_uses_original_call_and_checkpoint_cleanup(tmp_path):
    code, scope, calls, finished = entry_fixture(tmp_path, evaluating=False)
    original_call = scope['Trainer'].__call__
    exec(code, scope)
    assert scope['Trainer'].__call__ is original_call
    assert calls == ['training', 'save', 'close']
    assert finished[-1]['state'] == 'completed'
