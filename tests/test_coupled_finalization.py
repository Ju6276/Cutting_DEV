"""Checkpoint failures must preserve final state and the primary run error."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest


@pytest.fixture
def entry():
    path = Path(__file__).resolve().parents[1] / 'scripts/train_coupled.py'
    spec = importlib.util.spec_from_file_location('coupled_finalization_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def test_coupled_checkpoint_failure_still_records_final_state(entry):
    module, _ = entry
    status = {'state': 'budget_complete', 'wm_updates': 300, 'policy_updates': 0}
    records = []

    def save():
        raise OSError('checkpoint storage unavailable')

    errors = module.finalize_coupled_run(save, lambda: records.append(copy.deepcopy(status)), status)
    assert errors == [{'stage': 'save', 'error': "OSError('checkpoint storage unavailable')"}]
    assert len(records) == 2
    assert all(item['state'] == 'failed' for item in records)
    assert records[-1]['wm_updates'] == 300
    assert records[-1]['policy_updates'] == 0
    assert records[-1]['cleanup_errors'] == errors


def test_coupled_finally_keeps_primary_error_when_checkpoint_also_fails(entry, tmp_path):
    module, path = entry
    tree = ast.parse(path.read_text(), filename=str(path))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    execution = copy.deepcopy(next(node for node in run.body if isinstance(node, ast.Try)))
    execution.body = ast.parse('raise original_error').body
    code = compile(ast.fix_missing_locations(ast.Module(body=[execution], type_ignores=[])),
                   str(path), 'exec')
    original = RuntimeError('physical environment failed')
    records = []

    def checkpoint(name):
        assert name == 'last.pt'
        raise OSError('checkpoint storage unavailable')

    status = {'state': 'running'}
    learner = SimpleNamespace(stage='world_model_only', wm_only_updates=500,
                              wm_updates=300, updates=0,
                              optimizer_steps={'world_model': 3003, 'policy': 0, 'critic': 0})
    scope = dict(original_error=original, primary_error=None, status=status,
                 schedule=SimpleNamespace(iteration=800, transitions=819200),
                 learner=learner, learning_status=module.learning_status,
                 args=SimpleNamespace(evaluate=False), checkpoint_model=checkpoint,
                 finalize_coupled_run=module.finalize_coupled_run,
                 write_json=lambda path, data: records.append(copy.deepcopy(data)),
                 output=tmp_path, time=time, start=time.monotonic())
    with pytest.raises(RuntimeError, match='physical environment failed') as caught:
        exec(code, scope)
    assert caught.value is original
    assert records[-1]['state'] == 'failed'
    assert records[-1]['error'] == repr(original)
    assert 'checkpoint storage unavailable' in records[-1]['cleanup_errors'][0]['error']
    assert records[-1]['wm_updates'] == 300
    assert records[-1]['policy_updates'] == 0
    assert records[-1]['optimizer_steps']['world_model'] == 3003


@pytest.mark.parametrize('run_failure', [None, 'runtime', 'reset'])
@pytest.mark.parametrize('close_failure', [False, True])
def test_main_preserves_run_error_and_publishes_close_status(
        entry, monkeypatch, tmp_path, run_failure, close_failure):
    from cutting_world_model.lifecycle import ShutdownError

    module, _ = entry
    output = tmp_path / 'run'
    argv = ['train_coupled.py', '--sliceit-root', str(tmp_path),
            '--config', str(tmp_path / 'task.yaml'), '--output', str(output)]
    monkeypatch.setitem(sys.modules, 'rospy', SimpleNamespace(
        myargv=lambda: argv, init_node=lambda *args, **kwargs: None))
    monkeypatch.setattr(module.signal, 'signal', lambda *args: None)
    primary = RuntimeError('reset failed' if run_failure == 'reset' else 'physical step failed')
    closing = ShutdownError({'close': {'state': 'error', 'error': 'controller unavailable'}})
    calls = []

    def close():
        calls.append('close')
        if close_failure:
            raise closing
        return {'close': {'state': 'ok'}}

    def reset():
        raise primary

    monkeypatch.setattr(module, 'CoupledEnvironment', lambda config: SimpleNamespace(
        close=close, reset=reset))
    if run_failure != 'reset':
        def run(args, env, path):
            module.write_json(path / 'progress.json', {
                'state': 'running' if run_failure else 'completed',
                'iteration': 80, 'updates': 0, 'wm_updates': 16, 'wall_seconds': 3.,
                'cleanup_errors': [{'stage': 'save', 'error': 'earlier saved error'}]})
            if run_failure:
                raise primary
        monkeypatch.setattr(module, 'run', run)

    tracking_path = Path(__file__).resolve().parents[1] / 'scripts/train_coupled_tracked.py'
    tree = ast.parse(tracking_path.read_text())
    tracked_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name in ('tracked_write', 'learning_metrics')]
    tracking_scope = dict(write_json=module.write_json, tracker=SimpleNamespace(log=lambda values: None),
                          last_status={}, phase_codes={})
    exec(compile(ast.Module(body=tracked_functions, type_ignores=[]), str(tracking_path), 'exec'),
         tracking_scope)
    monkeypatch.setattr(module, 'write_json', tracking_scope['tracked_write'])

    if run_failure or close_failure:
        with pytest.raises(type(primary) if run_failure else ShutdownError) as caught:
            module.main()
        assert caught.value is (primary if run_failure else closing)
    else:
        module.main()
    assert calls == ['close']
    status = json.loads((output / 'progress.json').read_text())
    assert tracking_scope['last_status'] == status
    assert status['state'] == ('failed' if run_failure or close_failure else 'completed')
    assert status['shutdown']['close']['state'] == ('error' if close_failure else 'ok')
    if run_failure:
        assert status['error'] == repr(primary)
    if close_failure:
        assert status['cleanup_errors'][-1] == {'stage': 'close', 'error': repr(closing)}
    if run_failure != 'reset':
        assert status['iteration'] == 80 and status['wm_updates'] == 16
        assert status['cleanup_errors'][0]['stage'] == 'save'
