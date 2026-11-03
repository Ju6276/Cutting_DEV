import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutting_world_model.coupled_env import CoupledEnvironment
from cutting_world_model.lifecycle import ShutdownError


@pytest.mark.parametrize('failure', ['mode', 'snapshot_service'])
def test_constructor_failure_closes_source_environment(monkeypatch, failure):
    events = []
    base = SimpleNamespace(
        use_step_control=True,
        ur3e_arm=SimpleNamespace(use_step_control=True),
        robot_connection=SimpleNamespace(unpause=lambda: events.append('unpause')))
    space = SimpleNamespace(seed=lambda seed: None)
    env = SimpleNamespace(unwrapped=base, action_space=space, observation_space=space,
                          seed=lambda seed: None, close=lambda: events.append('close'))
    config = {'env_id': 'test', 'steps_per_episode': 100,
              'use_step_control': failure != 'mode'}

    def unavailable(timeout):
        raise RuntimeError('snapshot unavailable')

    monkeypatch.setitem(sys.modules, 'rospy', SimpleNamespace(
        get_param=lambda name: config,
        ServiceProxy=lambda *args: SimpleNamespace(wait_for_service=unavailable)))
    monkeypatch.setitem(sys.modules, 'std_srvs.srv', SimpleNamespace(Trigger=object))
    monkeypatch.setitem(sys.modules, 'ur3e_openai.common', SimpleNamespace(
        load_environment=lambda *args, **kwargs: env,
        clear_gym_params=lambda *args: None,
        load_ros_params=lambda **kwargs: None))
    monkeypatch.setenv('ROS_MASTER_URI', 'http://127.0.0.1:11322')
    message = 'runtime step-control' if failure == 'mode' else 'snapshot unavailable'
    with pytest.raises(RuntimeError, match=message):
        CoupledEnvironment('task.yaml')
    assert events == ['unpause', 'close']


@pytest.mark.parametrize('stepped', [False, True])
def test_close_allows_source_controller_shutdown_tick(stepped):
    physics = {'paused': stepped}

    def unpause():
        physics['paused'] = False

    def close():
        assert not physics['paused']
        physics['closed'] = True

    base = SimpleNamespace(use_step_control=stepped,
                           robot_connection=SimpleNamespace(unpause=unpause))
    coupled = CoupledEnvironment.__new__(CoupledEnvironment)
    coupled.env = SimpleNamespace(unwrapped=base, close=close)
    coupled.close()
    assert physics['closed']


@pytest.mark.parametrize('fail_unpause, fail_close', [(True, False), (False, True), (True, True)])
def test_close_attempts_both_operations_and_reports_failures(fail_unpause, fail_close):
    events = []

    def unpause():
        events.append('unpause')
        if fail_unpause:
            raise RuntimeError('Gazebo unavailable')

    def close():
        events.append('close')
        if fail_close:
            raise RuntimeError('Controller shutdown failed')

    base = SimpleNamespace(use_step_control=True,
                           robot_connection=SimpleNamespace(unpause=unpause))
    coupled = CoupledEnvironment.__new__(CoupledEnvironment)
    coupled.env = SimpleNamespace(unwrapped=base, close=close)
    with pytest.raises(ShutdownError) as captured:
        coupled.close()
    assert events == ['unpause', 'close']
    results = captured.value.results
    assert results['unpause']['state'] == ('error' if fail_unpause else 'ok')
    assert results['close']['state'] == ('error' if fail_close else 'ok')


def test_close_reaches_environment_close_when_unpause_blocks(monkeypatch):
    import threading
    from cutting_world_model import coupled_env
    from cutting_world_model.lifecycle import close_environment

    release = threading.Event()
    closed = threading.Event()
    base = SimpleNamespace(use_step_control=True,
                           robot_connection=SimpleNamespace(unpause=release.wait))
    coupled = CoupledEnvironment.__new__(CoupledEnvironment)
    coupled.env = SimpleNamespace(unwrapped=base, close=closed.set)
    monkeypatch.setattr(coupled_env, 'close_environment',
                        lambda env: close_environment(env, timeout=.02))
    try:
        with pytest.raises(ShutdownError) as captured:
            coupled.close()
        assert captured.value.results['unpause']['state'] == 'timeout'
        assert captured.value.results['close']['state'] == 'ok'
        assert closed.is_set()
    finally:
        release.set()


def test_training_setup_error_closes_constructed_environment(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/train_coupled.py'
    spec = importlib.util.spec_from_file_location('lifecycle_train_coupled', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = ['train_coupled.py', '--sliceit-root', str(tmp_path),
            '--config', str(tmp_path / 'task.yaml'), '--output', str(tmp_path / 'run')]
    monkeypatch.setitem(sys.modules, 'rospy', SimpleNamespace(
        myargv=lambda: argv, init_node=lambda *args, **kwargs: None))
    monkeypatch.setattr(module.signal, 'signal', lambda *args: None)
    closed = []
    env = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(module, 'CoupledEnvironment', lambda config: env)

    def fail_setup(*args):
        raise ValueError('checkpoint metadata mismatch')

    monkeypatch.setattr(module, 'run', fail_setup)
    with pytest.raises(ValueError, match='checkpoint metadata mismatch'):
        module.main()
    assert closed == [True]
