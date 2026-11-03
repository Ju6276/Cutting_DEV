"""Shutdown faults must not skip checkpoint, status, or tracking finalization."""
import copy
import threading
import time
from types import SimpleNamespace

import pytest

from cutting_world_model.lifecycle import ShutdownError, close_environment
from cutting_world_model.lifecycle import finalize_training, run_bounded


def test_bounded_callback_runs_once_and_reports_success():
    calls = []
    result = run_bounded(lambda: calls.append('called'), timeout=.1)
    assert result['state'] == 'ok'
    assert calls == ['called']


def test_bounded_callback_reports_its_exception():
    def fail():
        raise RuntimeError('service disappeared')
    result = run_bounded(fail, timeout=.1)
    assert result['state'] == 'error'
    assert 'service disappeared' in result['error']


def test_blocked_callback_is_daemon_and_does_not_hold_up_caller():
    entered = threading.Event()
    release = threading.Event()
    daemon = []
    def block():
        daemon.append(threading.current_thread().daemon)
        entered.set()
        release.wait()
    started = time.monotonic()
    try:
        result = run_bounded(block, timeout=.03)
        assert entered.is_set()
        assert result['state'] == 'timeout'
        assert time.monotonic() - started < .5
        assert daemon == [True]
    finally:
        release.set()


def environment(unpause, close, stepped=True):
    env = SimpleNamespace(use_step_control=stepped,
                          robot_connection=SimpleNamespace(unpause=unpause), close=close)
    env.unwrapped = env
    return env


def test_normal_close_preserves_unpause_then_ros_shutdown():
    calls = []
    close_environment(environment(lambda: calls.append('unpause'),
                                  lambda: calls.append('close')), timeout=.1)
    assert calls == ['unpause', 'close']


def test_continuous_environment_closes_without_gazebo_unpause():
    calls = []
    close_environment(environment(lambda: calls.append('unpause'),
                                  lambda: calls.append('close'), stepped=False), timeout=.1)
    assert calls == ['close']


@pytest.mark.parametrize('failing_stage', ['unpause', 'close'])
def test_service_exception_does_not_skip_other_shutdown_stage(failing_stage):
    calls = []
    def call(name):
        calls.append(name)
        if name == failing_stage:
            raise RuntimeError(name + ' unavailable')
    with pytest.raises(ShutdownError):
        close_environment(environment(lambda: call('unpause'), lambda: call('close')), timeout=.1)
    assert calls == ['unpause', 'close']


@pytest.mark.parametrize('blocked_stage', ['unpause', 'close', 'both'])
def test_blocked_services_have_wall_deadlines_and_still_attempt_close(blocked_stage):
    release = threading.Event()
    calls = []
    def call(name):
        calls.append(name)
        if blocked_stage in (name, 'both'):
            release.wait()
    started = time.monotonic()
    try:
        with pytest.raises(ShutdownError):
            close_environment(environment(lambda: call('unpause'), lambda: call('close')), timeout=.03)
        assert calls == ['unpause', 'close']
        assert time.monotonic() - started < .5
    finally:
        release.set()


def test_successful_finalization_records_after_close_and_finishes_tracking():
    calls = []
    state = {'state': 'completed'}
    errors = finalize_training(
        lambda: calls.append('save'), lambda: calls.append('record'),
        lambda: calls.append('finish'), lambda: calls.append('close'), state)
    assert not errors
    assert calls == ['save', 'close', 'record', 'finish', 'record']
    assert state['state'] == 'completed'


@pytest.mark.parametrize('failed_stage', ['save', 'close', 'record', 'finish'])
def test_one_finalization_fault_never_skips_other_stages(failed_stage):
    calls = []
    records = []
    finishes = []
    state = {'state': 'completed', 'steps': 290}
    failed = False
    def call(name):
        nonlocal failed
        calls.append(name)
        if name == failed_stage and not failed:
            failed = True
            raise RuntimeError(name + ' failed')
        if name == 'record':
            records.append(copy.deepcopy(state))
        if name == 'finish':
            finishes.append(copy.deepcopy(state))
    errors = finalize_training(lambda: call('save'), lambda: call('record'),
                               lambda: call('finish'), lambda: call('close'), state)
    assert errors
    assert calls == ['save', 'close', 'record', 'finish', 'record']
    assert state['state'] == 'failed'
    assert failed_stage + ' failed' in repr(state['cleanup_errors'])
    assert records[-1]['state'] == 'failed'
    assert records[-1]['steps'] == 290
    if failed_stage != 'finish':
        assert finishes[-1]['state'] == 'failed'


def test_multiple_failures_are_retained_and_existing_failure_is_not_cleared():
    calls = []
    state = {'state': 'failed', 'exit_reason': 'original training failure'}
    def fail(name):
        calls.append(name)
        raise RuntimeError(name + ' failure')
    errors = finalize_training(lambda: fail('save'), lambda: fail('record'),
                               lambda: fail('finish'), lambda: fail('close'), state)
    assert errors
    assert calls == ['save', 'close', 'record', 'finish', 'record']
    assert state['state'] == 'failed'
    assert state['exit_reason'] == 'original training failure'
    for name in ('save', 'record', 'finish', 'close'):
        assert name + ' failure' in repr(state['cleanup_errors'])


@pytest.mark.parametrize('hook_state', ['error', 'timeout'])
def test_rospy_swallowed_hook_failure_is_reported_by_environment_close(hook_state):
    arm = SimpleNamespace()
    def close():
        arm.shutdown_result = {'state': hook_state, 'error': 'controller hook failed'}
    env = environment(lambda: None, close)
    env.ur3e_arm = arm
    with pytest.raises(ShutdownError) as caught:
        close_environment(env, timeout=.1)
    assert 'controller hook failed' in repr(caught.value.results)


def test_environment_close_prefers_strict_shutdown_unpause_over_runtime_method():
    calls = []
    env = environment(lambda: calls.append('runtime_unpause'), lambda: calls.append('close'))
    env.robot_connection.unpause_for_shutdown = lambda: calls.append('strict_unpause')
    close_environment(env, timeout=.1)
    assert calls == ['strict_unpause', 'close']


def test_strict_unpause_error_is_recorded_and_ros_close_still_runs():
    calls = []
    env = environment(lambda: calls.append('runtime_unpause'), lambda: calls.append('close'))
    def unavailable():
        calls.append('strict_unpause')
        raise RuntimeError('strict unpause service failed')
    env.robot_connection.unpause_for_shutdown = unavailable
    with pytest.raises(ShutdownError) as caught:
        close_environment(env, timeout=.1)
    assert calls == ['strict_unpause', 'close']
    assert 'strict unpause service failed' in repr(caught.value.results)
