"""Configuration rejection and real worker shutdown, without running material physics."""
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import time

import pytest
import torch
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'backends/mujoco'))
from cutting_world_model import multi_env
from cutting_world_model.task_heads import validate_force_limits


@pytest.mark.parametrize('value', [
    None, [], [1.] * 5, [1.] * 7, [1., 1., 1., 1., 1., 0.],
    [-1.] * 6, [float('nan')] * 6, [float('inf')] * 6, [True] * 6,
])
def test_multi_object_limits_rejected_before_physics(monkeypatch, value):
    def forbidden(*args, **kwargs):
        raise AssertionError('Physics/assets must not be opened for invalid limits')
    monkeypatch.setattr(multi_env, 'knife_clearance', forbidden)
    config = {'groups': [{'name': 'fixture', 'scene': 'not-on-disk', 'worlds': 1}]}
    if value is not None:
        config['force_limits'] = value
    with pytest.raises(ValueError, match='six finite positive'):
        multi_env.MultiConditionEnvironment.from_config(ROOT, config)


def test_single_scene_keeps_legacy_default_and_explicit_limits_copy():
    assert validate_force_limits(None) is None
    limits = [60., 70., 80., 4., 5., 6.]
    result = validate_force_limits(limits, required=True)
    assert result == limits and result is not limits


def _worker(connection, mode):
    if mode == 'crash':
        os._exit(7)
    connection.send(('ready', None))
    if mode == 'stuck':
        time.sleep(60)
    else:
        assert connection.recv()[0] == 'close'
    connection.close()


def _start_worker(mode):
    group = multi_env.ProcessGroup.__new__(multi_env.ProcessGroup)
    context = mp.get_context('spawn')
    group.connection, child = context.Pipe()
    group.process = context.Process(target=_worker, args=(child, mode))
    group._closed = False
    group.process.start()
    child.close()
    return group


def test_process_group_close_is_idempotent():
    group = _start_worker('normal')
    try:
        assert group._receive_message()[0] == 'ready'
        group.close(timeout=5)
        group.close(timeout=5)
        assert not group.process.is_alive()
        assert group.process.exitcode == 0
        assert group.connection.closed
    finally:
        if group.process.is_alive():
            group.process.kill()
            group.process.join()


def test_worker_crash_does_not_leave_parent_waiting():
    group = _start_worker('crash')
    try:
        with pytest.raises(RuntimeError, match='Object group'):
            group._receive_message()
        group.close(timeout=5)
        assert not group.process.is_alive()
    finally:
        if group.process.is_alive():
            group.process.kill()
            group.process.join()


def test_stuck_worker_is_reaped_on_close():
    group = _start_worker('stuck')
    try:
        assert group._receive_message()[0] == 'ready'
        with pytest.warns(RuntimeWarning, match='terminating'):
            group.close(timeout=.2)
        assert not group.process.is_alive()
        assert group.connection.closed
    finally:
        if group.process.is_alive():
            group.process.kill()
            group.process.join()


def test_later_group_initialization_failure_closes_started_workers(monkeypatch):
    members = []
    class Member:
        def __init__(self, *args):
            self.closed = False
            members.append(self)
        def finish_start(self, device):
            raise RuntimeError('scene initialization failed')
        def close(self):
            self.closed = True
    monkeypatch.setattr(multi_env, 'ProcessGroup', Member)
    monkeypatch.setattr(multi_env, 'knife_clearance', lambda *args: 0.)
    groups = [multi_env.ObjectGroup(name=str(i), scene='fixture', worlds=1) for i in range(2)]
    with pytest.raises(RuntimeError, match='scene initialization failed'):
        multi_env.MultiConditionEnvironment(ROOT, groups, force_limits=[1.] * 6, parallel=True)
    assert len(members) == 2 and all(m.closed for m in members)


def test_training_setup_failure_closes_environment(monkeypatch, tmp_path):
    sys.path.append(str(ROOT / 'scripts'))
    from scripts import train_gpu
    from cutting_world_model import gpu_env
    env = SimpleNamespace(closed=False)
    def close():
        env.closed = True
    env.close = close
    monkeypatch.setattr(gpu_env, 'GPUEnvironment', lambda *args, **kwargs: env)
    def fail(*args, **kwargs):
        raise ValueError('bad checkpoint')
    monkeypatch.setattr(train_gpu, 'run_with_environment', fail)
    args = SimpleNamespace(backend_root=ROOT / 'backends/mujoco', seed=0,
                           force_penalty='sliceit', conditions=None, worlds=1)
    with pytest.raises(ValueError, match='bad checkpoint'):
        train_gpu.run(args, tmp_path, {}, time.monotonic())
    assert env.closed


def test_close_attempts_all_groups_if_one_cleanup_fails():
    environment = multi_env.MultiConditionEnvironment.__new__(multi_env.MultiConditionEnvironment)
    closed = []
    def fail():
        closed.append('first')
        raise RuntimeError('failed cleanup')
    environment.members = [SimpleNamespace(close=fail),
                           SimpleNamespace(close=lambda: closed.append('second'))]
    with pytest.raises(RuntimeError, match='Failed to close object group'):
        environment.close()
    assert closed == ['first', 'second']
