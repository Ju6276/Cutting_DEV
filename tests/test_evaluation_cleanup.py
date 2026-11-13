import json
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from cutting_world_model.evaluation_lifecycle import evaluation_environment
from scripts import evaluate_multi


@pytest.mark.parametrize('failure', ['normal', 'step', 'timeout', 'interrupt', 'max_steps'])
def test_evaluation_entrypoint_closes_environment_and_preserves_rows(monkeypatch, tmp_path, failure):
    original_zeros = torch.zeros
    def zeros(*args, **kwargs):
        kwargs['device'] = 'cpu'
        return original_zeros(*args, **kwargs)
    monkeypatch.setattr(torch, 'zeros', zeros)
    frame = dict(observation=zeros(1, 24), target_speed=torch.tensor([[.03]]),
                 total_springs=torch.tensor([[10.]]))
    heads = SimpleNamespace(max_distance=torch.ones(3), env={'goal_threshold': [.002]*3},
                            cutting_speed=lambda obs: zeros(1))
    class Env:
        worlds = 1
        current = frame
        calls = 0
        closes = 0
        def conditions(self, worlds):
            return [dict(group='test')]
        def step(self, action):
            self.calls += 1
            if self.calls == 2:
                if failure == 'step':
                    raise RuntimeError('step failed')
                if failure == 'interrupt':
                    raise KeyboardInterrupt
            info = {k: torch.tensor([False]) for k in ('collision', 'outside')}
            info.update(success=torch.tensor([True]), peak_force=zeros(1), safety_margin=zeros(1),
                        material_impulse=zeros(1), released_springs=torch.tensor([10]))
            return frame, zeros(1), torch.tensor([True]), torch.tensor([False]), info
        def reset(self, ended):
            return frame
        def close(self):
            self.closes += 1
    env = Env()
    env.heads = heads
    class Policy:
        history = 10
        def __init__(self, *args): pass
        def to(self, device): return self
        def load_state_dict(self, state): pass
        def eval(self): pass
        def __call__(self, history, goal): return zeros(1, 6)
    spec = SimpleNamespace(execute=lambda a: a)
    spec.to = lambda device: spec
    monkeypatch.setattr(evaluate_multi, 'CuttingWorldModel', lambda metadata: SimpleNamespace(spec=spec))
    monkeypatch.setattr(evaluate_multi, 'SliceItPolicy', Policy)
    monkeypatch.setattr(evaluate_multi, 'MultiConditionEnvironment', SimpleNamespace(
        from_config=lambda *args, **kwargs: env, goal=lambda frame: zeros(1, 8)))
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: dict(
        metadata=dict(goal_dim=8, state_shapes={'observation': [24]},
                      environment={'agent_control_dt': .05}), learner={'policy': {}}))
    monkeypatch.setattr(evaluate_multi, 'git_revision', lambda root: 'test')
    times = iter([0., 1., 4000.])
    monkeypatch.setattr(evaluate_multi.time, 'monotonic',
                        (lambda: next(times)) if failure == 'timeout' else (lambda: 0.))
    conditions = tmp_path / 'conditions.json'
    conditions.write_text('{}')
    out = tmp_path / 'out'
    args = ['evaluate_multi.py', '--checkpoint', 'dummy.pt', '--conditions', str(conditions),
            '--output', str(out), '--episodes', '2', '--max-hours', '1']
    if failure == 'max_steps': args += ['--max-steps', '1']
    monkeypatch.setattr(sys, 'argv', args)
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    errors = {'step': RuntimeError, 'timeout': TimeoutError, 'interrupt': KeyboardInterrupt}
    if failure in errors:
        with pytest.raises(errors[failure]): evaluate_multi.main()
    else:
        evaluate_multi.main()
    assert env.closes == 1
    rows = [json.loads(line) for line in (out/'episodes.jsonl').read_text().splitlines()]
    assert len(rows) == (2 if failure == 'normal' else 1)
    assert {s: signal.getsignal(s) for s in handlers} == handlers
    assert (out/'summary.json').exists() == (failure not in errors)


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_actual_process_signal_closes_child_and_exits(tmp_path, signum):
    script = """
import os, signal, subprocess, sys
from pathlib import Path
from cutting_world_model.evaluation_lifecycle import evaluation_environment
class Env:
    def __init__(self):
        self.worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    def close(self):
        self.worker.terminate()
        self.worker.wait(timeout=3)
        Path(sys.argv[1]).write_text('worker reaped')
with evaluation_environment(Env):
    os.kill(os.getpid(), int(sys.argv[2]))
"""
    marker = tmp_path/'closed'
    result = subprocess.run([sys.executable, '-c', script, str(marker), str(int(signum))],
                            capture_output=True, text=True, timeout=10)
    # CPython re-raises SIGINT after printing KeyboardInterrupt (Popen reports -SIGINT).
    assert result.returncode == (-signal.SIGINT if signum == signal.SIGINT else 128 + signum)
    assert marker.read_text() == 'worker reaped'


def test_handlers_restored_if_factory_or_close_raises():
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    def fail(): raise RuntimeError('expected')
    with pytest.raises(RuntimeError):
        with evaluation_environment(fail): pass
    assert {s: signal.getsignal(s) for s in handlers} == handlers
    with pytest.raises(RuntimeError):
        with evaluation_environment(lambda: SimpleNamespace(close=fail)): pass
    assert {s: signal.getsignal(s) for s in handlers} == handlers
