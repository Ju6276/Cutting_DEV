"""Reference collection must remain paired, auditable and cleaned up on failure."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/compare_free_cut_references.py'
spec = importlib.util.spec_from_file_location('reference_comparison', SCRIPT)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def condition():
    return dict(youngs_modulus=1., cut_spring_stiffness=1., density=1.,
                target_speed_mps=.03, start_lift_m=.01)


@pytest.mark.parametrize('field,value', [('density', 2.), ('target_speed_mps', float('nan')),
                                         ('start_lift_m', float('inf'))])
def test_conditions_reject_differences_and_nonfinite(field, value):
    actual = condition()
    actual[field] = value
    with pytest.raises(ValueError, match=field):
        ref.check_conditions([condition()], [actual], 'press')


def test_each_reference_is_checked_independently():
    ref.check_conditions([condition()], [condition()], 'press_saw')
    with pytest.raises(ValueError, match='press:'):
        ref.check_conditions([condition()], [], 'press')
    with pytest.raises(ValueError, match='order'):
        ref.check_conditions([dict(condition(), world=1)], [condition()], 'press')


class FakeEnv:
    def __init__(self, fail=False, worlds=1):
        self.fail = fail
        self.worlds = worlds
        self.closed = False
        self.resets = 0
        self.k = 0
        n = worlds
        material = SimpleNamespace(reference_linear_velocity=torch.tensor([0., -.03, 0.]),
                                   reference_duration=1.)
        self.physics = SimpleNamespace(
            material=material, dtype=torch.float32, initial_lift=torch.zeros(n),
            coupling=SimpleNamespace(base_from_disect=torch.tensor([0., 0., 0., 0., 0., 0., 1.])),
            reference_start=torch.tensor([[0., 0., 0., 0., 0., 0., 1.]]).repeat(n, 1),
            sim_time=torch.zeros(n))
        self.members = [SimpleNamespace(env=SimpleNamespace(physics=self.physics))]
        self.current = torch.zeros(n, 1)

    def conditions(self, indices):
        return [condition() for _ in indices]

    def step(self, action, reference):
        assert reference
        if self.fail:
            raise RuntimeError('injected step failure')
        self.k += 1
        self.physics.sim_time += .05
        self.current = torch.full((self.worlds, 1), float(self.k))
        # World 0 ends immediately; world 1 ends on step 2.
        done = torch.arange(self.worlds) < self.k
        return (self.current, torch.ones(self.worlds), done, torch.zeros_like(done),
                dict(separated=done, collision=torch.zeros_like(done),
                     outside=torch.zeros_like(done), timeout=torch.zeros_like(done)))

    def reset(self, ended):
        self.resets += 1
        return torch.full((self.worlds, 1), -999.)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_spec(monkeypatch):
    monkeypatch.setattr(ref.MultiConditionEnvironment, 'goal', staticmethod(lambda f: f.clone()))
    return SimpleNamespace(free_task=SimpleNamespace(horizon=2), pack_batch=lambda f: f.clone(),
                           pack_batch_transition=lambda f, state: f.clone(),
                           reset_batch_history=lambda f, state, ended: torch.where(ended[:, None], f, state))


def install(monkeypatch, env):
    monkeypatch.setattr(ref.MultiConditionEnvironment, 'from_config', staticmethod(lambda *a, **k: env))


def test_collection_saves_terminal_state_before_reset_and_first_episodes_only(monkeypatch, fake_spec, tmp_path):
    env = FakeEnv(worlds=2)
    install(monkeypatch, env)
    result = ref.run_reference({}, fake_spec, 0., 0., 'cpu',
                               expected_conditions=[condition(), condition()])
    data, worlds, times, _ = result
    assert env.closed and env.resets == 1
    assert worlds.tolist() == [0, 1, 1]
    assert times.tolist() == [0, 0, 1]
    assert data['next_state'][:, 0].tolist() == [1., 1., 2.]
    assert data['terminated'].tolist() == [True, False, True]
    ref.save_reference(tmp_path, 'press', result, 0., 0.)
    with np.load(tmp_path / 'press_transitions.npz') as saved:
        assert saved['next_state'][:, 0].tolist() == [1., 1., 2.]
        assert 'reference_target' in saved
        assert 'action' not in saved
    meta = json.loads((tmp_path / 'press_conditions.json').read_text())
    assert len(meta['conditions']) == 2
    assert meta['control']['mode'] == 'reference'


@pytest.mark.parametrize('mode', ['step', 'conditions', 'timeout', 'groups'])
def test_cleanup_on_all_collection_exit_paths(monkeypatch, fake_spec, mode):
    env = FakeEnv(fail=mode == 'step')
    if mode == 'groups':
        env.members = []
    install(monkeypatch, env)
    kwargs = {}
    if mode == 'conditions':
        kwargs['expected_conditions'] = [dict(condition(), density=2.)]
    if mode == 'timeout':
        kwargs['deadline'] = -1
    with pytest.raises((RuntimeError, ValueError, TimeoutError)):
        ref.run_reference({}, fake_spec, 0., 0., 'cpu', **kwargs)
    assert env.closed
