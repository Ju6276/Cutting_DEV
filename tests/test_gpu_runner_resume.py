"""Run the real GPU runner control flow with CPU doubles for expensive physics/learners.

The tests exercise on-disk milestone checkpoints/replay, ordering at resume, and
cleanup status. Physics and neural update correctness have separate GPU tests.
"""
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def runner(monkeypatch):
    sys.path.append(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from scripts import train_gpu as m
    from cutting_world_model.online_replay import BatchedOnlineReplay
    events = []
    zeros = torch.zeros
    monkeypatch.setattr(torch, 'zeros', lambda *a, **kw: zeros(*a, **{**kw, 'device': 'cpu'}))
    monkeypatch.setattr(torch.cuda, 'get_rng_state', torch.get_rng_state)
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(m.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(m, 'git_revision', lambda *a: 'test')
    monkeypatch.setattr(m, 'BatchedOnlineReplay', lambda n, d, device, **kw: BatchedOnlineReplay(n, d, 'cpu', **kw))
    meta = {'environment': {}, 'state_shapes': {'observation': [34]}, 'goal_dim': 8}
    monkeypatch.setattr(m.ControlStateSpec, 'metadata_from_frame', lambda *a: dict(meta))

    class Spec:
        state_dim = delta_dim = 1
        def pack_batch(self, f): return f['state']
        def pack_batch_transition(self, f, s): return f['state']
        def execute(self, a): return a

    class WM:
        terminal_gradient = 'dynamics'
        def __init__(self, *a): self.spec = Spec()
        def to(self, *a): return self

    class Policy:
        history = 10
        def __init__(self, *a): pass
        def to(self, *a): return self
        def __call__(self, h, goal): return torch.zeros(len(goal), 6)

    @dataclass
    class Config:
        window: int = 1
        lr: float = .0002

    class Learner:
        def __init__(self, wm, policy, critic, microbatch, **kw):
            self.wm, self.policy = wm, policy
            self.wcfg = self.pcfg = Config()
            self.wm_only_updates = kw['wm_only_updates']
            self.wm_updates = self.updates = 0
            self.risk_head = None
            self.critic_prefit_pending = False
            for k in ('critic_microbatch','actor_horizon','actor_bc_alpha','actor_objective'):
                setattr(self, k, kw[k])
        @property
        def should_update_policy(self): return self.wm_updates > self.wm_only_updates
        @property
        def stage(self): return 'joint' if self.should_update_policy else 'world_model_only'
        @property
        def optimizer_steps(self): return {'world_model': self.wm_updates, 'policy': self.updates, 'critic': self.updates}
        def set_lr(self, lr): pass
        def fit_world_model(self, replay):
            self.progress(phase='wm')
            self.wm_updates += 1
            events.append(('wm', self.wm_updates))
            return {'loss': 0.}
        def improve_policy(self, replay):
            self.updates += 1
            events.append(('policy', self.updates))
            return {'loss': 0.}
        def critic_variant(self): return {'target': 'real'}
        def checkpoint(self): return {'wm_updates': self.wm_updates, 'updates': self.updates}
        def restore(self, saved, **kw):
            self.wm_updates, self.updates = saved['wm_updates'], saved['updates']

    class Tracker:
        def __init__(self, *a): pass
        def log(self, metrics): pass
        def finish(self, checkpoint, status): events.append(('finish', status['state']))

    class Env:
        metadata = meta
        condition_sampler = None
        def __init__(self, fail_close=False):
            self.current = {'state': torch.zeros(3, 1), 'observation': torch.zeros(3, 34)}
            self.steps = 0
            self.fail_close = fail_close
        def step(self, action):
            self.steps += 1
            events.append(('step', self.steps))
            self.current['state'] = self.current['state'] + 1
            flag = torch.zeros(3, dtype=torch.bool)
            return self.current, torch.zeros(3), flag, flag, {
                'peak_force': torch.zeros(3), 'safety_margin': torch.zeros(3), 'collision': flag}
        def close(self):
            events.append(('close', None))
            if self.fail_close: raise RuntimeError('shutdown failed')

    monkeypatch.setattr(m, 'CuttingWorldModel', WM)
    monkeypatch.setattr(m, 'SliceItPolicy', Policy)
    monkeypatch.setattr(m, 'SliceItCritic', Policy)
    monkeypatch.setattr(m, 'OnlineLearner', Learner)
    monkeypatch.setattr(m, 'ExperimentTracker', Tracker)

    def execute(out, resume=None, steps=3, fail_close=False):
        args = SimpleNamespace(snapshot_at=[], worlds=3, microbatch=2, wm_only_updates=1,
            critic_target='real', critic_microbatch=2, actor_horizon=8, actor_bc_alpha=.25,
            actor_objective='model', wm_terminal_gradient='dynamics', risk_constraint=False,
            schedule='transitions', warmup_transitions=6, transitions_per_update=1,
            resume=resume, allow_substep_change=False, critic_reset=False,
            change_wm_terminal_gradient=False, critic_prefit_rounds=0, resume_without_replay=False,
            backend_root=Path(__file__).resolve().parents[1]/'backends/mujoco',
            wandb_project='test', wandb_entity=None, wandb_mode='disabled', max_hours=1,
            max_iterations=steps, max_transitions=None, checkpoint_every=0,
            cuda_cache_release='always', cuda_cache_reserve_gib=0., gpu_headroom_margin_gib=1.)
        out.mkdir()
        status = {'state': 'initializing'}
        env = Env(fail_close)
        m.run_with_environment(args, out, status, time.monotonic(), env, None,
                               lambda f: torch.zeros(3, 8))
        return torch.load(out/'last.pt', weights_only=False), env, status
    return execute, events


def test_real_runner_milestone_resume_drains_before_any_new_physics(runner, tmp_path):
    run, events = runner
    complete, _, _ = run(tmp_path/'continuous')
    mid = tmp_path/'continuous/wm_only_complete.pt'
    saved = torch.load(mid, weights_only=False)
    assert saved['learner']['wm_updates'] == 1
    assert len(saved['update_queue']['pending']) == 2
    assert saved['update_queue']['active'] is None
    events.clear()
    resumed, env, _ = run(tmp_path/'resumed', mid, steps=0)
    assert env.steps == 0
    assert resumed['learner'] == complete['learner'] == {'wm_updates': 3, 'updates': 2}
    assert events[:4] == [('wm', 2), ('policy', 1), ('wm', 3), ('policy', 2)]
    assert resumed['update_queue']['pending'] == []


def test_real_runner_cleanup_failure_reaches_progress_and_tracker(runner, tmp_path):
    run, events = runner
    with pytest.raises(RuntimeError, match='finalization failed'):
        run(tmp_path/'close_failure', steps=0, fail_close=True)
    progress = json.loads((tmp_path/'close_failure/progress.json').read_text())
    assert progress['state'] == 'failed'
    assert progress['cleanup_errors'][0]['stage'] == 'close'
    assert events[-2:] == [('close', None), ('finish', 'failed')]
