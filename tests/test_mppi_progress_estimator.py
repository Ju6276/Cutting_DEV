from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from cutting_world_model.mpc import EstimatedProgressState, MPPIConfig, MPPIPolicy
from cutting_world_model.world_model import CuttingWorldModel
from test_free_cut_v3 import v3_fixture


def fixture():
    wm, state, goal = v3_fixture()
    spec = wm.spec
    frame = {key: spec.field(state, key).detach().reshape(1, *spec.shapes[key])
             for key in spec.base_fields}
    frame.update(target_pose=goal[:, :7], target_speed=goal[:, 7:])
    return wm, frame, goal


def test_progress_predictor_uses_executed_action_and_measured_history():
    wm, frame, goal = fixture()
    estimator = EstimatedProgressState(wm)
    hidden = {'cut_progress', 'cut_spring_ke', 'particle_q', 'success_counter'}
    class Restricted(dict):
        def __getitem__(self, key):
            assert key not in hidden, 'Privileged state read: ' + key
            return super().__getitem__(key)
    observable = Restricted({k: v for k, v in frame.items() if k not in hidden})
    state = estimator.reset(observable)
    assert not wm.spec.field(state, 'cut_progress').any()
    # Make progress depend on action; verify it follows the ACTUAL executed action.
    def prediction(s, a, g):
        nxt = s.clone()
        rho = wm.spec.field(s, 'cut_progress')
        wm.spec.field(nxt, 'cut_progress').copy_(rho + (1-rho)*(a[:, :1]+1)*.1)
        return nxt, s.new_zeros(1), s.new_zeros(1)
    wm.predict = prediction
    a = state.new_zeros(1, 6)
    a[:, 0] = .5
    estimator.predict_executed(state, a, goal)
    next_frame = Restricted({k: v.clone() for k, v in observable.items()})
    next_frame['joint_q'] += .01
    next_frame['step_count'] += 1
    nxt = estimator.observe(next_frame, state)
    torch.testing.assert_close(wm.spec.field(nxt, 'cut_progress'), state.new_tensor([[.15]]))
    torch.testing.assert_close(wm.spec.field(nxt, 'joint_q'), next_frame['joint_q'])
    history = wm.spec.field(nxt, 'control_history')
    torch.testing.assert_close(history[:, -1], state[:, :wm.spec.base_dim])
    a[:, 0] = -1
    estimator.predict_executed(nxt, a, goal)
    twice = estimator.observe(next_frame, nxt)
    torch.testing.assert_close(wm.spec.field(twice, 'cut_progress'), state.new_tensor([[.15]]))
    reset = estimator.reset(observable)
    assert not wm.spec.field(reset, 'cut_progress').any()
    assert not reset.reshape(1, wm.spec.history_length, wm.spec.base_dim)[..., wm.spec.layout['cut_progress']].any()
    with pytest.raises(RuntimeError, match='preceding'):
        estimator.observe(next_frame, reset)


def test_estimator_rejects_replanning_without_observation_and_nonfinite_prediction():
    wm, frame, goal = fixture()
    estimator = EstimatedProgressState(wm)
    state = estimator.reset(frame)
    action = state.new_zeros(1, 6)
    estimator.predict_executed(state, action, goal)
    with pytest.raises(RuntimeError, match='Observe'):
        estimator.predict_executed(state, action, goal)
    estimator.reset(frame)
    def bad(s, a, g):
        nxt = s.clone()
        wm.spec.field(nxt, 'cut_progress').fill_(float('nan'))
        return nxt, s.new_zeros(1), s.new_zeros(1)
    wm.predict = bad
    with pytest.raises(FloatingPointError, match='progress'):
        estimator.predict_executed(state, action, goal)


def test_real_mppi_actions_invariant_to_missing_or_corrupted_true_progress(tmp_path, monkeypatch):
    pytest.importorskip('pytorch_mppi')
    wm, frame, goal = fixture()
    meta = deepcopy(wm.spec.metadata)
    path = tmp_path/'wm.pt'
    torch.save({'metadata': meta, 'learner': {'world_model': wm.state_dict(),
               'policy': {'w': torch.ones(1)}, 'updates': 0,
               'training_stage': {'wm_updates': 341}}}, path)
    monkeypatch.setattr('cutting_world_model.mpc.CuttingWorldModel',
                        lambda m: CuttingWorldModel(m, hidden=(12, 12)))
    env = SimpleNamespace(worlds=1, metadata=meta, current=frame, device=torch.device('cpu'))
    config = MPPIConfig(horizon=3, samples=16, discount=1., progress_source='estimated')
    planner = MPPIPolicy(path, env, config, path)
    assert not planner.manifest['information']['uses_simulator_cut_progress']
    original = {k: v.float() for k, v in frame.items()}
    def run(mode):
        observations = {k: v.clone() for k, v in original.items()}
        if mode == 'missing':
            del observations['cut_progress']
        else:
            observations['cut_progress'].fill_(float('nan'))
            observations['success_counter'].fill_(999)
        planner.reset(observations)
        actions, states = [], []
        for t in range(3):
            torch.manual_seed(12+t)
            actions.append(planner.act(observations).clone())
            observations['joint_q'] += .001
            observations['step_count'] += 1
            planner.observe(observations)
            states.append(planner.state.clone())
        return torch.stack(actions), torch.stack(states)
    a, s = run('missing')
    b, t = run('corrupted')
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(s, t, rtol=0, atol=0)
    assert planner.unchanged()
    assert all(p.grad is None and not p.requires_grad for p in planner.wm.parameters())
    assert 0 < float(planner.spec.field(planner.state, 'cut_progress')[0, 0]) < 1
