import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cutting_world_model.free_cut import FreeCutTask
from cutting_world_model.world_model import CuttingWorldModel
from test_free_cut_v2 import v2_fixture


def config():
    return json.loads((Path(__file__).parents[1] / 'configs/free_cut/review_v3.json').read_text())['task']


def v3_fixture():
    old, state, goal = v2_fixture()
    meta = copy.deepcopy(old.spec.metadata)
    meta['free_cut_task']['version'] = 'free_cut_v3'
    meta['free_cut_task']['weights']['speed'] = 20
    return CuttingWorldModel(meta, hidden=(12, 12)).double(), state, goal


def task(dt=.05):
    # Direct physical speed input, isolate the task equation from observation encoding.
    head = SimpleNamespace(env={'agent_control_dt': dt}, force_limits=[50]*3+[5]*3,
                           cutting_speed=lambda obs: obs[..., 0],
                           jerkiness=lambda obs: obs[..., 1])
    return FreeCutTask(config(), head)


def trajectory_return(ratio, steps, dt=.05):
    t = task(dt)
    before = torch.arange(steps, dtype=torch.float64)/steps
    after = before + 1/steps
    obs = torch.zeros(steps, 2, dtype=torch.float64)
    obs[:, 0] = ratio*.03
    r = t.step_reward(before, after, obs, torch.zeros(steps, 6), torch.full((steps,), .03))
    return float(r.sum())+t.weights['success']


def test_fast_completion_cannot_avoid_progress_speed_cost():
    normal = trajectory_return(1, 40)
    fast = trajectory_return(20, 2)
    assert normal-fast > 16
    assert normal-trajectory_return(3, 14) > 5
    # Reciprocal speed deviations have the same progress cost; time cost is disclosed.
    assert trajectory_return(3, 20) == pytest.approx(trajectory_return(1/3, 20), abs=1e-5)
    assert trajectory_return(.25, 90) < normal-12


def test_progress_cost_does_not_depend_on_how_release_is_split_into_steps():
    t = task()
    def value(increments):
        after = torch.tensor(increments).cumsum(0)
        before = torch.cat((torch.zeros(1), after[:-1]))
        obs = torch.tensor([[.6, 0.]]).repeat(len(after), 1)
        return t.step_reward(before, after, obs, torch.zeros(len(after), 6), torch.full((len(after),), .03)).sum()
    # Same number of elapsed steps, same total release, different damage timing.
    assert value([1., 0., 0., 0.]) == pytest.approx(float(value([.25]*4)), abs=1e-5)


def test_finite_penalty_bound_rejects_failure_incentive():
    head = task().heads
    cfg = config()
    cfg['weights']['speed'] = 60
    with pytest.raises(ValueError, match='penalty bound'):
        FreeCutTask(cfg, head)
    t = task()
    bound = 2*t.weights['speed'] + t.horizon*t.dt*(t.weights['load']+t.weights['jerk'])
    assert -bound > t.weights['progress']-t.weights['failure']
    assert t.weights['success']-bound > t.weights['progress']


def test_completion_bound_applies_to_fitting_reward_and_survival_logits():
    wm, state, goal = v3_fixture()
    with torch.no_grad():
        wm.term_head.net[-1].weight.zero_()
        wm.term_head.net[-1].bias.fill_(30.)
    nxt, reward, logits = wm.step(state, state.new_zeros(1, 6), goal)
    rho = wm.spec.field(nxt, 'cut_progress')[:, 0]
    assert rho.item() == pytest.approx(.001)
    assert logits.sigmoid().item() <= rho.item()+1e-12
    assert wm.spec.completion_reward(nxt, logits).item() <= .100000001
    for value in [0., .1, .8, 1.]:
        trial = state.clone()
        trial[:, wm.spec.layout['cut_progress']] = value
        for bias in [-1000., 0., 1000.]:
            with torch.no_grad():
                wm.term_head.net[-1].bias.fill_(bias)
            logit = wm.terminal_logits(trial, goal, state)
            assert torch.isfinite(logit).all()
            assert logit.sigmoid().item() <= value+1e-12
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.ones_like(logit))
            assert torch.isfinite(loss)
            wm.zero_grad(); loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in wm.parameters() if p.grad is not None)


def test_completion_bound_preserves_bptt_and_head_only_fitting_route():
    wm, state, goal = v3_fixture()
    torch.manual_seed(3)
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight, std=1e-4)
    action = state.new_full((1, 6), .1, requires_grad=True)
    def objective(a):
        _, reward, logits = wm.step(state, a, goal)
        return reward+logits.sigmoid()
    assert torch.autograd.gradcheck(objective, (action,), atol=2e-5, rtol=2e-3)
    wm.terminal_gradient = 'head'
    _, _, logits = wm.fit_predict(state, action, goal)
    grads = torch.autograd.grad(logits.sum(), tuple(wm.dynamics.parameters()), allow_unused=True)
    assert all(g is None for g in grads)


def test_imagination_uses_same_bounded_completion_for_reward_and_survival():
    from cutting_world_model.trainer import OnlineLearner
    wm, state, goal = v3_fixture()
    with torch.no_grad():
        wm.term_head.net[-1].weight.zero_()
        wm.term_head.net[-1].bias.fill_(30.)
    learner = object.__new__(OnlineLearner)
    learner.wm = wm
    learner.policy = lambda history, target: torch.zeros(len(target), 6, dtype=state.dtype)
    learner.pcfg = SimpleNamespace()
    learner.actor_horizon = 3
    ro = learner.imagine(state, state[None].repeat(10, 1, 1), goal)
    for i in range(3):
        rho = wm.spec.field(ro.states[i+1], 'cut_progress')[:, 0]
        assert bool((ro.dhat[i+1] <= rho+1e-12).all())
        torch.testing.assert_close(ro.survival[i+1],ro.survival[i]*(1-ro.dhat[i+1]))
        expected, reward, logits = wm.step(ro.states[i], torch.zeros(1,6,dtype=state.dtype), goal)
        torch.testing.assert_close(ro.rewards[i], reward)
        torch.testing.assert_close(ro.dhat[i+1], logits.sigmoid())


def test_partial_progress_cannot_accumulate_completion_without_new_release():
    wm, state, goal = v3_fixture()
    with torch.no_grad():
        wm.term_head.net[-1].weight.zero_()
        wm.term_head.net[-1].bias.fill_(30.)
    state[:, wm.spec.layout['cut_progress']] = .5
    assert wm.terminal_logits(state,goal,state).sigmoid().item() < 1e-15
    survival=state.new_ones(1)
    for rho in [.6,.7,.75,.8]:
        nxt=state.clone();nxt[:,wm.spec.layout['cut_progress']]=rho
        p=wm.terminal_logits(nxt,goal,state).sigmoid()
        survival=survival*(1-p)
        assert float(1-survival) <= (rho-.5)/.5+1e-12
        state=nxt
    with pytest.raises(ValueError,match='previous progress'):
        wm.terminal_logits(state,goal)
