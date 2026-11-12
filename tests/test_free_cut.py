import os
import sys

import numpy as np
import pytest
import torch

from cutting_world_model.free_cut import FreeCutTask
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.world_model import CuttingWorldModel
from test_state_spec import fixture
from test_task_heads import heads


def task_config(horizon=200):
    # Interface-check numbers, not an approved training configuration.
    return dict(version='free_cut_v1', horizon_steps=horizon,
                weights=dict(progress=10., speed=1., load=1., jerk=.1, success=100., failure=100.))


def model_fixture():
    metadata, frame = fixture()
    metadata.update(free_cut_task=task_config(), goal_dim=8)
    frame.update(cut_progress=np.array([0.]), step_count=np.array([0.]))
    metadata = ControlStateSpec.metadata_from_frame(metadata, frame)
    model = CuttingWorldModel(metadata, hidden=(12, 12)).double()
    state = model.spec.pack(frame, dtype=torch.float64)[None]
    goal = torch.tensor([[.06, 0., -.05, 0., 0., 0., 1., .03]], dtype=torch.float64)
    return model, state, goal


def test_free_cut_events_do_not_use_pose_success_and_failure_paid_once():
    head = heads()
    head.env['agent_control_dt'] = .05
    task = FreeCutTask(task_config(2), head)
    obs = torch.zeros(5, 24)
    peak = torch.zeros(5, 6)
    peak[2, 0] = 51.
    peak[3, 0] = 51.
    obs[3, 0] = 1000.  # force and workspace failures coincide
    progress = torch.tensor([0., 1., 1., .5, .5])
    events = task.events(obs, peak, progress, torch.tensor([1, 2, 2, 1, 2]))
    assert events['success'].tolist() == [False, True, False, False, False]
    assert events['timeout'].tolist() == [False, False, False, False, True]
    assert events['terminated'].tolist() == [False, True, True, True, True]
    step = task.step_reward(torch.zeros(5), progress, obs, peak, torch.full((5,), .03))
    total = task.reward(torch.zeros(5), progress, obs, peak, torch.full((5,), .03), events)
    torch.testing.assert_close(total - step, torch.tensor([0., 100., -100., -100., 0.]))


def test_free_cut_model_progress_time_and_completion_targets():
    model, state, goal = model_fixture()
    spec = model.spec
    assert spec.field(state, 'observation').shape == (1, 24)
    assert spec.field(state, 'step_count').item() == 0.
    delta = torch.zeros(1, spec.delta_dim, dtype=torch.float64)
    fields = spec.integrate_physical(state, delta)
    result, reward, _ = spec.derive(state, fields, torch.zeros(1, 6, dtype=torch.float64), goal)
    assert spec.field(result, 'step_count').item() == 1.
    assert spec.field(result, 'cut_progress').item() == .5
    # Nonterminal progress remains bounded, success is learned rather than rho==1.
    state = result
    for _ in range(5):
        result, _, _ = model.step(state, torch.zeros(1, 6, dtype=torch.float64), goal)
        assert bool((spec.field(result, 'cut_progress') >= spec.field(state, 'cut_progress')).all())
        assert spec.field(result, 'cut_progress').item() <= 1.
        state = result
    states = torch.stack((state, state.clone(), state.clone()))
    states[1, :, spec.layout['cut_progress']] = 1.
    states[2, :, spec.layout['cut_progress']] = .5
    # A force failure is not a positive separation target.
    torch.testing.assert_close(model.completion_labels(states, torch.ones(2, 1)),
                               torch.tensor([[1.], [0.]]))
    state[:, spec.layout['step_count']] = spec.free_task.horizon
    assert spec.hard_termination(state).item()
    state[:, spec.layout['peak_wrench']] = 1000.
    assert spec.completion_reward(state, torch.tensor([30.])).item() == 0.


def test_free_cut_bptt_gradient_and_head_only_fit_route():
    torch.manual_seed(5)
    model, state, goal = model_fixture()
    torch.nn.init.normal_(model.dynamics.mlp.net[-1].weight, std=1e-4)
    action = torch.full((1, 6), .1, dtype=torch.float64, requires_grad=True)
    def objective(a):
        final, reward, logits = model.step(state, a, goal)
        return reward + logits.sigmoid()
    assert torch.autograd.gradcheck(objective, (action,), atol=2e-5, rtol=2e-3)
    gradient = torch.autograd.grad(objective(action).sum(), action)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    model.terminal_gradient = 'head'
    _, _, logits = model.fit_predict(state, action, goal)
    grads = torch.autograd.grad(logits.sum(), tuple(model.dynamics.parameters()), allow_unused=True)
    assert all(g is None for g in grads)
    # Actual optimizer update verifies bounded-progress supervision and completion BCE connect.
    target = state.clone()
    target[:, model.spec.layout['cut_progress']] = 1.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = model.dynamics.mlp.net[-1].weight.detach().clone()
    loss, _ = model.window_loss(torch.stack((state, target)), action.detach()[None], goal,
                                torch.ones(1, 1, dtype=torch.float64))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss) and not torch.equal(before, model.dynamics.mlp.net[-1].weight)


def test_free_cut_imagination_deadline_has_zero_future_reward_and_bootstrap():
    from cutting_world_model.trainer import OnlineLearner
    from types import SimpleNamespace
    model, state, goal = model_fixture()
    state[:, model.spec.layout['step_count']] = model.spec.free_task.horizon - 1
    learner = object.__new__(OnlineLearner)
    learner.wm = model
    learner.policy = lambda history, goal: torch.zeros(len(goal), 6, dtype=state.dtype)
    learner.pcfg = SimpleNamespace()
    learner.actor_horizon = 3
    ro = learner.imagine(state, state[None].repeat(10, 1, 1), goal)
    assert ro.survival[1:].eq(0).all()
    assert ro.rewards[1:].eq(0).all()
    torch.testing.assert_close(ro.states[1], ro.states[-1])


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('CUTTING_BACKEND_ROOT'),
                    reason='CUDA backend assets')
def test_free_cut_randomized_reset_preserves_live_world_and_timeout():
    root = os.environ['CUTTING_BACKEND_ROOT']
    sys.path.insert(0, root)
    from cutting_world_model.multi_env import LocalGroup
    scene = 'runs/sweep/free_cucumber_003/scene.pt'
    if not os.path.exists(os.path.join(root, scene)):
        pytest.skip('prepared free cucumber')
    group = LocalGroup(root, scene, 2, torch.zeros(2), None, [50, 50, 50, 5, 5, 5],
                       force_model='net', friction_velocity=.01, settle_seconds=.002, task=task_config(2))
    mask = torch.ones(2, dtype=torch.bool, device='cuda')
    group.reset(mask, {'youngs_modulus': torch.tensor([1., 2.])}, torch.tensor([.03, .03]))
    group.step(torch.zeros(2, 6, device='cuda'))
    env = group.env
    before = {k: v.clone() for k, v in env.current.items()}
    saved = {k: v.clone() for k, v in env.physics.material.initial_state.items()}
    group.reset(torch.tensor([True, False], device='cuda'),
                {'youngs_modulus': torch.tensor([1.5])}, torch.tensor([.03]))
    for name in before:
        torch.testing.assert_close(env.current[name][1], before[name][1], atol=0, rtol=0)
    for name, value in saved.items():
        torch.testing.assert_close(env.physics.material.initial_state[name].reshape(2, -1)[1],
                                   value.reshape(2, -1)[1], atol=0, rtol=0)
    assert env.step_count.tolist() == [0, 1]
    _, _, done, trunc, info = group.step(torch.zeros(2, 6, device='cuda'))
    assert done.tolist() == [False, True] and not trunc.any() and info['timeout'][1]



def test_free_cut_event_logs_separate_timeout_from_truncation():
    from cutting_world_model.episode_log import episode_row
    info = dict(success=torch.tensor([False]), collision=torch.tensor([False]),
                outside=torch.tensor([False]), timeout=torch.tensor([True]),
                separated=torch.tensor([False]), cut_progress=torch.tensor([.5]),
                released_springs=torch.tensor([2]))
    row = episode_row(0, 0, 1, torch.tensor([-1.]), torch.tensor([2]), info,
                      torch.tensor([False]), torch.tensor([2.]), torch.tensor([.1]))
    assert row['ending'] == 'time_limit'
    assert row['timeout'] and not row['truncated'] and not row['separated']


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('CUTTING_BACKEND_ROOT'),
                    reason='CUDA backend assets')
@pytest.mark.parametrize('parallel', [False, True])
def test_free_cut_grouped_goal_replay_and_partial_reset(parallel):
    root = os.environ['CUTTING_BACKEND_ROOT']
    sys.path.insert(0, root)
    from cutting_world_model.multi_env import MultiConditionEnvironment, knife_clearance
    scene = 'runs/sweep/free_cucumber_003/scene.pt'
    if not os.path.exists(os.path.join(root, scene)):
        pytest.skip('prepared free cucumber')
    clearance = knife_clearance(root, scene)
    group = dict(scene=scene, worlds=1, clearance=[clearance, clearance],
                 target_speed=[.03, .03], settle_seconds=.002, force_model='net',
                 friction_velocity=.01, material={'youngs_modulus': [1., 2.]})
    cfg = dict(force_limits=[50, 50, 50, 5, 5, 5], parallel=parallel, task=task_config(3),
               groups=[dict(group, name='a'), dict(group, name='b')])
    env = MultiConditionEnvironment.from_config(root, cfg)
    try:
        one = {k: v[0].cpu().numpy() for k, v in env.current.items()}
        wm = CuttingWorldModel(ControlStateSpec.metadata_from_frame(env.metadata, one),
                               hidden=(12, 12)).cuda()
        state = wm.spec.pack_batch(env.current).float()
        goal = env.goal(env.current)
        assert goal.shape == (2, 8)
        frame, _, _, _, _ = env.step(torch.zeros(2, 6, device='cuda'))
        state = wm.spec.pack_batch_transition(frame, state)
        next_state, reward, logits = wm.step(state, torch.zeros(2, 6, device='cuda'), goal)
        assert torch.isfinite(next_state).all() and torch.isfinite(reward).all()
        assert logits.shape == (2,)
        before = {k: v.clone() for k, v in frame.items()}
        frame = env.reset(torch.tensor([True, False], device='cuda'))
        for name in before:
            torch.testing.assert_close(frame[name][1], before[name][1], atol=0, rtol=0)
        assert frame['step_count'][:, 0].tolist() == [0, 1]
    finally:
        env.close()



def test_free_cut_zero_initialized_force_outputs_can_learn_contact():
    # The source dynamics last layer is zero initialized. abs() on its raw outputs
    # would have zero derivative at zero and silently prevent force fitting.
    model, state, goal = model_fixture()
    predicted, _, _ = model.step(state, torch.zeros(1, 6, dtype=state.dtype), goal)
    target = state.clone()
    target[:, model.spec.layout['peak_wrench']] = 1.
    loss = model.spec.error_vector(predicted, target).square().sum()
    loss.backward()
    gradient = model.dynamics.mlp.net[-1].bias.grad[model.spec.delta_slices['peak_wrench']]
    assert torch.isfinite(gradient).all() and bool((gradient != 0).all())
