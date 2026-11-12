import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_free_cut import model_fixture
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.world_model import CuttingWorldModel
from cutting_world_model.free_cut import FreeCutTask, validate_task
from cutting_world_model.trainer import OnlineLearner
from cutting_world_model.policy import SliceItPolicy, SliceItCritic
from cutting_world_model.wm_diagnostics import completion_progress_report, window_loss_components
from third_party.supermimic.trainer_config import PolicyTrainerCfg, WMTrainerCfg
from test_state_spec import fixture
from test_task_heads import heads


def config():
    return json.loads((Path(__file__).parents[1] / 'configs/free_cut/review_v2.json').read_text())['task']


def v2_fixture():
    metadata, frame = fixture()
    task = config()
    # Synthetic six-joint fixture lives near (0.06, 0, 0), not the UR5e scene.
    task['workspace']['lower'] = [-.2, -.2, -.2]
    task['workspace']['upper'] = [.2, .2, .2]
    metadata.update(free_cut_task=task, goal_dim=8)
    frame.update(cut_progress=np.array([0.]), step_count=np.array([0.]),
                 observation=np.r_[frame['observation'], 1.])
    metadata = ControlStateSpec.metadata_from_frame(metadata, frame)
    model = CuttingWorldModel(metadata, hidden=(12, 12)).double()
    state = model.spec.pack(frame, dtype=torch.float64)[None]
    goal = torch.tensor([[.06, 0., -.05, 0., 0., 0., 1., .03]], dtype=torch.float64)
    return model, state, goal


def test_v2_progress_has_small_initial_increment_and_nonzero_gradient():
    model, state, goal = v2_fixture()
    predicted, _, _ = model.step(state, state.new_zeros(1, 6), goal)
    rho = model.spec.field(predicted, 'cut_progress')
    assert float(rho) == pytest.approx(.001, rel=1e-6)
    rho.sum().backward()
    bias_grad = model.dynamics.mlp.net[-1].bias.grad[model.spec.delta_slices['cut_progress']]
    assert bool((bias_grad > 0).all())
    old, old_state, old_goal = model_fixture()
    old_pred, _, _ = old.step(old_state, old_state.new_zeros(1, 6), old_goal)
    assert old.spec.field(old_pred, 'cut_progress').item() == .5


def test_v2_time_shared_by_model_observation_and_policy():
    model, state, goal = v2_fixture()
    predicted, _, _ = model.step(state, state.new_zeros(1, 6), goal)
    obs = model.spec.field(predicted, 'observation')
    assert obs.shape[-1] == 25 and float(obs[0, -1]) == pytest.approx(.99)
    policy = SliceItPolicy(8, 25).double()
    history = obs[None].repeat(10, 1, 1).detach().requires_grad_()
    policy(history, goal).sum().backward()
    assert history.grad[..., -1].abs().sum() > 0


def test_v2_workspace_is_explicit_robot_base_position_not_old_target_error():
    head = heads()
    head.env['agent_control_dt'] = .05
    task = FreeCutTask(config(), head)
    obs = torch.full((2, 25), 1000.)
    # Both source EE start and a 3 cm higher start, with room to lift further.
    pose = torch.tensor([[.138, .505, .189, 0, 0, 0, 1],
                         [.138, .505, .239, 0, 0, 0, 1.]])
    assert not task.workspace(obs, pose).any()
    pose[0, 2] = .3
    assert task.workspace(obs, pose).tolist() == [True, False]
    with pytest.raises(ValueError, match='pose'):
        task.workspace(obs)


def test_v2_discount_guard_and_legacy_default():
    model, state, goal = v2_fixture()
    policy = SliceItPolicy(8, 25).double()
    critic = SliceItCritic(model.spec.state_dim, 8).double()
    learner = OnlineLearner(model, policy, critic, wm_cfg=WMTrainerCfg(compile=None))
    assert learner.pcfg.gamma == 1
    with pytest.raises(ValueError, match='discount'):
        OnlineLearner(model, policy, critic, policy_cfg=PolicyTrainerCfg(),
                      wm_cfg=WMTrainerCfg(compile=None))
    cfg = config()
    cfg['gamma'] = .99
    with pytest.raises(ValueError, match='gamma'):
        validate_task(cfg)
    assert PolicyTrainerCfg().gamma == .99


def test_physical_completion_calibration_exposes_false_completion_not_pose_or_done():
    model, state, goal = v2_fixture()
    # Force a bad head. The physical trajectory is incomplete despite its terminal flag.
    with torch.no_grad():
        model.term_head.net[-1].weight.zero_()
        model.term_head.net[-1].bias.fill_(10)
    actions = state.new_zeros(2, 1, 6)
    states = state[None].repeat(3, 1, 1)
    valid = torch.ones(2, 1, dtype=torch.bool)
    rows = completion_progress_report(model, states, actions, goal, valid)
    assert rows[0]['false_complete_at_half'] == 1
    assert rows[0]['probability_excess_over_progress'] > .99
    assert rows[0]['brier'] > .99
    loss, _ = model.window_loss(states, actions, goal, state.new_ones(2, 1), valid)
    parts, term, count = window_loss_components(model, states, actions, goal, state.new_ones(2, 1), valid)
    torch.testing.assert_close(loss, (sum(parts.values()) + term) / count)


def test_v2_model_reward_gradient_and_physical_deadline_agree():
    model, state, goal = v2_fixture()
    spec = model.spec
    torch.manual_seed(19)
    torch.nn.init.normal_(model.dynamics.mlp.net[-1].weight, std=1e-5)
    action = state.new_full((1, 6), .1, requires_grad=True)
    next_state, reward, _ = model.step(state, action, goal)
    gradient = torch.autograd.grad(reward.sum(), action)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert not spec.hard_termination(next_state).item()
    state[:, spec.layout['step_count']] = spec.free_task.horizon - 1
    next_state, _, _ = model.step(state, action, goal)
    assert spec.hard_termination(next_state).item()
    assert spec.field(next_state, 'observation')[0, -1] == 0
