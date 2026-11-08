from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.mpc import MPPIConfig, MPPIPolicy, ModelRollout
from cutting_world_model.world_model import CuttingWorldModel
from cutting_world_model.policy import SliceItCritic
from test_control_state import compact_fixture


def test_rollout_counts_terminal_reward_once_and_keeps_candidates_independent():
    metadata, frame = compact_fixture()
    spec = ControlStateSpec(metadata)

    class Model:
        def __init__(self):
            self.spec = spec

        def predict(self, state, action, goal):
            result = state.clone()
            spec.field(result, 'peak_wrench')[:, 0] = torch.where(action[:, 0] > 0, 60., 0.)
            reward = torch.where(action[:, 0] > 0, -100., 2.)
            return result, reward, state.new_full((len(state),), torch.logit(torch.tensor(.2)))

    rollout = ModelRollout(Model(), .99)
    state = spec.pack(frame)[None]
    goal = torch.tensor([[.06, 0., 0., 0., 0., 0., 1.]])
    initial = rollout.initial(state, goal)
    batch = initial.repeat(2, 1)
    actions = torch.zeros(2, 6)
    actions[0, 0] = 1.
    first = rollout.dynamics(batch, actions, 0)
    torch.testing.assert_close(first[:, -1], torch.tensor([100., -2.]))
    torch.testing.assert_close(first[:, -3], torch.tensor([0., .8]))
    second = rollout.dynamics(first, torch.ones(2, 6), 1)
    torch.testing.assert_close(second[:, -1], torch.tensor([0., .99 * .8 * 100.]))
    assert second[:, -2].bool().all()
    third = rollout.dynamics(second, torch.zeros(2, 6), 2)
    torch.testing.assert_close(third[:, :-1], second[:, :-1])
    assert not third[:, -1].any()
    single = rollout.dynamics(initial[None], actions[1:], 0)
    torch.testing.assert_close(single[0], first[1])


@pytest.mark.parametrize('terminal_value', ['none', 'critic'])
def test_mppi_uses_six_task_bounds_physical_history_and_frozen_checkpoint(tmp_path, monkeypatch, terminal_value):
    pytest.importorskip('pytorch_mppi')
    metadata, frame = compact_fixture()
    metadata['action_bounds'] = [[0., 0., 0., 0., -1., -1.], [1.] * 6]
    model = CuttingWorldModel(metadata, hidden=(16, 16))
    path = tmp_path / 'checkpoint.pt'
    critic = SliceItCritic(model.spec.state_dim)
    torch.save({'metadata': metadata, 'learner': {'world_model': model.state_dict(),
               'critic': critic.state_dict(), 'updates': 62}}, path)
    monkeypatch.setattr('cutting_world_model.mpc.CuttingWorldModel',
                        lambda metadata: CuttingWorldModel(metadata, hidden=(16, 16)))
    current = {key: torch.as_tensor(np.asarray(value)[None], dtype=torch.float32)
               for key, value in frame.items()}
    current['target_pose'] = current['previous_ee_pose'].clone()
    current['observation'] = current['observation'].double()
    env = SimpleNamespace(worlds=1, metadata=metadata, current=current, device=torch.device('cpu'))
    policy = MPPIPolicy(path, env, MPPIConfig(horizon=3, samples=32, refinements=2,
                                            terminal_value=terminal_value))
    policy.reset(current)
    assert policy.state.dtype == next(model.parameters()).dtype
    low, high = policy.spec.action_low, policy.spec.action_high
    torch.testing.assert_close(policy.rollout.task_action(-torch.ones(6)), low)
    torch.testing.assert_close(policy.rollout.task_action(torch.ones(6)), high)
    torch.manual_seed(11)
    action = policy.act(current)
    assert action.shape == (1, 6) and torch.isfinite(action).all()
    assert ((action >= low) & (action <= high)).all()
    assert policy.unchanged() and all(p.grad is None for p in policy.wm.parameters())
    assert len(policy.plans) == 1 and policy.plans[0]['step'] == 0
    next_frame = {key: value.clone() for key, value in current.items()}
    next_frame['joint_q'] += .02
    next_frame['observation'] += .1
    previous = policy.state.clone()
    policy.observe(next_frame)
    torch.testing.assert_close(policy.spec.field(policy.state, 'joint_q'), next_frame['joint_q'])
    torch.testing.assert_close(policy.state, policy.spec.pack_batch_transition(next_frame, previous))
    policy.reset(current)
    assert not policy.planner.U.any()
    torch.manual_seed(11)
    torch.testing.assert_close(policy.act(current), action)
    assert policy.plans[-1]['episode'] == 1


def test_terminal_value_uses_survival_discount_and_excludes_absorbed_states():
    metadata, frame = compact_fixture()
    model = CuttingWorldModel(metadata, hidden=(16, 16))

    def critic(state, goal, error):
        assert len(state) == 1 and torch.isfinite(state).all()
        return state.new_full((1,), 7.)

    rollout = ModelRollout(model, .99, critic)
    start = rollout.initial(model.spec.pack(frame)[None], torch.zeros(1, 7))
    augmented = start.repeat(2, 1)
    augmented[:, -3] = torch.tensor([.8, 0.])
    augmented[1, :-3] = float('nan')
    cost = rollout.terminal_cost(augmented, 3)
    torch.testing.assert_close(cost, torch.tensor([-.99 ** 3 * .8 * 7, 0.]))
    stored = augmented[None, :, None].repeat(1, 1, 3, 1)
    torch.testing.assert_close(rollout.final_cost(stored, None), cost)


def test_mppi_rejects_nonfinite_model_predictions():
    metadata, frame = compact_fixture()
    model = CuttingWorldModel(metadata, hidden=(16, 16))
    rollout = ModelRollout(model, .99)
    initial = rollout.initial(model.spec.pack(frame)[None], torch.zeros(1, 7))
    model.predict = lambda state, action, goal: (state * float('nan'), torch.ones(len(state)), torch.zeros(len(state)))
    with pytest.raises(FloatingPointError, match='horizon step 0'):
        rollout.dynamics(initial[None], torch.zeros(1, 6), 0)


@pytest.mark.parametrize('kwargs', [{'samples': 0}, {'horizon': -1}, {'refinements': 0},
                                   {'temperature': float('nan')}, {'noise_std': 0.}, {'discount': 1.1},
                                   {'terminal_value': 'bad'}])
def test_mppi_rejects_invalid_solver_configuration(kwargs):
    with pytest.raises(ValueError):
        MPPIConfig(**kwargs)
