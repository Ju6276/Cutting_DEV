import pytest
import torch

from cutting_world_model.online_replay import OnlineReplay
from cutting_world_model.trainer import OnlineLearner
from cutting_world_model.world_model import CuttingWorldModel
from cutting_world_model.policy import SliceItCritic
from third_party.supermimic.dynamics import MLP
from third_party.supermimic.trainer_config import WMTrainerCfg, PolicyTrainerCfg
from test_state_spec import fixture


def replay_fixture(model, frame):
    replay = OnlineReplay(model.spec.state_dim, 'cpu')
    state = model.spec.pack(frame)
    goal = torch.tensor([.06, 0., -.05, 0., 0., 0., 1.])
    for i in range(12):
        action = torch.full((6,), .02 * i)
        with torch.no_grad():
            next_state, reward, _ = model.step(state[None], action[None], goal[None])
        replay.add(state, action, action, goal, float(reward), next_state[0], i == 11, False)
        state = next_state[0]
    return replay.buffer


def test_window_loss_microbatch_gradients_match_full_batch():
    torch.set_num_threads(2)
    torch.manual_seed(5)
    meta, frame = fixture()
    model = CuttingWorldModel(meta, hidden=(16, 16))
    buffer = replay_fixture(model, frame)
    batch = buffer.sample_windows(4, 3)
    loss, _ = model.window_loss(batch.states, batch.a_exec, batch.xi, batch.done)
    loss.backward()
    full = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    for i in range(4):
        loss, _ = model.window_loss(batch.states[:, i:i + 1], batch.a_exec[:, i:i + 1],
                                    batch.xi[i:i + 1], batch.done[:, i:i + 1])
        (loss / 4).backward()
    for expected, param in zip(full, model.parameters()):
        torch.testing.assert_close(param.grad, expected, atol=2e-6, rtol=2e-5)


def test_connected_update_changes_weights_and_restores_gradient_flags():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    meta, frame = fixture()
    wm = CuttingWorldModel(meta, hidden=(16, 16))
    buffer = replay_fixture(wm, frame)

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = MLP(24 * 10 + 7, (16,), 6, last_gain=.01)

        def forward(self, history, goal):
            return self.net(torch.cat((history.permute(1, 0, 2).flatten(1), goal), -1))

    policy = TinyPolicy()
    critic = SliceItCritic(wm.spec.state_dim)
    learner = OnlineLearner(wm, policy, critic, microbatch=2,
        wm_cfg=WMTrainerCfg(compile=None, window=3, batch_size=4, batches_per_update=1),
        policy_cfg=PolicyTrainerCfg(compile=None, horizon=3, batch_size=4, critic_epochs=1, critic_minibatches=2))
    before_wm = [p.detach().clone() for p in wm.parameters()]
    before_pi = [p.detach().clone() for p in policy.parameters()]
    fit = learner.fit_world_model(buffer)
    metrics = learner.improve_policy(buffer)
    assert all(torch.isfinite(torch.tensor(v)) for v in {**fit, **metrics}.values())
    assert any(not torch.equal(a, b) for a, b in zip(before_wm, wm.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(before_pi, policy.parameters()))
    assert all(p.requires_grad for p in wm.parameters())
    assert all(p.requires_grad for p in critic.parameters())
    assert learner.updates == 1


def test_streaming_replay_keeps_terminal_labels_and_context():
    meta, frame = fixture()
    wm = CuttingWorldModel(meta, hidden=(8,))
    buffer = replay_fixture(wm, frame)
    assert buffer.size == 13
    assert buffer.done[0, 12] == 1 and buffer.terminal[0, 12]
    assert buffer.done[0, :12].sum() == 0
    window = buffer.sample_windows(32, 8)
    assert window.xi.shape == (32, 7)
    assert window.states.shape[0] == 9


@pytest.mark.parametrize('critic_minibatches', [1, 4])
def test_policy_and_critic_microbatches_match_full_effective_batch(critic_minibatches):
    from copy import deepcopy
    from types import SimpleNamespace

    torch.set_num_threads(2)
    torch.manual_seed(19)
    meta, frame = fixture()
    wm = CuttingWorldModel(meta, hidden=(16, 16))
    buffer = replay_fixture(wm, frame)
    starts = buffer.sample_starts(8, 10)
    fixed_buffer = SimpleNamespace(sample_starts=lambda *args: starts)

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = MLP(24 + 7, (16,), 6, last_gain=.01)

        def forward(self, history, goal):
            return self.net(torch.cat((history[-1], goal), -1))

    policy = TinyPolicy()
    critic = SliceItCritic(wm.spec.state_dim)
    config = PolicyTrainerCfg(compile=None, horizon=3, batch_size=8,
                              critic_epochs=2, critic_minibatches=critic_minibatches)
    full = OnlineLearner(wm, policy, critic, microbatch=8,
                         wm_cfg=WMTrainerCfg(compile=None), policy_cfg=config)
    split = OnlineLearner(deepcopy(wm), deepcopy(policy), deepcopy(critic), microbatch=3,
                          wm_cfg=WMTrainerCfg(compile=None), policy_cfg=config)
    torch.manual_seed(20)
    full_metrics = full.improve_policy(fixed_buffer)
    torch.manual_seed(20)
    split_metrics = split.improve_policy(fixed_buffer)
    for left, right in [(full.policy, split.policy), (full.critic, split.critic)]:
        for expected, actual in zip(left.parameters(), right.parameters()):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for key in full_metrics:
        torch.testing.assert_close(torch.tensor(split_metrics[key]),
                                   torch.tensor(full_metrics[key]), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(torch.tensor(full_metrics['imag_objective']),
                               torch.tensor(full_metrics['imag_return_discounted'] +
                                            full_metrics['imag_terminal_bootstrap']))
