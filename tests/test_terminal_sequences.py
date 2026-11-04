from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from cutting_world_model.online_replay import OnlineReplay
from cutting_world_model.policy import SliceItCritic
from cutting_world_model.trainer import OnlineLearner
from cutting_world_model.world_model import CuttingWorldModel
from third_party.supermimic.trainer_config import WMTrainerCfg
from test_state_spec import fixture


def append_episode(replay, offset, length, terminated=True, truncated=False):
    goal = torch.zeros(7)
    for t in range(length):
        state = torch.tensor([float(offset + t)])
        action = torch.full((6,), float(t + 1))
        replay.add(state, action, action, goal, float(t + 1), state + 1,
                   terminated and t == length - 1, truncated and t == length - 1)


def test_short_terminal_and_truncation_windows_keep_real_transitions():
    replay = OnlineReplay(1, 'cpu', capacity=40)
    append_episode(replay, 0, 1)
    append_episode(replay, 100, 2, terminated=False, truncated=True)
    append_episode(replay, 200, 10)
    append_episode(replay, 300, 3, terminated=False)
    b = replay.buffer
    lengths = b.window_lengths(8)[0]
    assert lengths[:5].tolist() == [1, 0, 2, 1, 0]
    assert lengths[5:16].tolist() == [8, 8, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert not lengths[16:].any()  # unfinished three-step tail has no H8 target
    batch = b.sample_windows(2048, 8)
    delta = batch.states[1:, :, 0] - batch.states[:-1, :, 0]
    torch.testing.assert_close(delta, batch.valid_steps.float())
    assert set(batch.states[0, :, 0].tolist()) == {0., 100., 101., *map(float, range(200, 210))}
    assert not batch.done[~batch.valid_steps].any()
    assert not batch.a_exec[~batch.valid_steps].any()
    for col in range(batch.states.shape[1]):
        start = float(batch.states[0, col, 0])
        length = int(batch.valid_steps[:, col].sum())
        expected = start == 0 or (200 <= start < 210 and start + length == 210)
        assert bool(batch.done[length - 1, col]) == expected
        assert not batch.done[:length - 1, col].any()


def test_wrapped_long_episode_cannot_cross_write_pointer():
    replay = OnlineReplay(1, 'cpu', capacity=9)
    append_episode(replay, 0, 18, terminated=False)
    batch = replay.buffer.sample_windows(10, 8)
    torch.testing.assert_close(batch.states[:, :, 0], torch.arange(9., 18.)[:, None].expand(9, 10))
    assert batch.valid_steps.all()


def test_overwritten_short_terminal_keeps_only_remaining_real_suffix():
    replay = OnlineReplay(1, 'cpu', capacity=10)
    append_episode(replay, 0, 2)
    append_episode(replay, 100, 3)
    append_episode(replay, 200, 4, terminated=False)
    b = replay.buffer
    starts = b.s[0, b.window_mask(8)[0], 0].tolist()
    assert starts == [1., 100., 101., 102.]
    batch = b.sample_windows(100, 8)
    assert batch.valid_steps.sum(0).max() == 3
    torch.testing.assert_close(batch.states[1:, :, 0] - batch.states[:-1, :, 0], batch.valid_steps.float())


def model_batch(lengths):
    torch.set_num_threads(2)
    torch.manual_seed(41)
    metadata, frame = fixture()
    wm = CuttingWorldModel(metadata, hidden=(12, 12)).double()
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight, std=1e-5)
    k, n = max(lengths), len(lengths)
    state = wm.spec.pack(frame).double()[None].repeat(n, 1)
    goal = torch.tensor([[.06, 0., -.05, 0., 0., 0., 1.]], dtype=torch.float64).repeat(n, 1)
    actions = torch.randn(k, n, 6, dtype=torch.float64) * .05
    states = [state]
    with torch.no_grad():
        for t in range(k):
            state, _, _ = wm.step(state, actions[t], goal)
            states.append(state)
    valid = torch.arange(k)[:, None] < torch.tensor(lengths)[None]
    done = torch.zeros(k, n, dtype=torch.float64)
    for i, length in enumerate(lengths):
        done[length - 1, i] = 1
    return wm, SimpleNamespace(states=torch.stack(states), a_exec=actions, xi=goal,
                               done=done, valid_steps=valid)


def test_full_h8_mask_matches_original_loss_and_gradients():
    wm, batch = model_batch([8, 8, 8])
    args = (batch.states, batch.a_exec, batch.xi, batch.done)
    old_loss, old_metrics = wm.window_loss(*args)
    old_grads = torch.autograd.grad(old_loss, tuple(wm.parameters()))
    new_loss, new_metrics = wm.window_loss(*args, batch.valid_steps)
    new_grads = torch.autograd.grad(new_loss, tuple(wm.parameters()))
    torch.testing.assert_close(new_loss, old_loss, atol=1e-12, rtol=1e-12)
    for old, new in zip(old_grads, new_grads):
        torch.testing.assert_close(new, old, atol=1e-12, rtol=1e-12)
    for key in old_metrics:
        torch.testing.assert_close(new_metrics[key].double(), old_metrics[key].double(), atol=1e-7, rtol=1e-7)


def test_masked_mixture_matches_unpadded_episodes_and_ignores_poisoned_padding():
    lengths = [1, 2, 8, 3]
    wm, batch = model_batch(lengths)
    reference = sum(wm.window_loss(batch.states[:length + 1, i:i + 1],
                                  batch.a_exec[:length, i:i + 1], batch.xi[i:i + 1],
                                  batch.done[:length, i:i + 1])[0] * length
                    for i, length in enumerate(lengths)) / sum(lengths)
    expected = torch.autograd.grad(reference, tuple(wm.parameters()))
    for i, length in enumerate(lengths):
        batch.states[length + 1:, i] = float('nan')
        batch.a_exec[length:, i] = float('nan')
        batch.done[length:, i] = float('nan')
    actual, _ = wm.window_loss(batch.states, batch.a_exec, batch.xi, batch.done, batch.valid_steps)
    gradients = torch.autograd.grad(actual, tuple(wm.parameters()))
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    for got, want in zip(gradients, expected):
        torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('microbatch', [1, 3])
def test_mixed_length_effective_batch_update_matches_unsplit(microbatch):
    wm, batch = model_batch([1, 2, 8, 3])
    config = WMTrainerCfg(compile=None, window=8, batch_size=4, batches_per_update=1)
    # Only the WM optimizer is exercised here; policy and critic do not update.
    policy = torch.nn.Linear(1, 1).double()
    critic = SliceItCritic(wm.spec.state_dim).double()
    full = OnlineLearner(wm, policy, critic, microbatch=4, wm_cfg=config)
    split = OnlineLearner(deepcopy(wm), deepcopy(policy), deepcopy(critic),
                          microbatch=microbatch, wm_cfg=config)
    buffer = SimpleNamespace(sample_windows=lambda *args: batch)
    a = full.fit_world_model(buffer)
    b = split.fit_world_model(buffer)
    assert a['sampled_transitions'] == b['sampled_transitions'] == 14
    assert a['sample_terminal_transitions'] == b['sample_terminal_transitions'] == 4
    assert a['short_windows'] == b['short_windows'] == 3
    for key in a:
        assert b[key] == pytest.approx(a[key], rel=1e-10, abs=1e-10)
    for left, right in zip(full.wm.parameters(), split.wm.parameters()):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(full.wopt.state[left]['exp_avg'], split.wopt.state[right]['exp_avg'],
                                   atol=1e-12, rtol=1e-12)
