"""Critic-target variants: source imagined targets and recorded-replay targets."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from cutting_world_model.online_replay import BatchedOnlineReplay, OnlineReplay
from cutting_world_model.policy import SliceItCritic
from cutting_world_model.task_heads import pose_error
from cutting_world_model.trainer import (OnlineLearner, chunks, finite_gradient_norm,
                                         masked_gae_targets)
from cutting_world_model.world_model import CuttingWorldModel
from third_party.supermimic.objective import gae_targets, policy_objective
from third_party.supermimic.trainer_config import PolicyTrainerCfg, WMTrainerCfg
from test_state_spec import fixture
from test_training_stage import TinyPolicy


def episodic_replay(model, frame):
    """Terminated (5), truncated (4), terminated (7) and an unfinished tail (3)."""
    replay = OnlineReplay(model.spec.state_dim, 'cpu')
    goal = torch.tensor([.06, 0., -.05, 0., 0., 0., 1.])
    step = 0
    for length, terminated, truncated in ((5, True, False), (4, False, True),
                                          (7, True, False), (3, False, False)):
        state = model.spec.pack(frame)
        for t in range(length):
            action = torch.full((6,), .01 * (step % 5))
            with torch.no_grad():
                next_state, reward, _ = model.step(state[None], action[None], goal[None])
            last = t == length - 1
            replay.add(state, action, action, goal, float(reward) + .1 * step, next_state[0],
                       terminated and last, truncated and last)
            state = next_state[0]
            step += 1
    return replay.buffer


def make_learner(critic_target='imagined', seed=81, horizon=3, **kwargs):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    metadata, frame = fixture()
    model = CuttingWorldModel(metadata, hidden=(16, 16))
    replay = episodic_replay(model, frame)
    learner = OnlineLearner(
        model, TinyPolicy(), SliceItCritic(model.spec.state_dim), microbatch=2,
        wm_only_updates=0, critic_target=critic_target,
        wm_cfg=WMTrainerCfg(compile=None, window=3, batch_size=4, batches_per_update=2),
        policy_cfg=PolicyTrainerCfg(compile=None, horizon=horizon, batch_size=4,
                                   critic_epochs=2, critic_minibatches=2), **kwargs)
    return learner, replay


def source_improve_policy(self, buffer):
    """Frozen copy of OnlineLearner.improve_policy at 71be9ec (source imagined targets)."""
    cfg = self.pcfg
    starts = buffer.sample_starts(cfg.batch_size, cfg.history)
    self.wm.requires_grad_(False)
    self.critic.requires_grad_(False)
    self.popt.zero_grad(set_to_none=True)
    metrics, critic_data = {}, []
    try:
        for part in chunks(cfg.batch_size, self.microbatch):
            goal = starts.xi[part]
            weight = (part.stop - part.start) / cfg.batch_size
            ro = self.imagine(starts.state[part], starts.hist_states[:, part], goal)
            terminal = self.critic_values(ro.states[-1], goal) if cfg.use_critic else None
            objective = policy_objective(ro, terminal, cfg.gamma)
            loss = -objective * weight / cfg.horizon
            loss.backward()
            with torch.no_grad():
                values = self.critic_values(ro.states.detach(), goal.expand(len(ro.states), -1, -1))
                targets = gae_targets(ro.rewards.detach(), ro.dhat[1:].detach(), values,
                                      cfg.gamma, cfg.gae_lambda)
                states = ro.states[:-1].detach().cpu()
                goals = goal[None].expand(cfg.horizon, -1, -1).cpu()
                critic_data.append((states, goals, targets.cpu()))
                metrics['critic_target_mean'] = (metrics.get('critic_target_mean', 0.)
                                                 + float(targets.mean()) * weight)
        gn = finite_gradient_norm(self.policy.parameters(), cfg.grad_clip_norm)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
        self.popt.step()
        metrics['policy_grad_norm'] = float(gn)
    finally:
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)
    data = [torch.cat([item[i] for item in critic_data], dim=1).flatten(0, 1) for i in range(3)]
    device = next(self.critic.parameters()).device
    n = len(data[2])
    for epoch in range(cfg.critic_epochs):
        permutation = torch.randperm(n, device=device).cpu()
        for indices in permutation.tensor_split(cfg.critic_minibatches):
            self.copt.zero_grad(set_to_none=True)
            total_loss = 0.
            for part in chunks(len(indices), self.microbatch):
                idx = indices[part]
                states, goals, target = [value[idx].to(device) for value in data]
                value = self.critic_values(states, goals)
                loss = F.mse_loss(value, target)
                weight = len(idx) / len(indices)
                (loss * weight).backward()
                total_loss += float(loss.detach()) * weight
            finite_gradient_norm(self.critic.parameters(), cfg.critic_grad_clip)
            self.copt.step()
            metrics['critic_loss'] = total_loss
    self.updates += 1
    return metrics


def assert_same_tensors(left, right):
    for a, b in zip(left, right, strict=True):
        assert torch.equal(a, b)


def test_default_imagined_update_is_bitwise_the_source_update():
    learner, replay = make_learner()
    reference = deepcopy(learner)
    replay_copy = deepcopy(replay)
    assert learner.critic_target == 'imagined'
    rng = torch.get_rng_state()
    for _ in range(2):
        metrics = learner.improve_policy(replay)
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    for _ in range(2):
        expected = source_improve_policy(reference, replay_copy)
    assert torch.equal(after, torch.get_rng_state())
    for name in ('wm', 'policy', 'critic'):
        assert_same_tensors(getattr(learner, name).parameters(), getattr(reference, name).parameters())
    for name in ('popt', 'copt'):
        mine, theirs = getattr(learner, name).state_dict(), getattr(reference, name).state_dict()
        for key in mine['state']:
            for field in mine['state'][key]:
                assert torch.equal(mine['state'][key][field], theirs['state'][key][field])
    for key in ('critic_loss', 'policy_grad_norm', 'critic_target_mean'):
        assert metrics[key] == expected[key]
    assert not any(key.startswith('critic_real') for key in metrics)


def test_masked_recursion_is_source_gae_when_every_step_is_valid():
    torch.manual_seed(3)
    rewards, done = torch.randn(6, 5), (torch.rand(6, 5) < .3).float()
    values = torch.randn(7, 5)
    valid = torch.ones(6, 5, dtype=torch.bool)
    assert torch.equal(masked_gae_targets(rewards, done, values, valid, .99, .95),
                       gae_targets(rewards, done, values, .99, .95))


def synthetic_batch():
    """Window 0 terminates at step 1, window 1 is truncated after 2 steps, window 2 is full."""
    H, n = 4, 3
    states = torch.zeros(H + 1, n, 1)
    valid = torch.tensor([[1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=torch.bool)
    rewards = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
    rewards = torch.where(valid, rewards, float('nan'))  # padding must never be read
    done = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    values = torch.tensor([[.5, .6, .7], [.8, .9, 1.], [1e6, 1.1, 1.2],
                           [1e6, 1e6, 1.3], [1e6, 1e6, 1.4]])
    return rewards, done, values, valid


def test_absorbing_terminal_and_truncation_bootstrap():
    rewards, done, values, valid = synthetic_batch()
    g, lam = .9, .8
    out = masked_gae_targets(rewards, done, values, valid, g, lam)
    # Natural terminal: the terminal reward is kept once and nothing follows it.
    assert out[1, 0] == 4.
    torch.testing.assert_close(out[0, 0], torch.tensor(1. + g * ((1 - lam) * .8 + lam * 4.)))
    # Truncation after two steps bootstraps from V(s_2) = 1.1.
    torch.testing.assert_close(out[1, 1], torch.tensor(5. + g * 1.1))
    torch.testing.assert_close(out[0, 1], torch.tensor(2. + g * ((1 - lam) * .9 + lam * (5. + g * 1.1))))
    # Full windows reproduce the source recursion.
    torch.testing.assert_close(out[:, 2], gae_targets(rewards[:, 2:], done[:, 2:], values[:, 2:], g, lam)[:, 0])
    assert torch.isfinite(out[valid]).all() and out[valid].abs().max() < 100


def independent_targets(learner, batch):
    """Plain-Python lambda returns per window from separately evaluated single-state values."""
    gamma, lam = learner.pcfg.gamma, learner.pcfg.gae_lambda
    spec, critic = learner.wm.spec, deepcopy(learner.critic).cpu()
    out = []
    for w in range(batch.states.shape[1]):
        length = int(batch.valid_steps[:, w].sum())
        assert batch.valid_steps[:length, w].all() and not batch.valid_steps[length:, w].any()
        goal = batch.xi[w][None].cpu()
        values = []
        for t in range(length + 1):
            state = batch.states[t, w][None].cpu()
            with torch.no_grad():
                error = pose_error(goal, spec.field(state, 'previous_ee_pose'))
                values.append(float(critic(spec.canonical(state), goal, error)[0]))
        rewards = [float(r) for r in batch.reward[:length, w]]
        dones = [float(d) for d in batch.done[:length, w]]
        ret, targets = values[length], [0.] * length
        for t in reversed(range(length)):
            ret = rewards[t] + gamma * (1 - dones[t]) * ((1 - lam) * values[t + 1] + lam * ret)
            targets[t] = ret
        out.extend((t, w, value) for t, value in enumerate(targets))
    return out


@pytest.mark.parametrize('horizon', (3, 8))
def test_real_targets_match_independent_cpu_recomputation(horizon):
    learner, replay = make_learner('real', horizon=horizon)
    torch.manual_seed(11)
    batch = replay.sample_windows(64, horizon)
    assert (~batch.valid_steps).any() and batch.done[batch.valid_steps].any()
    states, goals, targets, _ = learner.real_window_targets(batch)
    expected = independent_targets(learner, batch)
    assert len(expected) == len(targets) == int(batch.valid_steps.sum())
    # Boolean indexing is time-major: order the reference the same way.
    expected.sort(key=lambda item: (item[0], item[1]))
    torch.testing.assert_close(targets.double(), torch.tensor([v for *_, v in expected], dtype=torch.float64),
                               atol=1e-4, rtol=1e-5)
    index = [(t, w) for t, w, _ in expected]
    assert torch.equal(states, torch.stack([batch.states[t, w] for t, w in index]))
    assert torch.equal(goals, torch.stack([batch.xi[w] for _, w in index]))


def test_invalid_steps_are_excluded_and_cannot_change_targets():
    learner, replay = make_learner('real', horizon=8)
    torch.manual_seed(12)
    batch = replay.sample_windows(32, 8)
    _, _, targets, _ = learner.real_window_targets(batch)
    invalid = ~batch.valid_steps
    assert invalid.any()
    poisoned = deepcopy(batch)
    poisoned.reward = torch.where(invalid, 1e9, batch.reward)
    poisoned.done = torch.where(invalid, .5, batch.done)
    after_end = torch.zeros(len(invalid) + 1, invalid.shape[1], dtype=torch.bool)
    after_end[1:] = invalid  # padded states strictly after s_L of each window
    poisoned.states = torch.where(after_end[..., None], 1e3, batch.states)
    _, _, poisoned_targets, _ = learner.real_window_targets(poisoned)
    assert torch.equal(targets, poisoned_targets)


def test_real_update_fits_critic_on_source_sample_count():
    learner, replay = make_learner('real')
    policy, wm = deepcopy(learner.policy.state_dict()), deepcopy(learner.wm.state_dict())
    critic = deepcopy(learner.critic.state_dict())
    metrics = learner.improve_policy(replay)
    cfg = learner.pcfg
    assert metrics['critic_real_samples'] == cfg.batch_size * cfg.horizon
    assert metrics['critic_real_valid_steps'] >= cfg.batch_size * cfg.horizon
    assert metrics['critic_target_abs_max'] < 200 and 'imag_critic_target_mean' in metrics
    assert learner.optimizer_steps == {'world_model': 0, 'policy': 1,
                                       'critic': cfg.critic_epochs * cfg.critic_minibatches}
    assert any(not torch.equal(v, policy[k]) for k, v in learner.policy.state_dict().items())
    assert any(not torch.equal(v, critic[k]) for k, v in learner.critic.state_dict().items())
    assert all(torch.equal(v, wm[k]) for k, v in learner.wm.state_dict().items())


def test_real_critic_data_draws_more_windows_until_source_count():
    learner, replay = make_learner('real', horizon=8)
    (states, goals, targets), metrics = learner.real_critic_data(replay)
    need = learner.pcfg.batch_size * learner.pcfg.horizon
    assert len(states) == len(goals) == len(targets) == need
    assert metrics['critic_real_windows'] > learner.pcfg.batch_size  # windows are shorter than H8


def test_checkpoint_rejects_mixed_critic_targets_and_counts_reset_history():
    real, replay = make_learner('real')
    real.improve_policy(replay)
    saved = deepcopy(real.checkpoint())
    assert saved['critic_variant']['target'] == 'real'
    imagined, _ = make_learner('imagined')
    with pytest.raises(ValueError, match='critic target differs'):
        imagined.restore(deepcopy(saved))
    with pytest.raises(ValueError, match='real-target variant'):
        imagined.restore(deepcopy(saved), reset_critic=True)
    source, source_replay = make_learner('imagined')
    source.improve_policy(source_replay)
    old = deepcopy(source.checkpoint())
    del old['critic_variant']  # written before the variant existed
    other, _ = make_learner('real')
    with pytest.raises(ValueError, match='critic target differs'):
        other.restore(deepcopy(old))
    # Explicit reset: keep the fresh critic, then require the pre-fit before policy updates.
    fresh, _ = make_learner('real', seed=5)
    fresh_critic = deepcopy(fresh.critic.state_dict())
    fresh.restore(deepcopy(old), reset_critic=True)
    assert all(torch.equal(v, fresh_critic[k]) for k, v in fresh.critic.state_dict().items())
    assert fresh.optimizer_steps['critic'] == 0 and fresh.critic_reset_update == 1
    with pytest.raises(RuntimeError, match='pre-fit'):
        fresh.improve_policy(replay)
    pending = deepcopy(fresh.checkpoint())
    assert pending['critic_variant']['prefit_pending']
    fresh.prefit_critic(replay, 2)
    fresh.improve_policy(replay)
    resumed, _ = make_learner('real')
    resumed.restore(deepcopy(fresh.checkpoint()))
    assert resumed.critic_variant() == {'target': 'real', 'reset_update': 1, 'prefit_rounds': 2,
                                        'prefit_pending': False}
    assert resumed.optimizer_steps == fresh.optimizer_steps
    again, _ = make_learner('real')
    again.restore(pending)
    assert again.critic_prefit_pending
    tampered = deepcopy(fresh.checkpoint())
    tampered['critic_variant']['prefit_rounds'] = 9
    with pytest.raises(ValueError, match='critic updates exceed'):
        make_learner('real')[0].restore(tampered)


def test_replay_snapshot_round_trip(tmp_path):
    torch.manual_seed(4)
    replay = BatchedOnlineReplay(3, 5, 'cpu', capacity=12)
    for step in range(9):
        state = torch.randn(3, 5)
        terminated = torch.tensor([step % 4 == 3, False, step == 6])
        truncated = torch.tensor([False, step == 7, False])
        replay.add(state, torch.randn(3, 6), torch.randn(3, 6), torch.randn(3, 7), torch.randn(3),
                   torch.randn(3, 5), terminated, truncated)
    replay.save(tmp_path / 'replay.pt', iteration=9)
    saved = torch.load(tmp_path / 'replay.pt', weights_only=False)
    assert saved['iteration'] == 9 and saved['buffer']['rows'] < 12
    restored = BatchedOnlineReplay(3, 5, 'cpu', capacity=12)
    restored.load_state_dict(saved)
    for name in restored.buffer.tensors:
        assert torch.equal(getattr(restored.buffer, name), getattr(replay.buffer, name)), name
    assert torch.equal(restored.frame, replay.frame) and torch.equal(restored.episode, replay.episode)
    torch.manual_seed(1)
    a = replay.buffer.sample_windows(16, 3)
    torch.manual_seed(1)
    b = restored.buffer.sample_windows(16, 3)
    assert torch.equal(a.states, b.states) and torch.equal(a.valid_steps, b.valid_steps)


def test_model_state_samples_are_wm_predictions_of_recorded_states_with_real_targets():
    learner, replay = make_learner('real_wm', horizon=8)
    torch.manual_seed(13)
    batch = replay.sample_windows(16, 8)
    states, goals, targets, _ = learner.real_window_targets(batch)
    full = learner._last_window_targets
    model_states, model_goals, model_targets, _ = learner.model_state_targets(batch, full)
    expected_states, expected_targets = [], []
    for w in range(batch.states.shape[1]):
        state = batch.states[0, w][None]
        for j in range(1, int(batch.valid_steps[:, w].sum())):
            with torch.no_grad():
                state, _, _ = learner.wm.step(state, batch.a_exec[j - 1, w][None], batch.xi[w][None])
            expected_states.append((j, w, state[0]))
            expected_targets.append((j, w, float(full[j, w])))
    order = sorted(range(len(expected_states)), key=lambda i: expected_states[i][:2])
    assert len(model_states) == len(order) > 0
    torch.testing.assert_close(model_states, torch.stack([expected_states[i][2] for i in order]),
                               atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model_targets, torch.tensor([expected_targets[i][2] for i in order]))
    assert torch.equal(model_goals, torch.stack([batch.xi[expected_states[i][1]] for i in order]))
    (data, metrics) = learner.real_critic_data(replay)
    assert len(data[2]) == learner.pcfg.batch_size * learner.pcfg.horizon
    assert 0 < metrics['critic_model_state_fraction'] < 1


def test_actor_horizon_shortens_imagination_and_is_recorded():
    learner, replay = make_learner('real_wm', horizon=3, actor_horizon=2)
    starts = replay.sample_starts(4, learner.pcfg.history)
    ro = learner.imagine(starts.state, starts.hist_states, starts.xi)
    assert ro.rewards.shape[0] == 2 and ro.states.shape[0] == 3
    learner.improve_policy(replay)
    saved = deepcopy(learner.checkpoint())
    assert saved['actor_horizon'] == 2
    other, _ = make_learner('real_wm', horizon=3)
    with pytest.raises(ValueError, match='horizon differs'):
        other.restore(deepcopy(saved))
    same, _ = make_learner('real_wm', horizon=3, actor_horizon=2)
    same.restore(saved)
    with pytest.raises(ValueError, match='actor_horizon'):
        make_learner('real', horizon=3, actor_horizon=4)
    # A critic reset may start a new horizon from a source checkpoint.
    source, source_replay = make_learner('imagined', horizon=3)
    source.improve_policy(source_replay)
    start, _ = make_learner('real_wm', horizon=3, actor_horizon=2)
    start.restore(deepcopy(source.checkpoint()), reset_critic=True)
    assert start.actor_horizon == 2 and start.critic_prefit_pending


def control_learner(**kwargs):
    from test_control_state import compact_fixture
    torch.set_num_threads(2)
    torch.manual_seed(21)
    metadata, frame = compact_fixture()
    model = CuttingWorldModel(metadata, hidden=(16, 16))
    replay = episodic_replay(model, frame)
    learner = OnlineLearner(
        model, TinyHistoryPolicy(), SliceItCritic(model.spec.state_dim), microbatch=2,
        wm_only_updates=0, wm_cfg=WMTrainerCfg(compile=None, window=3, batch_size=4, batches_per_update=2),
        policy_cfg=PolicyTrainerCfg(compile=None, horizon=3, batch_size=4, critic_epochs=1,
                                   critic_minibatches=2), **kwargs)
    return learner, replay


class TinyHistoryPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(10 * 24 + 7, 6)

    def forward(self, history, goal):
        return self.net(torch.cat((history.transpose(0, 1).flatten(1), goal), -1))


def test_observation_history_from_state_matches_replay_history():
    learner, replay = control_learner(critic_target='real_wm')
    torch.manual_seed(2)
    starts = replay.sample_starts(16, learner.pcfg.history)
    expected = learner.wm.spec.field(starts.hist_states, 'observation')
    assert torch.equal(learner.observation_history(starts.state), expected)


def test_behavior_regularization_gradient_is_scaled_td3bc_term():
    learner, replay = control_learner(critic_target='real_wm', actor_bc_alpha=2.5)
    rng = torch.get_rng_state()
    learner.policy.zero_grad(set_to_none=True)
    metrics = learner.behavior_regularization(replay, torch.tensor(7.))
    grads = [p.grad.clone() for p in learner.policy.parameters()]
    torch.set_rng_state(rng)
    batch = replay.sample_windows(learner.pcfg.batch_size, 1)
    learner.policy.zero_grad(set_to_none=True)
    history = learner.wm.spec.field(batch.states[0], 'control_history')
    mse = F.mse_loss(learner.policy(learner.observation_history(batch.states[0]), batch.xi),
                     batch.a_clean[0])
    (mse * 7. / (2.5 * learner.actor_horizon)).backward()
    for actual, param in zip(grads, learner.policy.parameters()):
        torch.testing.assert_close(actual, param.grad)
    assert metrics['actor_bc_scale'] == pytest.approx(7. / (2.5 * 3))
    assert history.shape[-2] == 9
    learner.improve_policy(replay)  # full update with the regularizer
    saved = deepcopy(learner.checkpoint())
    assert saved['actor_bc_alpha'] == 2.5
    with pytest.raises(ValueError, match='behavior regularization'):
        control_learner(critic_target='real_wm')[0].restore(saved)


def _policy_gradient_at_bc(learner):
    """Records whether the policy had gradients when the BC term was added."""
    seen = {}
    original = learner.behavior_regularization

    def wrapped(buffer, mean_abs_objective):
        seen['before'] = [None if p.grad is None else p.grad.clone() for p in learner.policy.parameters()]
        seen['mean_abs'] = float(mean_abs_objective)
        return original(buffer, mean_abs_objective)

    learner.behavior_regularization = wrapped
    return seen


@pytest.mark.parametrize('objective', ['model', 'bc'])
def test_bc_objective_removes_only_the_model_gradient(objective):
    learner, replay = control_learner(critic_target='real', actor_bc_alpha=2.5, actor_objective=objective)
    seen = _policy_gradient_at_bc(learner)
    metrics = learner.improve_policy(replay)
    model_gradient = [g for g in seen['before'] if g is not None and bool(g.abs().sum() > 0)]
    if objective == 'bc':
        assert not model_gradient
    else:
        assert model_gradient
    # The imagined objective is still evaluated for the BC scale and diagnostics.
    assert seen['mean_abs'] > 0 and metrics['actor_bc_scale'] == pytest.approx(
        seen['mean_abs'] / (2.5 * learner.actor_horizon))
    assert 'imag_objective' in metrics and 'critic_loss' in metrics
    saved = deepcopy(learner.checkpoint())
    assert saved['actor_objective'] == objective
    other = 'model' if objective == 'bc' else 'bc'
    with pytest.raises(ValueError, match='actor objective differs'):
        control_learner(critic_target='real', actor_bc_alpha=2.5, actor_objective=other)[0].restore(saved)
    control_learner(critic_target='real', actor_bc_alpha=2.5, actor_objective=objective)[0].restore(saved)


def test_bc_objective_requires_alpha_and_old_checkpoints_are_model():
    with pytest.raises(ValueError, match="requires actor_bc_alpha"):
        control_learner(critic_target='real', actor_objective='bc')
    with pytest.raises(ValueError, match='actor_objective must be'):
        control_learner(critic_target='real', actor_bc_alpha=2.5, actor_objective='mixed')
    learner, replay = control_learner(critic_target='real', actor_bc_alpha=2.5)
    learner.improve_policy(replay)
    saved = deepcopy(learner.checkpoint())
    del saved['actor_objective']
    control_learner(critic_target='real', actor_bc_alpha=2.5)[0].restore(deepcopy(saved))
    with pytest.raises(ValueError, match='actor objective differs'):
        control_learner(critic_target='real', actor_bc_alpha=2.5, actor_objective='bc')[0].restore(saved)
    # A critic reset may switch the objective, as it may switch the horizon.
    start, _ = control_learner(critic_target='real', actor_bc_alpha=2.5, actor_objective='bc')
    start.restore(deepcopy(saved), reset_critic=True)
    assert start.actor_objective == 'bc'
