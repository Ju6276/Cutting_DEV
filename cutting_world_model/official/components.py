"""Official SuperMimic algorithm components (HybridRobotics/supermimic @ 5efbf11), adapted to cutting.

Every class/function names its official counterpart (``src/supermimic/...``). Robot/task-specific
parts (state representation, prediction integration, reward, termination rules) use the cutting
task's ``ControlStateSpec``; all algorithmic parts follow the official code. Deviations are listed
in docs/OFFICIAL_SUPERMIMIC.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .muon import Muon
from .normalization import EmpiricalNormalization

OFFICIAL_REPOSITORY = 'https://github.com/HybridRobotics/supermimic'
OFFICIAL_COMMIT = '5efbf11'


# --- configs: official defaults (learners/wm_learner.py, learners/policy_learner.py,
#     runners/supermimic_runner.py) -----------------------------------------------------------

@dataclass
class WorldModelLearnerCfg:          # learners/wm_learner.py
    lr: float = 2e-4
    num_batches: int = 10
    batch_size: int = 2048
    horizon: int = 8


@dataclass
class ActorCriticPolicyLearnerCfg:   # learners/policy_learner.py (BpttPolicyLearnerCfg + AC)
    lr: float = 2e-4
    num_batches: int = 1
    batch_size: int = 4096
    noise: float = 0.1
    horizon: int = 32
    gamma: float = 0.99
    lam: float = 0.95
    survival_masking: bool = True
    beta2: float = 0.95
    grad_clip_value: float = 0.1


@dataclass
class RunnerCfg:                     # runners/supermimic_runner.py (BaseRunnerCfg, SuperMimicRunnerCfg)
    buffer_size: int = 1000
    actor_obs: str = 'rep_state'
    history_length: int = 1
    termination_softness: float = 0.05
    replay_ratio: int = 16
    latent_dim: int = 32
    actor_hidden_dims: tuple = (2048, 1024, 512, 256)
    critic_hidden_dims: tuple = (256, 128)
    dynamics_hidden_dims: tuple = (2048, 2048, 2048)
    save_interval: int = 1000
    log_interval: int = 10


# --- models.py ---------------------------------------------------------------------------------

def build_mlp(input_dim, hidden_dims, output_dim):
    """models.build_mlp: input -> (Linear -> LayerNorm -> GELU)* -> Linear(output)."""
    layers, current = [], input_dim
    for h in hidden_dims:
        layers += [nn.Linear(current, h), nn.LayerNorm(h), nn.GELU()]
        current = h
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class Dynamics(nn.Module):
    """models.Dynamics: normalized (state, proc(action)) -> MLP -> representation.apply_prediction.

    Cutting adaptation: rep_state is the control-history state of ``ControlStateSpec``;
    ``proc`` clamps to the SliceIt action bounds (that action space has no scale/offset);
    ``apply_prediction`` is ``spec.integrate_physical`` followed by ``spec.derive`` (which also
    rebuilds the derived observation fields and returns the task reward). No sim context
    (fixed task, no domain randomization), so its input width is 0.
    """

    def __init__(self, spec, num_actions=6, hidden_dims=(2048, 2048, 2048)):
        super().__init__()
        self.spec = spec
        self.state_normalizer = EmpiricalNormalization(spec.state_dim)
        self.action_normalizer = EmpiricalNormalization(num_actions)
        self.mlp = build_mlp(spec.state_dim + num_actions, hidden_dims, spec.delta_dim)

    def proc(self, action):
        return self.spec.execute(action)

    def forward(self, state, action, goal):
        executed = self.proc(action)
        prediction = self.mlp(torch.cat((self.state_normalizer(state), self.action_normalizer(executed)), -1))
        fields = self.spec.integrate_physical(state, prediction)
        return self.spec.derive(state, fields, executed, goal)   # (next_state, reward, hard done)

    def normalizers(self):
        return [self.state_normalizer, self.action_normalizer]


class Actor(nn.Module):
    """models.MultiEncoderAE with one task (the anchor), so its latent-BC auxiliary loss is zero:
    goal -> normalizer -> encoder MLP -> tanh latent; [latent, normalized obs history] -> decoder."""

    def __init__(self, obs_dim, goal_dim, action_dim, latent_dim=32, hidden_dims=(2048, 1024, 512, 256),
                 history_length=1):
        super().__init__()
        self.obs_normalizer = EmpiricalNormalization(obs_dim)
        self.goal_normalizer = EmpiricalNormalization(goal_dim)
        self.encoder = build_mlp(goal_dim, hidden_dims, latent_dim)
        self.decoder = build_mlp(latent_dim + history_length * obs_dim, hidden_dims, action_dim)

    def forward(self, history, goal):
        """``history`` (B, H, obs_dim), newest last."""
        z = torch.tanh(self.encoder(self.goal_normalizer(goal)))
        return self.decoder(torch.cat((z, self.obs_normalizer(history).flatten(1)), -1))

    @torch.no_grad()
    def update_normalization(self, obs, goal):
        for n in (self.obs_normalizer, self.goal_normalizer):
            n.train()
        self.obs_normalizer.update(obs)
        self.goal_normalizer.update(goal)
        for n in (self.obs_normalizer, self.goal_normalizer):
            n.eval()


# --- learners/learner_utils.py ----------------------------------------------------------------

def build_optimizers(modules, lr, betas=(0.9, 0.999)):
    """learner_utils.build_optimizers: Muon (torch 2.10 defaults, adjust_lr_fn='match_rms_adamw')
    on the Linear weights between the first and last Linear of each nn.Sequential; Adam on the
    rest. (Official Adam is fused/capturable for CUDA graphs; same update rule.)"""
    hidden = [linear.weight for module in modules for seq in module.modules() if isinstance(seq, nn.Sequential)
              for linear in [m for m in seq if isinstance(m, nn.Linear)][1:-1]]
    hidden_ids = {id(p) for p in hidden}
    rest = [p for m in modules for p in m.parameters() if id(p) not in hidden_ids]
    adam = torch.optim.Adam(rest, lr=lr, betas=betas)
    if not hidden:
        return [adam]
    return [adam, Muon(hidden, lr=lr, adjust_lr_fn='match_rms_adamw')]


def optimize_step(optimizers, loss, params=None, max_norm=None, clip_value=0.0, backward=True):
    """learner_utils._optimize_step: zero_grad -> backward -> clip norm (then value) -> step.
    ``backward=False`` when gradients were accumulated by the caller."""
    if backward:
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
    grad_norm = None
    if params is not None and max_norm is not None:
        params = list(params)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
        if clip_value > 0:
            torch.nn.utils.clip_grad_value_(params, clip_value)
    for optimizer in optimizers:
        optimizer.step()
    return grad_norm


def gae_returns(values, last_value, rewards, dones, gamma, lam):
    """policy_learner.gae_returns (time-major; a soft done p discounts the next state by 1 - p)."""
    returns = torch.empty_like(values)
    advantage = 0.0
    next_value = last_value
    for t in reversed(range(values.shape[0])):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + not_done * gamma * next_value - values[t]
        advantage = delta + not_done * gamma * lam * advantage
        returns[t] = advantage + values[t]
        next_value = values[t]
    return returns


class Ratio:
    """runners/supermimic_runner.Ratio (updates per environment step)."""

    def __init__(self, ratio):
        self._ratio = ratio
        self._prev = None

    def __call__(self, step):
        step = int(step)
        if self._ratio == 0:
            return 0
        if self._ratio < 0:
            return 1
        if self._prev is None:
            self._prev = step
            return 1
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio
        return repeats


# --- replay_buffer.py -------------------------------------------------------------------------

class ReplayBuffer:
    """replay_buffer.ReplayBuffer: per-env ring of pre-step frames (state, action, goal,
    terminated, truncated). The frame after a done belongs to the next episode, so a terminal
    transition's post-step state is never stored; windows never cross a done."""

    def __init__(self, schema, buffer_size, num_envs, device):
        if not {'terminated', 'truncated'} <= schema.keys():
            raise ValueError("schema needs 'terminated' and 'truncated'")
        self.num_transitions = 0
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self._step = 0
        self._storage = {name: torch.zeros((num_envs, buffer_size, *shape), dtype=dtype, device=device)
                         for name, (shape, dtype) in schema.items()}

    def insert(self, fields):
        if set(fields) != set(self._storage):
            raise ValueError('schema mismatch')
        for name, value in fields.items():
            self._storage[name][:, self._step] = value
        self._step = (self._step + 1) % self.buffer_size
        self.num_transitions = min(self.buffer_size, self.num_transitions + 1)

    def sample(self, sequence_length, batch_size):
        L, B, E, T = sequence_length, batch_size, self.num_envs, self.buffer_size
        dev, n = self.device, self.num_transitions
        if L > T:
            raise ValueError(f'sequence_length {L} exceeds buffer_size {T}')
        base = self._step if n == T else 0
        ages = torch.arange(T, device=dev)
        ended = (self._storage['terminated'] | self._storage['truncated'])[:, (base + ages) % T].long()
        c = torch.cat([ended.new_zeros(E, 1), ended.cumsum(1), ended.new_zeros(E, L - 1)], dim=1)
        valid = (c[:, L - 1:L - 1 + T] == c[:, :T]) & (ages + L - 1 < n)
        csum = valid.cumsum(dim=1, dtype=torch.long)
        total = csum[:, -1]
        if not (total > 0).any():
            raise RuntimeError(f'no valid sequences L={L} n={n}')
        env_idx = torch.multinomial((total > 0).float(), B, replacement=True)
        u = (torch.rand(B, device=dev) * total[env_idx].float()).long()
        age_start = torch.searchsorted(csum[env_idx], u[:, None], right=True).squeeze(-1)
        storage_idx = (base + age_start[:, None] + torch.arange(L, device=dev)) % T
        return {name: value[env_idx[:, None], storage_idx] for name, value in self._storage.items()}


# --- learners/wm_learner.py -------------------------------------------------------------------

class WorldModelLearner:
    """wm_learner.WorldModelLearner: autoregressive rollout over a window (the last action's
    prediction is never scored), Bryson-weighted state loss only (no termination head),
    one optimizer step per batch, gradient-norm clip 1.0.

    Cutting adaptation: the loss channel groups are the physical fields of the state, and the
    nominal errors are the representation's per-channel ``err_scale`` (Bryson's rule)."""

    def __init__(self, dynamics, cfg):
        self.dynamics = dynamics
        self.spec = dynamics.spec
        self.cfg = cfg
        self._params = list(dynamics.parameters())
        self.optimizers = build_optimizers([dynamics], cfg.lr)

    @torch.no_grad()
    def update_normalization(self, states, actions):
        for n in self.dynamics.normalizers():
            n.train()
        self.dynamics.state_normalizer.update(states.reshape(-1, states.shape[-1]))
        self.dynamics.action_normalizer.update(self.dynamics.proc(actions.reshape(-1, actions.shape[-1])))
        for n in self.dynamics.normalizers():
            n.eval()

    def losses(self, predicted, target):
        """(loss, per-field normalized MSE): sum_g width_g * mse_g / sum_g width_g, with each
        channel divided by its nominal error."""
        error = self.spec.error_vector(predicted, target)
        group = {name: error[..., s].square().mean() for name, s in self.spec.delta_slices.items()}
        widths = {name: s.stop - s.start for name, s in self.spec.delta_slices.items()}
        loss = sum(widths[k] * v for k, v in group.items()) / sum(widths.values())
        return loss, group

    def update(self, batches):
        self.dynamics.train()
        for n in self.dynamics.normalizers():
            n.eval()
        stats = []
        for batch in batches:
            states, actions, goal = batch['state'], batch['action'], batch['goal'][:, 0]
            state, predicted = states[:, 0], []
            for k in range(actions.shape[1] - 1):
                state, _, _ = self.dynamics(state, actions[:, k], goal)
                predicted.append(state)
            loss, group = self.losses(torch.stack(predicted, 1), states[:, 1:])
            optimize_step(self.optimizers, loss, self._params, max_norm=1.0)
            stats.append((loss.detach(), {k: v.detach() for k, v in group.items()}))
        out = {'Loss/state': float(torch.stack([s[0] for s in stats]).mean())}
        for name in stats[0][1]:
            out['Loss/' + name] = float(torch.stack([s[1][name] for s in stats]).mean())
        return out


# --- envs/world_model_env.py ------------------------------------------------------------------

class WorldModelEnv:
    """world_model_env.WorldModelEnv: imagination with the learned dynamics, starts sampled from
    the replay, task reward computed from the predicted state, soft termination from the task's
    threshold rules: p = sigmoid((normalized_error - 1) / softness).

    Cutting adaptation of the termination rules: the failure rules are the force limit
    (max_i |peak_wrench_i| / limit_i) and the workspace limit (max_i |pose error_i| / limit_i);
    reaching the goal (success counter) ends a cutting episode, so a predicted success is a
    termination with probability 1 (motion tracking has no success termination)."""

    def __init__(self, dynamics, replay, batch_size, termination_softness):
        if termination_softness <= 0.0:
            raise ValueError('termination_softness must be positive')
        self.dynamics = dynamics
        self.spec = dynamics.spec
        self.replay = replay
        self.num_envs = batch_size
        self.softness = termination_softness
        cost = self.spec.heads.env['cost']
        # world_model_env: max_step_reward = analytic per-step reward ceiling (positive terms).
        # Cutting reward: every shaped term and the step cost are <= 0; the only positive term is
        # the goal bonus.
        self.max_step_reward = float(cost['done'])

    def sample_initial_states(self):
        batch = self.replay.sample(1, self.num_envs)
        return {k: v[:, 0] for k, v in batch.items()}

    def reset(self, start):
        self.state = start['state'].detach()
        self.goal = start['goal'].detach()
        return self.observations()

    def observations(self):
        return {'history': self.state[:, None], 'goal': self.goal,
                'critic': torch.cat((self.state, self.goal), -1)}

    def termination_probability(self, previous, state):
        spec, heads = self.spec, self.spec.heads
        force = (spec.field(state, 'peak_wrench').abs() / spec.force_limits).amax(-1)
        obs = spec.field(state, 'observation')
        dims = len(heads.dims)
        error = (obs[..., :dims] * obs.new_tensor(heads.max_distance)).abs()
        workspace = (error / error.new_tensor(heads.env['workspace_limit'])).amax(-1)
        failure = torch.sigmoid((torch.maximum(force, workspace) - 1.0) / self.softness)
        no_collision = torch.zeros_like(force, dtype=torch.bool)
        _, success, _, _ = heads.termination(obs, spec.field(previous, 'success_counter'), no_collision)
        return torch.maximum(failure, success.to(failure.dtype))

    def step(self, action):
        previous = self.state
        self.state, reward, _ = self.dynamics(previous, action, self.goal)
        return self.observations(), reward, self.termination_probability(previous, self.state)


# --- learners/policy_learner.py ---------------------------------------------------------------

class ActorCriticPolicyLearner:
    """policy_learner.ActorCriticPolicyLearner: one imagined rollout of ``horizon`` steps from
    replay starts, return = sum_t gamma^t survival_t r_t + gamma^H V(s_H) survival_H (critic
    clamped at v_max = max_step_reward / (1 - gamma)), one actor step (norm clip 1.0, value clip
    0.1, Adam betas (0.9, beta2) / Muon), then the critic is fitted to the clamped GAE returns of
    that rollout (5 epochs x 4 minibatches).

    Memory adaptation: the batch is split into chunks whose gradients are accumulated before the
    single actor step; the gradient equals the full-batch one (per-sample LayerNorm, mean loss)."""

    def __init__(self, actor, critic, wm_env, cfg, chunk=512):
        self.actor, self.critic, self.wm_env, self.cfg = actor, critic, wm_env, cfg
        self.chunk = chunk
        self.actor_params = list(actor.parameters())
        self.actor_optimizers = build_optimizers([actor], cfg.lr, betas=(0.9, cfg.beta2))
        self.critic_optimizers = build_optimizers([critic], cfg.lr)
        self.v_max = wm_env.max_step_reward / (1 - cfg.gamma)

    def update(self):
        self.actor.train()
        self.wm_env.dynamics.eval()
        self.wm_env.dynamics.requires_grad_(False)
        totals = {}
        try:
            for _ in range(self.cfg.num_batches):
                for k, v in self._train_batch(self.wm_env.sample_initial_states()).items():
                    totals[k] = totals.get(k, 0.) + v
        finally:
            self.wm_env.dynamics.requires_grad_(True)
        n = self.cfg.num_batches
        reward = totals.pop('return') / self.cfg.horizon / n
        return {'Policy/reward': reward, **{f'Policy/{k}': v / n for k, v in totals.items()}}

    def _train_batch(self, start):
        cfg = self.cfg
        size = start['state'].shape[0]
        for optimizer in self.actor_optimizers:
            optimizer.zero_grad(set_to_none=True)
        rollouts, last_obs, total_return = [], [], 0.
        for lo in range(0, size, self.chunk):
            part = {k: v[lo:lo + self.chunk] for k, v in start.items()}
            obs = self.wm_env.reset(part)
            ret, survival, rollout = 0., 1., []
            for t in range(cfg.horizon):
                actions = self.actor(obs['history'], obs['goal'])
                next_obs, reward, term_prob = self.wm_env.step(actions)
                rollout.append({'critic': obs['critic'].detach(), 'reward': reward.detach(),
                                'done': term_prob.detach()})
                ret = ret + cfg.gamma ** t * survival * reward
                if cfg.survival_masking:
                    survival = survival * (1 - term_prob)
                obs = next_obs
            self.critic.requires_grad_(False)
            value = self.critic(obs['critic']).clamp(max=self.v_max).squeeze(-1)
            alive = survival if cfg.survival_masking else 1.0 - term_prob
            ret = ret + cfg.gamma ** cfg.horizon * value * alive
            loss = -ret.sum() / size
            loss.backward()
            total_return += float(ret.detach().sum())
            rollouts.append(rollout)
            last_obs.append(obs['critic'].detach())
        grad_norm = optimize_step(self.actor_optimizers, None, self.actor_params, 1.0, cfg.grad_clip_value,
                                  backward=False)
        steps = {k: torch.cat([torch.stack([r[k] for r in rollout]) for rollout in rollouts], 1)
                 for k in ('critic', 'reward', 'done')}
        return {'return': total_return / size, 'grad_norm': float(grad_norm),
                'value_function': self._fit_critic(steps, torch.cat(last_obs))}

    def _fit_critic(self, steps, last_obs):
        cfg = self.cfg
        with torch.no_grad():
            values = self.critic(steps['critic'])
            last_value = self.critic(last_obs)
        returns = gae_returns(values, last_value, steps['reward'].unsqueeze(-1), steps['done'].unsqueeze(-1),
                              cfg.gamma, cfg.lam).clamp(max=self.v_max)
        obs, returns = steps['critic'].flatten(0, 1), returns.flatten(0, 1)
        perm = torch.randperm(len(returns) // 4 * 4, device=returns.device)
        self.critic.requires_grad_(True)
        losses = []
        for _ in range(5):
            for idx in perm.view(4, -1):
                loss = (returns[idx] - self.critic(obs[idx])).pow(2).mean()
                optimize_step(self.critic_optimizers, loss, self.critic.parameters(), max_norm=1.0)
                losses.append(loss.detach())
        return float(torch.stack(losses).mean())
