"""SuperMimic world-model and BPTT/GAE updates with effective-batch accumulation."""
# Adapted from SuperMimic-Reproduced ba610534; MIT, third_party/supermimic/LICENSE.
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from third_party.supermimic.objective import policy_objective, gae_targets
from third_party.supermimic.trainer_config import WMTrainerCfg, PolicyTrainerCfg
from .task_heads import pose_error


def chunks(n, size):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def finite_gradient_norm(parameters, maximum):
    return torch.nn.utils.clip_grad_norm_(parameters, maximum, error_if_nonfinite=True)


def optimizer_step_count(states):
    return max((int(state['step']) for state in states.values() if 'step' in state), default=0)


def discount_weights(ro, gamma):
    """gamma^t c_t of the source objective for each imagined reward, shape (H, n)."""
    H = ro.rewards.shape[0]
    disc = gamma ** torch.arange(H, device=ro.rewards.device, dtype=ro.rewards.dtype)
    return disc[:, None] * ro.survival[:-1]


CRITIC_TARGETS = ('imagined', 'real', 'real_wm')
# Route B adaptive force-risk penalty (PLAN): an upper-quantile head predicts the next interval's
# sensor-limit utilization R(s_t, a_t) from WM features; the actor objective penalizes
# lambda * survival-weighted max(0, R_up - threshold) over imagined steps. lambda is adapted from
# the physical force-limit episode rate minus a target rate (dual-ascent style, cf. Stooke et
# al. 2020); the penalized and the rate-driving quantities differ, so this is not an exact
# Lagrangian.
# 'quantile': pinball-trained risk head (proposed). 'ensemble': precedent comparator
# (LAMBDA/CAP style): the main WM plus ensemble_size-1 independently initialized WMs, each
# fitted every update; upper estimate = mean + ensemble_std * std of their predicted R.
RISK_ESTIMATORS = ('quantile', 'ensemble')
RISK_DEFAULTS = {'quantile': .95, 'threshold': .9, 'lambda_init': 10., 'lambda_lr': 50.,
                 'target_rate': .02, 'head_lr': 3e-4, 'head_batch': 4096, 'head_steps': 10,
                 'estimator': 'quantile', 'ensemble_size': 5, 'ensemble_std': 2.}
ACTOR_OBJECTIVES = ('model', 'bc')


@torch.no_grad()
def masked_gae_targets(rewards, done_next, values, valid, gamma, lam):
    """Source GAE recursion over recorded windows with a per-window horizon.

    rewards/done_next/valid are (H, n); values is (H+1, n). A window with L valid
    steps uses R_L = V(s_L), as the source uses R_H = V_H; a natural terminal
    (done=1) on its last valid step removes that bootstrap (absorbing). Steps after
    the window end return their own value and are excluded from the critic loss.
    With every step valid this is the source ``gae_targets`` computation.
    """
    H = rewards.shape[0]
    R = values[H].clone()
    out = torch.empty_like(rewards)
    for t in range(H - 1, -1, -1):
        step = rewards[t] + gamma * (1.0 - done_next[t]) * ((1.0 - lam) * values[t + 1] + lam * R)
        R = torch.where(valid[t], step, values[t])
        out[t] = R
    return out


class OnlineLearner:
    def __init__(self, wm, policy, critic, microbatch=64, wm_cfg=None, policy_cfg=None,
                 wm_only_updates=500, critic_target='imagined', critic_microbatch=None,
                 actor_horizon=None, actor_bc_alpha=None, actor_objective='model',
                 wm_terminal_gradient='dynamics', risk_constraint=None):
        if type(wm_only_updates) is not int or wm_only_updates < 0:
            raise ValueError('wm_only_updates must be a nonnegative integer')
        if critic_target not in CRITIC_TARGETS:
            raise ValueError('critic_target must be one of ' + ', '.join(CRITIC_TARGETS))
        # 'imagined' is the source variant; 'real' fits the critic on recorded replay windows;
        # 'real_wm' adds WM open-loop predictions of the recorded states with the same targets.
        self.critic_target = critic_target
        horizon = (policy_cfg or PolicyTrainerCfg()).horizon
        self.actor_horizon = horizon if actor_horizon is None else actor_horizon
        if type(self.actor_horizon) is not int or not 1 <= self.actor_horizon <= horizon:
            raise ValueError('actor_horizon must be an integer in [1, policy horizon]')
        if actor_bc_alpha is not None and not actor_bc_alpha > 0:
            raise ValueError('actor_bc_alpha must be positive')
        # TD3+BC-style behavior regularization of the actor toward recorded clean actions.
        self.actor_bc_alpha = actor_bc_alpha
        # 'model': gradient of the imagined objective (plus the optional BC term); 'bc': the
        # BC term alone with the same scale, an ablation without the model-based gradient.
        if actor_objective not in ACTOR_OBJECTIVES:
            raise ValueError('actor_objective must be one of ' + ', '.join(ACTOR_OBJECTIVES))
        if actor_objective == 'bc' and actor_bc_alpha is None:
            raise ValueError("actor_objective 'bc' requires actor_bc_alpha")
        self.actor_objective = actor_objective
        # Storage-only chunking of real-target critic minibatches; effective minibatches are unchanged.
        self.critic_microbatch = critic_microbatch or microbatch
        # Critic optimizer history after a deliberate critic reinitialization (variant 'real').
        self.critic_reset_update = None
        self.critic_prefit_rounds = 0
        self.critic_prefit_pending = False
        self.wm, self.policy, self.critic = wm, policy, critic
        # WM-fitting variant; see world_model.TERMINAL_GRADIENTS. Policy imagination is unchanged.
        wm.terminal_gradient = wm_terminal_gradient
        self.microbatch = microbatch
        self.wcfg, self.pcfg = wm_cfg or WMTrainerCfg(), policy_cfg or PolicyTrainerCfg()
        self.wopt = torch.optim.Adam(wm.parameters(), lr=self.wcfg.lr, weight_decay=self.wcfg.weight_decay)
        self.popt = torch.optim.Adam(policy.parameters(), lr=self.pcfg.lr, betas=self.pcfg.betas)
        self.copt = torch.optim.Adam(critic.parameters(), lr=self.pcfg.critic_lr)
        wm.enable_compile(self.wcfg.compile)
        self.risk_head = None
        self.risk = None
        self.risk_members, self.risk_member_opts = [], []
        if risk_constraint is not None:
            from .force_risk import QuantileHead
            unknown = set(risk_constraint) - set(RISK_DEFAULTS)
            if unknown:
                raise ValueError('Unknown risk settings: ' + ', '.join(sorted(unknown)))
            self.risk = {**RISK_DEFAULTS, **risk_constraint}
            if self.risk['estimator'] not in RISK_ESTIMATORS:
                raise ValueError('risk estimator must be one of ' + ', '.join(RISK_ESTIMATORS))
            device = next(wm.parameters()).device
            self.risk_head = QuantileHead(wm.dynamics.mlp.net[-1].in_features).to(device)
            self.risk_opt = torch.optim.Adam(self.risk_head.parameters(), lr=self.risk['head_lr'])
            if self.risk['estimator'] == 'ensemble':
                if int(self.risk['ensemble_size']) < 2:
                    raise ValueError('An ensemble needs at least two members')
                for _ in range(int(self.risk['ensemble_size']) - 1):
                    member = type(wm)(wm.spec.metadata).to(device)
                    member.terminal_gradient = wm.terminal_gradient
                    member.enable_compile(self.wcfg.compile)
                    self.risk_members.append(member)
                    self.risk_member_opts.append(torch.optim.Adam(
                        member.parameters(), lr=self.wcfg.lr, weight_decay=self.wcfg.weight_decay))
            self.risk_lambda = float(self.risk['lambda_init'])
            self.risk_head_updates = 0
        self.wm_only_updates = wm_only_updates
        self.wm_updates = 0
        self.updates = 0
        self.progress = lambda **values: None

    @property
    def should_update_policy(self):
        return self.wm_only_updates == 0 or self.wm_updates > self.wm_only_updates

    @property
    def stage(self):
        return 'joint' if self.should_update_policy else 'world_model_only'

    @property
    def optimizer_steps(self):
        return {name: optimizer_step_count(optimizer.state)
                for name, optimizer in (('world_model', self.wopt),
                                        ('policy', self.popt), ('critic', self.copt))}

    def set_lr(self, lr):
        for optimizer in (self.wopt, self.popt, self.copt, *self.risk_member_opts):
            for group in optimizer.param_groups:
                group['lr'] = lr

    def fit_world_model(self, buffer, model=None, optimizer=None):
        """One WM update (``batches_per_update`` minibatches). ``model``/``optimizer`` default
        to the learner's WM; ensemble-risk members reuse the same procedure."""
        member = model is not None
        model = self.wm if model is None else model
        optimizer = self.wopt if optimizer is None else optimizer
        cfg = self.wcfg
        aggregate = {}
        for batch_index in range(cfg.batches_per_update):
            self.progress(phase="risk_ensemble" if member else "world_model", minibatch=batch_index + 1, minibatches=cfg.batches_per_update)
            batch = buffer.sample_windows(cfg.batch_size, cfg.window)
            valid = getattr(batch, 'valid_steps', None)
            if valid is not None:
                count = valid.sum()
                for key, value in {
                    'sampled_transitions': count,
                    'sample_terminal_transitions': torch.where(valid, batch.done, 0.).sum(),
                    'short_windows': (valid.sum(0) < cfg.window).sum(),
                }.items():
                    aggregate[key] = aggregate.get(key, 0.) + value.detach()
            optimizer.zero_grad(set_to_none=True)
            for part in chunks(cfg.batch_size, self.microbatch):
                if valid is None:
                    weight = (part.stop - part.start) / cfg.batch_size
                    loss, metrics = model.window_loss(
                        batch.states[:, part], batch.a_exec[:, part], batch.xi[part], batch.done[:, part])
                else:
                    weight = valid[:, part].sum().to(batch.states.dtype) / count
                    loss, metrics = model.window_loss(
                        batch.states[:, part], batch.a_exec[:, part], batch.xi[part], batch.done[:, part],
                        valid[:, part])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite world-model loss')
                (loss * weight).backward()
                metrics['loss'] = loss.detach()
                for key, value in metrics.items():
                    metric_weight = ((part.stop - part.start) / cfg.batch_size
                                     if key in ('nse_1step', 'nse_kstep') else weight)
                    aggregate[key] = (aggregate.get(key, 0.)
                                      + value.detach() * metric_weight / cfg.batches_per_update)
            gn = finite_gradient_norm(model.parameters(), cfg.grad_clip)
            optimizer.step()
            aggregate['grad_norm'] = aggregate.get('grad_norm', 0.) + gn.detach() / cfg.batches_per_update
        metrics = {key: float(value) for key, value in aggregate.items()}
        if not member:
            self.wm_updates += 1
        return metrics

    def risk_upper(self, state, action, goal):
        if self.risk['estimator'] == 'ensemble':
            from .force_risk import wm_features
            predicted = torch.stack([wm_features(m, state, action, goal)[1]
                                     for m in [self.wm] + self.risk_members])
            return predicted.mean(0) + self.risk['ensemble_std'] * predicted.std(0)
        return self._quantile_upper(state, action, goal)

    def _quantile_upper(self, state, action, goal):
        """Upper-quantile estimate (pinball-trained, not coverage-guaranteed) of the next
        interval's sensor-limit utilization."""
        from .force_risk import utilization, wm_features
        hidden, predicted = wm_features(self.wm, state, action, goal)
        return self.risk_head(hidden, predicted, utilization(self.wm.spec, state, goal))

    def fit_risk_head(self, buffer):
        """``head_steps`` pinball-loss steps of the risk head on recorded transitions (WM frozen);
        the same count as the WM's minibatches per update."""
        if self.risk['estimator'] == 'ensemble':
            metrics = {}
            for index, (member, optimizer) in enumerate(zip(self.risk_members, self.risk_member_opts)):
                row = self.fit_world_model(buffer, member, optimizer)
                metrics['risk_member_loss_%d' % index] = row['loss']
            self.risk_head_updates += 1
            return metrics
        for _ in range(self.risk['head_steps']):
            metrics = self._fit_risk_head_step(buffer)
        return metrics

    def _fit_risk_head_step(self, buffer):
        from .force_risk import pinball, utilization, wm_features
        spec = self.wm.spec
        batch = buffer.sample_windows(self.risk['head_batch'], 1)
        valid = batch.valid_steps[0]
        state, action, goal = batch.states[0][valid], batch.a_exec[0][valid], batch.xi[valid]
        target = utilization(spec, batch.states[1][valid], goal)
        with torch.no_grad():
            hidden, predicted = wm_features(self.wm, state, action, goal)
        upper = self.risk_head(hidden, predicted, utilization(spec, state, goal))
        loss = pinball(upper, target, self.risk['quantile'])
        self.risk_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.risk_opt.step()
        self.risk_head_updates += 1
        with torch.no_grad():
            events = target > 1
            return {'risk_pinball': float(loss), 'risk_coverage': float((upper >= target).float().mean()),
                    'risk_event_recall': float((upper[events] > self.risk['threshold']).float().mean())
                    if bool(events.any()) else float('nan'),
                    'risk_events': int(events.sum())}

    def update_risk_lambda(self, force_limit_episodes, ended_episodes):
        """Dual step: lambda += lr * (physical force-limit episode rate - target), lambda >= 0."""
        if ended_episodes <= 0:
            return self.risk_lambda
        rate = force_limit_episodes / ended_episodes
        self.risk_lambda = max(0., self.risk_lambda + self.risk['lambda_lr'] * (rate - self.risk['target_rate']))
        return self.risk_lambda

    def imagine(self, state, history_states, goal):
        spec, cfg = self.wm.spec, self.pcfg
        history = spec.field(history_states, 'observation')
        survival = torch.ones(state.shape[0], device=state.device)
        absorbed = torch.zeros(state.shape[0], dtype=torch.bool, device=state.device)
        states, rewards, probabilities, survivals, actions = [state], [], [survival * 0], [survival], []
        risks = []
        for _ in range(self.actor_horizon):
            action = self.policy(history, goal)
            if getattr(self, 'risk_head', None) is not None:
                # Risk of the interval about to be executed; zero once the rollout has ended.
                risk = self.risk_upper(state, action, goal)
                risks.append(torch.where(absorbed, torch.zeros_like(risk), risk))
            next_state, reward, logits = self.wm.predict(state, action, goal)
            collision = (spec.field(next_state, 'peak_wrench') > spec.limits(goal)).any(-1)
            task_done, _, _, _ = spec.heads.termination(
                spec.field(next_state, 'observation'),
                spec.field(state, 'success_counter'), collision)
            # The transition into a terminal state keeps its reward exactly once.
            reward = torch.where(absorbed, torch.zeros_like(reward), reward)
            next_state = torch.where(absorbed[:, None], state, next_state)
            absorbed = absorbed | task_done
            probability = torch.where(absorbed, torch.ones_like(logits), logits.sigmoid())
            state = next_state
            survival = survival * (1 - probability)
            history = torch.cat((history[1:], spec.field(state, 'observation')[None]), 0)
            states.append(state)
            rewards.append(reward)
            probabilities.append(probability)
            survivals.append(survival)
            actions.append(action)
        return SimpleNamespace(states=torch.stack(states), rewards=torch.stack(rewards),
                               dhat=torch.stack(probabilities), survival=torch.stack(survivals),
                               actions=torch.stack(actions),
                               risk=torch.stack(risks) if risks else None)

    def critic_values(self, states, goal):
        error = pose_error(goal[..., :7], self.wm.spec.field(states, 'previous_ee_pose'))
        return self.critic(self.wm.spec.canonical(states), goal, error)

    def improve_policy(self, buffer):
        if not self.should_update_policy:
            raise RuntimeError('Policy updates require completion of the world-model-only stage')
        if self.critic_prefit_pending:
            raise RuntimeError('Policy updates require the pending critic pre-fit')
        cfg = self.pcfg
        starts = buffer.sample_starts(cfg.batch_size, cfg.history)
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        if self.risk_head is not None:
            self.risk_head.requires_grad_(False)
            for member in self.risk_members:
                member.requires_grad_(False)
        self.popt.zero_grad(set_to_none=True)
        metrics, critic_data = {}, []
        abs_objective = 0.
        try:
            for part in chunks(cfg.batch_size, self.microbatch):
                self.progress(phase="policy", samples=part.start, batch_size=cfg.batch_size)
                goal = starts.xi[part]
                weight = (part.stop - part.start) / cfg.batch_size
                # The 'bc' ablation evaluates the same imagined objective (for the BC scale and
                # diagnostics) without building or applying its gradient.
                with torch.set_grad_enabled(self.actor_objective == 'model'):
                    ro = self.imagine(starts.state[part], starts.hist_states[:, part], goal)
                    terminal = self.critic_values(ro.states[-1], goal) if cfg.use_critic else None
                    objective = policy_objective(ro, terminal, cfg.gamma)
                    loss = -objective * weight / self.actor_horizon
                    if ro.risk is not None:
                        excess = torch.relu(ro.risk - self.risk['threshold'])
                        risk_cost = (ro.survival[:-1].detach() * excess).sum(0).mean()
                        loss = loss + self.risk_lambda * risk_cost * weight / self.actor_horizon
                        metrics['imag_risk_upper_mean'] = (metrics.get('imag_risk_upper_mean', 0.)
                                                           + float(ro.risk.detach().mean()) * weight)
                        metrics['imag_risk_cost'] = metrics.get('imag_risk_cost', 0.) + float(risk_cost) * weight
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite policy objective')
                if self.actor_objective == 'model':
                    loss.backward()
                with torch.no_grad():
                    values = self.critic_values(ro.states.detach(), goal.expand(len(ro.states), -1, -1))
                    targets = gae_targets(ro.rewards.detach(), ro.dhat[1:].detach(), values,
                                          cfg.gamma, cfg.gae_lambda)
                    discounted_reward = policy_objective(ro, None, cfg.gamma)
                    bootstrap = discounted_reward.new_zeros(())
                    if cfg.use_critic:
                        bootstrap = ((cfg.gamma ** self.actor_horizon) * ro.survival[-1] * terminal).mean()
                    spec = self.wm.spec
                    collision = (spec.field(ro.states[1:], 'peak_wrench') > spec.limits(goal)).any(-1)
                    analytic_done, _, _, _ = spec.heads.termination(
                        spec.field(ro.states[1:], 'observation'),
                        spec.field(ro.states[:-1], 'success_counter'), collision)
                    if cfg.use_critic and self.critic_target == 'imagined':
                        states = ro.states[:-1].detach().cpu()
                        goals = goal[None].expand(self.actor_horizon, -1, -1).cpu()
                        critic_data.append((states, goals, targets.cpu()))
                    measures = {
                        'imag_objective': objective.detach(),
                        'imag_return_discounted': discounted_reward,
                        'imag_terminal_bootstrap': bootstrap,
                        'imag_reward_mean': ro.rewards.detach().mean(),
                        'imag_return_undisc': (ro.survival[:-1] * ro.rewards).detach().sum(0).mean(),
                        'imag_survival_H': ro.survival[-1].detach().mean(),
                        'imag_dhat_mean': ro.dhat[1:].mean(),
                        'imag_analytic_done_mean': analytic_done.float().mean(),
                        'imag_terminal_brier': (ro.dhat[1:] - analytic_done.float()).square().mean(),
                        'action_abs_mean': ro.actions.detach().abs().mean(),
                    }
                    if cfg.use_critic and self.critic_target == 'imagined':
                        measures.update(
                            critic_value_mean=values.mean(),
                            critic_target_mean=targets.mean(),
                            critic_terminal_value_mean=values[-1].mean(),
                            critic_target_abs_mean=targets.abs().mean(),
                        )
                    elif cfg.use_critic:
                        # Diagnostics only: this variant does not fit the critic on imagined targets.
                        measures.update(
                            imag_critic_value_mean=values.mean(),
                            imag_critic_target_mean=targets.mean(),
                            critic_terminal_value_mean=values[-1].mean(),
                            imag_critic_target_abs_mean=targets.abs().mean(),
                            imag_critic_terminal_value_abs_mean=values[-1].abs().mean(),
                        )
                    for key, value in measures.items():
                        metrics[key] = metrics.get(key, 0.) + float(value) * weight
                    if self.actor_bc_alpha is not None:
                        per_sample = (discount_weights(ro, cfg.gamma) * ro.rewards).sum(0)
                        if terminal is not None:
                            per_sample = (per_sample + (cfg.gamma ** self.actor_horizon)
                                          * ro.survival[-1] * terminal)
                        abs_objective = abs_objective + per_sample.detach().abs().sum()
                del ro, loss, terminal, objective
            if self.actor_bc_alpha is not None:
                metrics.update(self.behavior_regularization(buffer, abs_objective / cfg.batch_size))
            gn = finite_gradient_norm(self.policy.parameters(), cfg.grad_clip_norm)
            torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
            self.popt.step()
            metrics['policy_grad_norm'] = float(gn)
        finally:
            self.wm.requires_grad_(True)
            self.critic.requires_grad_(True)
            if self.risk_head is not None:
                self.risk_head.requires_grad_(True)
                for member in self.risk_members:
                    member.requires_grad_(True)
                metrics['risk_lambda'] = self.risk_lambda
        if cfg.use_critic and self.critic_target == 'imagined':
            # Restore source (time, sample) order before shuffling critic minibatches.
            data = [torch.cat([item[i] for item in critic_data], dim=1).flatten(0, 1)
                    for i in range(3)]
            metrics['critic_loss'] = self.fit_critic(data, self.microbatch)
        elif cfg.use_critic:
            data, real_metrics = self.real_critic_data(buffer)
            metrics.update(real_metrics)
            metrics['critic_loss'] = self.fit_critic(data, self.critic_microbatch)
        self.updates += 1
        return metrics

    def observation_history(self, states):
        """Policy input (time, n, obs) from the history frames stored in a control-history state."""
        spec = self.wm.spec
        frames = torch.cat((spec.field(states, 'control_history'),
                            states[..., None, :spec.base_dim]), -2)  # oldest first, current last
        return spec.field(frames, 'observation').transpose(0, 1)[-self.pcfg.history:]

    def behavior_regularization(self, buffer, mean_abs_objective):
        """Adds the gradient of (mean|J| / alpha) * MSE(pi(o_t), a_t) / H on recorded transitions.

        With the source loss -J/H this is (1/lambda) times the TD3+BC loss
        -lambda J + MSE with lambda = alpha / mean|J| (Fujimoto & Gu, 2021).
        """
        cfg = self.pcfg
        batch = buffer.sample_windows(cfg.batch_size, 1)
        history = self.observation_history(batch.states[0])
        scale = float(mean_abs_objective) / (self.actor_bc_alpha * self.actor_horizon)
        total = 0.
        for part in chunks(cfg.batch_size, self.microbatch * 16):
            weight = (part.stop - part.start) / cfg.batch_size
            action = self.policy(history[:, part], batch.xi[part])
            mse = F.mse_loss(action, batch.a_clean[0, part])
            (mse * weight * scale).backward()
            total += float(mse.detach()) * weight
        return {'actor_bc_mse': total, 'actor_bc_scale': scale,
                'actor_mean_abs_objective': float(mean_abs_objective)}

    def fit_critic(self, data, microbatch):
        """Source critic regression: epochs x shuffled minibatches of clipped Adam steps."""
        cfg = self.pcfg
        device = next(self.critic.parameters()).device
        n = len(data[2])
        critic_loss = None
        for epoch in range(cfg.critic_epochs):
            self.progress(phase="critic", epoch=epoch + 1, epochs=cfg.critic_epochs)
            permutation = torch.randperm(n, device=device).cpu()
            for indices in permutation.tensor_split(cfg.critic_minibatches):
                self.copt.zero_grad(set_to_none=True)
                total_loss = 0.
                for part in chunks(len(indices), microbatch):
                    idx = indices[part]
                    states, goals, target = [value[idx].to(device) for value in data]
                    value = self.critic_values(states, goals)
                    loss = F.mse_loss(value, target)
                    weight = len(idx) / len(indices)
                    (loss * weight).backward()
                    total_loss += float(loss.detach()) * weight
                finite_gradient_norm(self.critic.parameters(), cfg.critic_grad_clip)
                self.copt.step()
                critic_loss = total_loss
        return critic_loss

    @torch.no_grad()
    def real_window_targets(self, batch):
        """Critic inputs and GAE targets of the valid steps of one recorded window batch."""
        cfg = self.pcfg
        valid = batch.valid_steps
        steps, n = valid.shape
        goals = batch.xi[None].expand(steps + 1, -1, -1)
        flat_states, flat_goals = batch.states.flatten(0, 1), goals.flatten(0, 1)
        values = torch.cat([self.critic_values(flat_states[part], flat_goals[part])
                            for part in chunks(len(flat_states), self.critic_microbatch)])
        values = values.view(steps + 1, n)
        targets = masked_gae_targets(batch.reward, batch.done, values, valid,
                                     cfg.gamma, cfg.gae_lambda)
        self._last_window_targets = targets
        return batch.states[:-1][valid], goals[:-1][valid], targets[valid], values[:-1][valid]

    @torch.no_grad()
    def model_state_targets(self, batch, targets):
        """Variant 'real_wm': WM open-loop predictions of recorded states with their real targets.

        From each window start s_t the current WM is rolled out with the recorded executed
        actions; the prediction of s_{t+j} (j >= 1, a valid transition start) receives the
        recorded-window target of s_{t+j}. The actor bootstraps the critic on WM states, so
        the critic is fitted where it is queried while its targets stay physical.
        """
        valid = batch.valid_steps
        steps = valid.shape[0]
        mask = valid[1:]
        if not bool(mask.any()):
            # Window of one step, or every window ends after its first transition: no WM state.
            return (batch.states.new_zeros((0, batch.states.shape[-1])),
                    batch.xi.new_zeros((0, batch.xi.shape[-1])), targets.new_zeros(0), targets.new_zeros(0))
        state, predictions = batch.states[0], []
        for t in range(steps - 1):
            state, _, _ = self.wm.predict(state, batch.a_exec[t], batch.xi)
            predictions.append(state)
        predicted = torch.stack(predictions)  # predicted[j] estimates states[j + 1]
        goals = batch.xi[None].expand(steps - 1, -1, -1)
        states = predicted[mask]
        values = torch.cat([self.critic_values(states[part], goals[mask][part])
                            for part in chunks(len(states), self.critic_microbatch)])
        return states, goals[mask], targets[1:][mask], values

    @torch.no_grad()
    def real_critic_data(self, buffer):
        """Recorded H-step windows supplying exactly batch_size x horizon critic samples.

        The source fits each critic update on batch_size x horizon imagined states.
        Recorded windows stop at episode ends, so batches of batch_size windows are
        drawn until their valid steps cover that count; a uniform subset is kept.
        """
        cfg = self.pcfg
        need = cfg.batch_size * cfg.horizon
        parts, count, windows, terminals, model_count = [], 0, 0, 0., 0
        while count < need:
            self.progress(phase="critic_data", samples=count, required=need)
            batch = buffer.sample_windows(cfg.batch_size, cfg.horizon)
            part = self.real_window_targets(batch)
            source = torch.zeros(len(part[2]), dtype=torch.bool, device=part[2].device)
            if self.critic_target == 'real_wm':
                model = self.model_state_targets(batch, self._last_window_targets)
                part = tuple(torch.cat(pair) for pair in zip(part, model))
                source = torch.cat((source, torch.ones(len(model[2]), dtype=torch.bool,
                                                       device=source.device)))
                model_count += len(model[2])
            parts.append((*part, source))
            count += len(part[2])
            windows += cfg.batch_size
            terminals += float(torch.where(batch.valid_steps, batch.done, 0.).sum())
        data = [torch.cat([part[i] for part in parts]) for i in range(5)]
        keep = torch.randperm(count, device=data[2].device)[:need]
        states, goals, targets, values, model_state = [value[keep] for value in data]
        if not torch.isfinite(targets).all():
            raise FloatingPointError('Non-finite recorded critic targets')
        metrics = {
            'critic_target_mean': float(targets.mean()),
            'critic_target_abs_mean': float(targets.abs().mean()),
            'critic_target_abs_max': float(targets.abs().max()),
            'critic_value_mean': float(values.mean()),
            'critic_value_target_mae': float((values - targets).abs().mean()),
            'critic_real_windows': windows,
            'critic_real_valid_steps': count,
            'critic_real_samples': len(targets),
            'critic_real_terminal_labels': terminals,
        }
        if self.critic_target == 'real_wm':
            error = (values - targets).abs()
            metrics.update(critic_model_state_fraction=float(model_state.float().mean()),
                           critic_model_state_candidates=model_count)
            # Each MAE only over a non-empty subset of the kept samples.
            if model_state.any():
                metrics['critic_model_state_value_target_mae'] = float(error[model_state].mean())
            if (~model_state).any():
                metrics['critic_recorded_state_value_target_mae'] = float(error[~model_state].mean())
        return (states, goals, targets), metrics

    def reset_critic(self, critic_state, optimizer_state):
        """Variant 'real': install a fresh critic/optimizer, pre-fit before the next policy update."""
        if self.critic_target == 'imagined':
            raise ValueError('Critic reset is part of the real-target variant only')
        self.critic.load_state_dict(critic_state)
        self.copt.load_state_dict(optimizer_state)
        self.critic_reset_update = self.updates
        self.critic_prefit_rounds = 0
        self.critic_prefit_pending = True

    def prefit_critic(self, buffer, rounds):
        """Real-target critic updates without policy updates; one round equals one critic update."""
        if self.critic_target == 'imagined' or not self.critic_prefit_pending:
            raise RuntimeError('Critic pre-fit requires a pending real-target critic reset')
        if not self.should_update_policy:
            raise RuntimeError('Critic pre-fit requires completion of the world-model-only stage')
        history = []
        for round_index in range(rounds):
            self.progress(phase="critic_prefit", round=round_index + 1, rounds=rounds)
            data, metrics = self.real_critic_data(buffer)
            metrics['critic_loss'] = self.fit_critic(data, self.critic_microbatch)
            self.critic_prefit_rounds += 1
            history.append(metrics)
        self.critic_prefit_pending = False
        return history

    def critic_variant(self):
        return {'target': self.critic_target, 'reset_update': self.critic_reset_update,
                'prefit_rounds': self.critic_prefit_rounds,
                'prefit_pending': self.critic_prefit_pending}

    def checkpoint(self):
        return {'imagination_terminal_rule': 'task_absorbing_v1',
                'world_model': self.wm.state_dict(), 'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(), 'wm_optimizer': self.wopt.state_dict(),
                'policy_optimizer': self.popt.state_dict(), 'critic_optimizer': self.copt.state_dict(),
                'updates': self.updates,
                'critic_variant': self.critic_variant(),
                'actor_horizon': self.actor_horizon, 'actor_bc_alpha': self.actor_bc_alpha,
                'actor_objective': self.actor_objective,
                'wm_terminal_gradient': self.wm.terminal_gradient,
                **({'risk': {'config': self.risk, 'head': self.risk_head.state_dict(),
                             'optimizer': self.risk_opt.state_dict(), 'lambda': self.risk_lambda,
                             'head_updates': self.risk_head_updates,
                             'members': [m.state_dict() for m in self.risk_members],
                             'member_optimizers': [o.state_dict() for o in self.risk_member_opts]}}
                   if self.risk_head is not None else {}),
                'training_stage': {'wm_only_updates': self.wm_only_updates,
                                   'wm_updates': self.wm_updates,
                                   'optimizer_steps': self.optimizer_steps,
                                   'world_model_config': asdict(self.wcfg),
                                   'policy_config': asdict(self.pcfg)}}

    def restore(self, data, reset_critic=False, change_wm_terminal_gradient=False):
        """Load a checkpoint; ``reset_critic`` (variant 'real') keeps this learner's fresh critic.

        ``change_wm_terminal_gradient`` starts this learner's WM-fitting variant from a
        checkpoint written with the other one (e.g. a fixed-data comparison from WM562).
        """
        if data.get('imagination_terminal_rule') != 'task_absorbing_v1':
            raise ValueError('Checkpoint imagination terminal rule differs')
        # Checkpoints written before the variant existed were trained on imagined targets.
        variant = data.get('critic_variant', {'target': 'imagined', 'reset_update': None,
                                              'prefit_rounds': 0, 'prefit_pending': False})
        if not isinstance(variant, dict) or variant.get('target') not in CRITIC_TARGETS:
            raise ValueError('Checkpoint critic variant is invalid')
        if reset_critic and self.critic_target == 'imagined':
            raise ValueError('Critic reset is part of the real-target variant only')
        if not reset_critic and variant['target'] != self.critic_target:
            raise ValueError('Checkpoint critic target differs: %s, configured %s'
                             % (variant['target'], self.critic_target))
        # A critic reset starts a new variant from the saved WM/policy; it may change the horizon.
        if not reset_critic and data.get('actor_horizon', self.pcfg.horizon) != self.actor_horizon:
            raise ValueError('Checkpoint actor imagination horizon differs')
        if not reset_critic and data.get('actor_bc_alpha') != self.actor_bc_alpha:
            raise ValueError('Checkpoint actor behavior regularization differs')
        # Checkpoints written before the option existed used the model-based actor objective.
        if not reset_critic and data.get('actor_objective', 'model') != self.actor_objective:
            raise ValueError('Checkpoint actor objective differs')
        # Checkpoints written before the option existed used the source route.
        saved_route = data.get('wm_terminal_gradient', 'dynamics')
        if saved_route != self.wm.terminal_gradient and not change_wm_terminal_gradient:
            raise ValueError('Checkpoint WM terminal-gradient route differs: %s, configured %s'
                             % (saved_route, self.wm.terminal_gradient))
        reset_update = variant.get('reset_update')
        prefit_rounds = variant.get('prefit_rounds', 0)
        if reset_update is not None and (type(reset_update) is not int or reset_update < 0
                                         or reset_update > data.get('updates', -1)):
            raise ValueError('Checkpoint critic reset update is invalid')
        if type(prefit_rounds) is not int or prefit_rounds < 0:
            raise ValueError('Checkpoint critic pre-fit rounds are invalid')
        stage = data.get('training_stage')
        if not isinstance(stage, dict):
            raise ValueError('Checkpoint lacks training-stage configuration and counts')
        for name in ('wm_only_updates', 'wm_updates'):
            if type(stage.get(name)) is not int or stage[name] < 0:
                raise ValueError('Checkpoint ' + name + ' must be a nonnegative integer')
        if stage['wm_only_updates'] != self.wm_only_updates:
            raise ValueError('Checkpoint world-model-only configuration differs')
        if (stage.get('world_model_config') != asdict(self.wcfg)
                or stage.get('policy_config') != asdict(self.pcfg)):
            raise ValueError('Checkpoint trainer configuration differs')
        optimizer_steps = stage.get('optimizer_steps')
        if not isinstance(optimizer_steps, dict):
            raise ValueError('Checkpoint lacks optimizer step counts')
        for name, key in (('world_model', 'wm_optimizer'), ('policy', 'policy_optimizer'),
                          ('critic', 'critic_optimizer')):
            count = optimizer_steps.get(name)
            if (type(count) is not int or count < 0
                    or count != optimizer_step_count(data[key]['state'])):
                raise ValueError('Checkpoint optimizer step count differs: ' + name)
        if type(data.get('updates')) is not int or data['updates'] < 0:
            raise ValueError('Checkpoint policy updates must be a nonnegative integer')
        if optimizer_steps['world_model'] < stage['wm_updates'] * self.wcfg.batches_per_update:
            raise ValueError('Checkpoint completed world-model updates exceed optimizer steps')
        if optimizer_steps['policy'] < data['updates']:
            raise ValueError('Checkpoint completed policy updates exceed optimizer steps')
        critic_rounds = data['updates'] - (reset_update or 0) + prefit_rounds
        critic_steps = critic_rounds * self.pcfg.critic_epochs * self.pcfg.critic_minibatches
        if self.pcfg.use_critic and optimizer_steps['critic'] < critic_steps:
            raise ValueError('Checkpoint completed critic updates exceed optimizer steps')
        if (self.wm_only_updates > 0 and data['updates'] > 0
                and stage['wm_updates'] <= self.wm_only_updates):
            raise ValueError('Checkpoint policy updates precede the joint training stage')
        if (self.wm_only_updates > 0 and stage['wm_updates'] <= self.wm_only_updates
                and (optimizer_steps['policy'] > 0 or optimizer_steps['critic'] > 0)):
            raise ValueError('Checkpoint policy or critic optimizer steps precede the joint training stage')
        fresh_critic = (deepcopy(self.critic.state_dict()), deepcopy(self.copt.state_dict()))
        for key, module in [('world_model', self.wm), ('policy', self.policy), ('critic', self.critic),
                            ('wm_optimizer', self.wopt), ('policy_optimizer', self.popt), ('critic_optimizer', self.copt)]:
            module.load_state_dict(data[key])
        saved_risk = data.get('risk')
        if (saved_risk is None) != (self.risk_head is None):
            raise ValueError('Checkpoint and learner differ in the force-risk constraint')
        if saved_risk is not None:
            if saved_risk['config'] != self.risk:
                raise ValueError('Checkpoint force-risk configuration differs')
            self.risk_head.load_state_dict(saved_risk['head'])
            self.risk_opt.load_state_dict(saved_risk['optimizer'])
            self.risk_lambda = float(saved_risk['lambda'])
            self.risk_head_updates = int(saved_risk['head_updates'])
            for member, state in zip(self.risk_members, saved_risk.get('members', [])):
                member.load_state_dict(state)
            for optimizer, state in zip(self.risk_member_opts, saved_risk.get('member_optimizers', [])):
                optimizer.load_state_dict(state)
        self.wm_updates = stage['wm_updates']
        self.updates = data['updates']
        if reset_critic:
            self.reset_critic(*fresh_critic)
        else:
            self.critic_reset_update = reset_update
            self.critic_prefit_rounds = prefit_rounds
            self.critic_prefit_pending = bool(variant.get('prefit_pending', False))
