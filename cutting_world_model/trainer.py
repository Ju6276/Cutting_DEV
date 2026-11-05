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


CRITIC_TARGETS = ('imagined', 'real')


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
                 wm_only_updates=500, critic_target='imagined', critic_microbatch=None):
        if type(wm_only_updates) is not int or wm_only_updates < 0:
            raise ValueError('wm_only_updates must be a nonnegative integer')
        if critic_target not in CRITIC_TARGETS:
            raise ValueError('critic_target must be one of ' + ', '.join(CRITIC_TARGETS))
        # 'imagined' is the source variant; 'real' fits the critic on recorded replay windows.
        self.critic_target = critic_target
        # Storage-only chunking of real-target critic minibatches; effective minibatches are unchanged.
        self.critic_microbatch = critic_microbatch or microbatch
        # Critic optimizer history after a deliberate critic reinitialization (variant 'real').
        self.critic_reset_update = None
        self.critic_prefit_rounds = 0
        self.critic_prefit_pending = False
        self.wm, self.policy, self.critic = wm, policy, critic
        self.microbatch = microbatch
        self.wcfg, self.pcfg = wm_cfg or WMTrainerCfg(), policy_cfg or PolicyTrainerCfg()
        self.wopt = torch.optim.Adam(wm.parameters(), lr=self.wcfg.lr, weight_decay=self.wcfg.weight_decay)
        self.popt = torch.optim.Adam(policy.parameters(), lr=self.pcfg.lr, betas=self.pcfg.betas)
        self.copt = torch.optim.Adam(critic.parameters(), lr=self.pcfg.critic_lr)
        wm.enable_compile(self.wcfg.compile)
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
        for optimizer in (self.wopt, self.popt, self.copt):
            for group in optimizer.param_groups:
                group['lr'] = lr

    def fit_world_model(self, buffer):
        cfg = self.wcfg
        aggregate = {}
        for batch_index in range(cfg.batches_per_update):
            self.progress(phase="world_model", minibatch=batch_index + 1, minibatches=cfg.batches_per_update)
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
            self.wopt.zero_grad(set_to_none=True)
            for part in chunks(cfg.batch_size, self.microbatch):
                if valid is None:
                    weight = (part.stop - part.start) / cfg.batch_size
                    loss, metrics = self.wm.window_loss(
                        batch.states[:, part], batch.a_exec[:, part], batch.xi[part], batch.done[:, part])
                else:
                    weight = valid[:, part].sum().to(batch.states.dtype) / count
                    loss, metrics = self.wm.window_loss(
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
            gn = finite_gradient_norm(self.wm.parameters(), cfg.grad_clip)
            self.wopt.step()
            aggregate['grad_norm'] = aggregate.get('grad_norm', 0.) + gn.detach() / cfg.batches_per_update
        metrics = {key: float(value) for key, value in aggregate.items()}
        self.wm_updates += 1
        return metrics

    def imagine(self, state, history_states, goal):
        spec, cfg = self.wm.spec, self.pcfg
        history = spec.field(history_states, 'observation')
        survival = torch.ones(state.shape[0], device=state.device)
        absorbed = torch.zeros(state.shape[0], dtype=torch.bool, device=state.device)
        states, rewards, probabilities, survivals, actions = [state], [], [survival * 0], [survival], []
        for _ in range(cfg.horizon):
            action = self.policy(history, goal)
            next_state, reward, logits = self.wm.predict(state, action, goal)
            collision = (spec.field(next_state, 'peak_wrench') > spec.force_limits).any(-1)
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
                               actions=torch.stack(actions))

    def critic_values(self, states, goal):
        error = pose_error(goal, self.wm.spec.field(states, 'previous_ee_pose'))
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
        self.popt.zero_grad(set_to_none=True)
        metrics, critic_data = {}, []
        try:
            for part in chunks(cfg.batch_size, self.microbatch):
                self.progress(phase="policy", samples=part.start, batch_size=cfg.batch_size)
                goal = starts.xi[part]
                weight = (part.stop - part.start) / cfg.batch_size
                ro = self.imagine(starts.state[part], starts.hist_states[:, part], goal)
                terminal = self.critic_values(ro.states[-1], goal) if cfg.use_critic else None
                objective = policy_objective(ro, terminal, cfg.gamma)
                loss = -objective * weight / cfg.horizon
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite policy objective')
                loss.backward()
                with torch.no_grad():
                    values = self.critic_values(ro.states.detach(), goal.expand(len(ro.states), -1, -1))
                    targets = gae_targets(ro.rewards.detach(), ro.dhat[1:].detach(), values,
                                          cfg.gamma, cfg.gae_lambda)
                    discounted_reward = policy_objective(ro, None, cfg.gamma)
                    bootstrap = discounted_reward.new_zeros(())
                    if cfg.use_critic:
                        bootstrap = ((cfg.gamma ** cfg.horizon) * ro.survival[-1] * terminal).mean()
                    spec = self.wm.spec
                    collision = (spec.field(ro.states[1:], 'peak_wrench') > spec.force_limits).any(-1)
                    analytic_done, _, _, _ = spec.heads.termination(
                        spec.field(ro.states[1:], 'observation'),
                        spec.field(ro.states[:-1], 'success_counter'), collision)
                    if cfg.use_critic and self.critic_target == 'imagined':
                        states = ro.states[:-1].detach().cpu()
                        goals = goal[None].expand(cfg.horizon, -1, -1).cpu()
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
                del ro, loss, terminal, objective
            gn = finite_gradient_norm(self.policy.parameters(), cfg.grad_clip_norm)
            torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
            self.popt.step()
            metrics['policy_grad_norm'] = float(gn)
        finally:
            self.wm.requires_grad_(True)
            self.critic.requires_grad_(True)
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
        return batch.states[:-1][valid], goals[:-1][valid], targets[valid], values[:-1][valid]

    @torch.no_grad()
    def real_critic_data(self, buffer):
        """Recorded H-step windows supplying exactly batch_size x horizon critic samples.

        The source fits each critic update on batch_size x horizon imagined states.
        Recorded windows stop at episode ends, so batches of batch_size windows are
        drawn until their valid steps cover that count; a uniform subset is kept.
        """
        cfg = self.pcfg
        need = cfg.batch_size * cfg.horizon
        parts, count, windows, terminals = [], 0, 0, 0.
        while count < need:
            self.progress(phase="critic_data", samples=count, required=need)
            batch = buffer.sample_windows(cfg.batch_size, cfg.horizon)
            parts.append(self.real_window_targets(batch))
            count += len(parts[-1][2])
            windows += cfg.batch_size
            terminals += float(torch.where(batch.valid_steps, batch.done, 0.).sum())
        data = [torch.cat([part[i] for part in parts]) for i in range(4)]
        keep = torch.randperm(count, device=data[2].device)[:need]
        states, goals, targets, values = [value[keep] for value in data]
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
        return (states, goals, targets), metrics

    def reset_critic(self, critic_state, optimizer_state):
        """Variant 'real': install a fresh critic/optimizer, pre-fit before the next policy update."""
        if self.critic_target != 'real':
            raise ValueError('Critic reset is part of the real-target variant only')
        self.critic.load_state_dict(critic_state)
        self.copt.load_state_dict(optimizer_state)
        self.critic_reset_update = self.updates
        self.critic_prefit_rounds = 0
        self.critic_prefit_pending = True

    def prefit_critic(self, buffer, rounds):
        """Real-target critic updates without policy updates; one round equals one critic update."""
        if self.critic_target != 'real' or not self.critic_prefit_pending:
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
                'training_stage': {'wm_only_updates': self.wm_only_updates,
                                   'wm_updates': self.wm_updates,
                                   'optimizer_steps': self.optimizer_steps,
                                   'world_model_config': asdict(self.wcfg),
                                   'policy_config': asdict(self.pcfg)}}

    def restore(self, data, reset_critic=False):
        """Load a checkpoint; ``reset_critic`` (variant 'real') keeps this learner's fresh critic."""
        if data.get('imagination_terminal_rule') != 'task_absorbing_v1':
            raise ValueError('Checkpoint imagination terminal rule differs')
        # Checkpoints written before the variant existed were trained on imagined targets.
        variant = data.get('critic_variant', {'target': 'imagined', 'reset_update': None,
                                              'prefit_rounds': 0, 'prefit_pending': False})
        if not isinstance(variant, dict) or variant.get('target') not in CRITIC_TARGETS:
            raise ValueError('Checkpoint critic variant is invalid')
        if reset_critic and self.critic_target != 'real':
            raise ValueError('Critic reset is part of the real-target variant only')
        if not reset_critic and variant['target'] != self.critic_target:
            raise ValueError('Checkpoint critic target differs: %s, configured %s'
                             % (variant['target'], self.critic_target))
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
        self.wm_updates = stage['wm_updates']
        self.updates = data['updates']
        if reset_critic:
            self.reset_critic(*fresh_critic)
        else:
            self.critic_reset_update = reset_update
            self.critic_prefit_rounds = prefit_rounds
            self.critic_prefit_pending = bool(variant.get('prefit_pending', False))
