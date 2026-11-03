"""SuperMimic world-model and BPTT/GAE updates with effective-batch accumulation."""
# Adapted from SuperMimic-Reproduced ba610534; MIT, third_party/supermimic/LICENSE.
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


class OnlineLearner:
    def __init__(self, wm, policy, critic, microbatch=64, wm_cfg=None, policy_cfg=None,
                 wm_only_updates=500):
        if type(wm_only_updates) is not int or wm_only_updates < 0:
            raise ValueError('wm_only_updates must be a nonnegative integer')
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
            self.wopt.zero_grad(set_to_none=True)
            for part in chunks(cfg.batch_size, self.microbatch):
                weight = (part.stop - part.start) / cfg.batch_size
                loss, metrics = self.wm.window_loss(batch.states[:, part], batch.a_exec[:, part],
                                                   batch.xi[part], batch.done[:, part])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite world-model loss')
                (loss * weight).backward()
                metrics['loss'] = loss.detach()
                for key, value in metrics.items():
                    aggregate[key] = aggregate.get(key, 0.) + value.detach() * weight / cfg.batches_per_update
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
        states, rewards, probabilities, survivals, actions = [state], [], [survival * 0], [survival], []
        for _ in range(cfg.horizon):
            action = self.policy(history, goal)
            state, reward, logits = self.wm.predict(state, action, goal)
            probability = logits.sigmoid()
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
                    if cfg.use_critic:
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
                    if cfg.use_critic:
                        measures.update(
                            critic_value_mean=values.mean(),
                            critic_target_mean=targets.mean(),
                            critic_terminal_value_mean=values[-1].mean(),
                            critic_target_abs_mean=targets.abs().mean(),
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
        if cfg.use_critic:
            # Restore source (time, sample) order before shuffling critic minibatches.
            data = [torch.cat([item[i] for item in critic_data], dim=1).flatten(0, 1)
                    for i in range(3)]
            device = next(self.critic.parameters()).device
            n = len(data[2])
            for epoch in range(cfg.critic_epochs):
                self.progress(phase="critic", epoch=epoch + 1, epochs=cfg.critic_epochs)
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

    def checkpoint(self):
        return {'world_model': self.wm.state_dict(), 'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(), 'wm_optimizer': self.wopt.state_dict(),
                'policy_optimizer': self.popt.state_dict(), 'critic_optimizer': self.copt.state_dict(),
                'updates': self.updates,
                'training_stage': {'wm_only_updates': self.wm_only_updates,
                                   'wm_updates': self.wm_updates,
                                   'optimizer_steps': self.optimizer_steps,
                                   'world_model_config': asdict(self.wcfg),
                                   'policy_config': asdict(self.pcfg)}}

    def restore(self, data):
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
        critic_steps = data['updates'] * self.pcfg.critic_epochs * self.pcfg.critic_minibatches
        if self.pcfg.use_critic and optimizer_steps['critic'] < critic_steps:
            raise ValueError('Checkpoint completed critic updates exceed optimizer steps')
        if (self.wm_only_updates > 0 and data['updates'] > 0
                and stage['wm_updates'] <= self.wm_only_updates):
            raise ValueError('Checkpoint policy updates precede the joint training stage')
        if (self.wm_only_updates > 0 and stage['wm_updates'] <= self.wm_only_updates
                and (optimizer_steps['policy'] > 0 or optimizer_steps['critic'] > 0)):
            raise ValueError('Checkpoint policy or critic optimizer steps precede the joint training stage')
        for key, module in [('world_model', self.wm), ('policy', self.policy), ('critic', self.critic),
                            ('wm_optimizer', self.wopt), ('policy_optimizer', self.popt), ('critic_optimizer', self.copt)]:
            module.load_state_dict(data[key])
        self.wm_updates = stage['wm_updates']
        self.updates = data['updates']
