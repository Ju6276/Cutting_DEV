"""Source residual MLP, autoregressive loss and terminal head with cutting task interfaces."""
# Adapted from SuperMimic-Reproduced ba610534; MIT, third_party/supermimic/LICENSE.
import torch
from torch import nn
from torch.nn import functional as F

from third_party.supermimic.dynamics import DynamicsMLP, MLP
from .state_spec import CuttingStateSpec
from .control_state import ControlStateSpec

# Route of the terminal BCE gradient during WM fitting. 'dynamics' (source): the BCE also
# reaches the dynamics through the predicted state its terminal features read. 'head': the
# terminal head reads detached predicted features in the fitting loss, so the BCE updates
# only the head. Policy imagination (step/predict) is identical for both.
TERMINAL_GRADIENTS = ('dynamics', 'head')


class CuttingWorldModel(nn.Module):
    def __init__(self, metadata, hidden=(2048, 2048, 2048)):
        super().__init__()
        layout = metadata.get('prediction_layout')
        if layout not in (None, 'control_history_v2'):
            raise ValueError('Unsupported prediction layout: ' + str(layout))
        spec_class = ControlStateSpec if layout == 'control_history_v2' else CuttingStateSpec
        if metadata.get('free_cut_task'):
            if layout != 'control_history_v2':
                raise ValueError('free_cut requires the compact control-history layout')
            from .free_cut_state import FreeCutStateSpec
            spec_class = FreeCutStateSpec
        self.spec = spec_class(metadata)
        self.dynamics = DynamicsMLP(self.spec.dynamics_dim, 6, 7, hidden, self.spec.delta_dim)
        if metadata.get('free_cut_task', {}).get('version') == 'free_cut_v2':
            import math
            fraction = metadata['free_cut_task']['initial_progress_fraction']
            with torch.no_grad():
                self.dynamics.mlp.net[-1].bias[self.spec.delta_slices['cut_progress']] = math.log(fraction / (1 - fraction))
        self.term_head = MLP(2 * getattr(self.spec, 'terminal_feature_dim', 13), (64, 64), 1)
        self.terminal_gradient = 'dynamics'

    @property
    def terminal_gradient(self):
        return self._terminal_gradient

    @terminal_gradient.setter
    def terminal_gradient(self, value):
        if value not in TERMINAL_GRADIENTS:
            raise ValueError('terminal_gradient must be one of ' + ', '.join(TERMINAL_GRADIENTS))
        self._terminal_gradient = value

    def transition(self, state, action, goal):
        executed = self.spec.execute(action)
        # Physics prediction uses the target pose only; task conditions enter reward/termination.
        delta = self.dynamics(self.spec.dynamics_features(state), executed, goal[..., :7])
        fields = self.spec.integrate_physical(state, delta)
        next_state, reward, done = self.spec.derive(state, fields, executed, goal)
        return next_state, reward

    def terminal_logits(self, next_state, goal=None):
        feats = self.spec.terminal_features(next_state, goal)
        return self.term_head(torch.cat((feats, feats.abs()), -1)).squeeze(-1)

    def step(self, state, action, goal):
        next_state, reward = self.transition(state, action, goal)
        logits = self.terminal_logits(next_state, goal)
        if hasattr(self.spec, 'free_task'):
            reward = reward + self.spec.completion_reward(next_state, logits)
        return next_state, reward, logits

    def _fit_step_head(self, state, action, goal):
        next_state, reward = self.transition(state, action, goal)
        logits = self.terminal_logits(next_state.detach(), goal)
        if hasattr(self.spec, 'free_task'):
            reward = reward + self.spec.completion_reward(next_state, logits)
        return next_state, reward, logits

    def enable_compile(self, mode="default"):
        compile = (lambda f: torch.compile(f, mode=mode, dynamic=False)) if mode else (lambda f: f)
        self._step_fn = compile(self.step)
        self._fit_head_fn = compile(self._fit_step_head)

    def predict(self, state, action, goal):
        return getattr(self, '_step_fn', self.step)(state, action, goal)

    def fit_predict(self, state, action, goal):
        """Step used by the WM fitting losses; differs from predict only in the BCE gradient route."""
        if self.terminal_gradient == 'dynamics':
            return self.predict(state, action, goal)
        return getattr(self, '_fit_head_fn', self._fit_step_head)(state, action, goal)

    def completion_labels(self, states, done):
        # Physical separation, independent of force failures and finite task deadlines.
        if hasattr(self.spec, 'free_task'):
            return (self.spec.field(states[1:], 'cut_progress')[..., 0] >= 1).to(done.dtype)
        return done

    def window_loss(self, states, actions, goal, done, valid_steps=None):
        if valid_steps is None and hasattr(self.spec, 'free_task'):
            valid_steps = torch.ones_like(done, dtype=torch.bool)
        if valid_steps is not None:
            return self.masked_window_loss(states, actions, goal, done, valid_steps)
        done = self.completion_labels(states, done)
        state = states[0]
        loss_state = states.new_zeros(())
        loss_term = states.new_zeros(())
        errors, accuracies = [], []
        for t in range(len(actions)):
            state, _, logits = self.fit_predict(state, actions[t], goal)
            sq = self.spec.error_vector(state, states[t + 1]).square()
            loss_state = loss_state + sq.sum(-1).mean()
            loss_term = loss_term + F.binary_cross_entropy_with_logits(logits, done[t])
            errors.append(sq.detach().mean())
            accuracies.append(((logits.detach() > 0) == done[t].bool()).float().mean())
        k = len(actions)
        loss = (loss_state + loss_term) / k
        return loss, {'state_loss': loss_state.detach() / k, 'term_bce': loss_term.detach() / k,
                      'nse_1step': errors[0], 'nse_kstep': errors[-1],
                      'term_acc': torch.stack(accuracies).mean()}

    def masked_window_loss(self, states, actions, goal, done, valid_steps):
        if valid_steps.shape != done.shape or valid_steps.dtype != torch.bool:
            raise ValueError('Valid-step mask must be boolean and align with termination labels')
        if not valid_steps[0].all() or (valid_steps[1:] & ~valid_steps[:-1]).any():
            raise ValueError('Each window must contain a nonempty valid prefix')
        count = valid_steps.sum()
        done = self.completion_labels(states, done)
        state = states[0]
        loss_state = states.new_zeros(())
        loss_term = states.new_zeros(())
        correct = states.new_zeros(())
        completion_metrics = {k: states.new_zeros(()) for k in
                              ('completion_brier_physical', 'completion_probability_excess',
                               'completion_false_positive_fraction', 'completion_incomplete_fraction',
                               'completion_missed_fraction', 'completion_positive_fraction')}
        last_error = states.new_zeros(states.shape[1])
        for t in range(len(actions)):
            valid = valid_steps[t]
            # Inactive rows use a finite recorded start, avoiding extrapolation past termination.
            input_state = torch.where(valid[:, None], state, states[0])
            action = torch.where(valid[:, None], actions[t], 0.)
            predicted, _, logits = self.fit_predict(input_state, action, goal)
            target = torch.where(valid[:, None], states[t + 1], states[0])
            compared = torch.where(valid[:, None], predicted, target)
            sq = self.spec.error_vector(compared, target).square()
            loss_state = loss_state + sq.sum()
            logits = torch.where(valid, logits, 0.)
            labels = torch.where(valid, done[t], 0.)
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            loss_term = loss_term + torch.where(valid, bce, 0.).sum()
            correct = correct + (valid & ((logits.detach() > 0) == labels.bool())).sum()
            if hasattr(self.spec, 'free_task'):
                probability = logits.detach().sigmoid()
                rho = self.spec.field(predicted.detach(), 'cut_progress')[..., 0]
                completion_metrics['completion_brier_physical'] += torch.where(
                    valid, (probability - labels).square(), 0.).sum()
                completion_metrics['completion_probability_excess'] += torch.where(
                    valid, (probability - rho).clamp_min(0), 0.).sum()
                completion_metrics['completion_false_positive_fraction'] += (
                    valid & (probability >= .5) & ~labels.bool()).sum()
                completion_metrics['completion_incomplete_fraction'] += (valid & ~labels.bool()).sum()
                completion_metrics['completion_missed_fraction'] += (
                    valid & (probability < .5) & labels.bool()).sum()
                completion_metrics['completion_positive_fraction'] += (valid & labels.bool()).sum()
            error = sq.detach().mean(-1)
            last_error = torch.where(valid, error, last_error)
            if t == 0:
                first_error = error.mean()
            state = torch.where(valid[:, None], predicted, state)
        loss = (loss_state + loss_term) / count
        return loss, {'state_loss': loss_state.detach() / count,
                      'term_bce': loss_term.detach() / count,
                      'nse_1step': first_error, 'nse_kstep': last_error.mean(),
                      'term_acc': correct.detach() / count,
                      **({key: value.detach() / count for key, value in completion_metrics.items()}
                         if hasattr(self.spec, 'free_task') else {})}
