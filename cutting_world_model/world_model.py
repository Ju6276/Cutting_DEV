"""Source residual MLP, autoregressive loss and terminal head with cutting task interfaces."""
# Adapted from SuperMimic-Reproduced ba610534; MIT, third_party/supermimic/LICENSE.
import torch
from torch import nn
from torch.nn import functional as F

from third_party.supermimic.dynamics import DynamicsMLP, MLP
from .metric_counts import RATE_COUNTS
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
        if metadata.get('free_cut_task', {}).get('version') == 'free_cut_v4':
            raise ValueError('v4 is currently physical-task/frozen-policy evaluation only; do not reuse v3 WM completion or critic labels')
        layout = metadata.get('prediction_layout')
        if layout not in (None, 'control_history_v2'):
            raise ValueError('Unsupported prediction layout: ' + str(layout))
        spec_class = ControlStateSpec if layout == 'control_history_v2' else CuttingStateSpec
        if metadata.get('free_cut_task'):
            if layout != 'control_history_v2':
                raise ValueError('free_cut requires the compact control-history layout')
            from .free_cut_state import FreeCutStateSpec
            spec_class = FreeCutStateSpec
            if metadata['free_cut_task']['version'] == 'free_cut_pose_v1':
                from .pose_cut import PoseCutStateSpec
                spec_class = PoseCutStateSpec
        if metadata.get('free_cut_task', {}).get('version') == 'blademaster_pose_v1':
            from .blademaster_state import BladeMasterStateSpec
            spec_class = BladeMasterStateSpec
        self.spec = spec_class(metadata)
        self.dynamics = DynamicsMLP(self.spec.dynamics_dim, 6, 7, hidden, self.spec.delta_dim)
        if metadata.get('free_cut_task', {}).get('version') in ('free_cut_v2', 'free_cut_v3', 'free_cut_pose_v1'):
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

    def _transition(self, state, action, goal):
        executed = self.spec.execute(action)
        # Physics prediction uses the target pose only; task conditions enter reward/termination.
        delta = self.dynamics(self.spec.dynamics_features(state), executed, goal[..., :7])
        fields = self.spec.integrate_physical(state, delta)
        next_state, reward, done = self.spec.derive(state, fields, executed, goal)
        release_fraction = None
        if self.spec.metadata.get('free_cut_task', {}).get('version') == 'free_cut_v3':
            # Use the original fraction, not a subtraction of nearly equal progress
            # values. The latter can create enormous spurious float32 gradients.
            release_fraction = delta[..., self.spec.delta_slices['cut_progress']].sigmoid()[..., 0]
            old = self.spec.field(state, 'cut_progress')[..., 0]
            release_fraction = torch.where(old < 1, release_fraction, 0.)
        return next_state, reward, release_fraction

    def transition(self, state, action, goal):
        next_state, reward, _ = self._transition(state, action, goal)
        return next_state, reward

    def terminal_logits(self, next_state, goal=None, previous_state=None, release_fraction=None):
        feats = self.spec.terminal_features(next_state, goal)
        raw = self.term_head(torch.cat((feats, feats.abs()), -1)).squeeze(-1)
        if self.spec.metadata.get('free_cut_task', {}).get('version') == 'free_cut_v3':
            # Completion of a previously incomplete cut releases ALL remaining material.
            # Bound the per-step completion hazard by the predicted fraction of the
            # remainder released THIS step. A rho-only bound would still pay completion
            # repeatedly at constant partial progress through accumulated hazard.
            if previous_state is None:
                raise ValueError('free_cut_v3 completion needs the previous progress state')
            rho = self.spec.field(next_state, 'cut_progress')[..., 0].clamp(0, 1)
            old = self.spec.field(previous_state, 'cut_progress')[..., 0].clamp(0, 1)
            tiny = torch.finfo(rho.dtype).tiny
            if release_fraction is None:
                # State-only reconstruction is for diagnostics, not a BPTT path.
                if torch.is_grad_enabled() and (rho.requires_grad or old.requires_grad):
                    raise ValueError('Differentiable completion requires the original release fraction')
                fraction = ((rho - old).clamp_min(0) / (1 - old).clamp_min(tiny)).clamp(0, 1)
            else:
                fraction = release_fraction.clamp(0, 1)
            log_fraction = fraction.clamp_min(tiny).log()
            log_not_fraction = (1 - fraction).clamp_min(tiny).log()
            return (log_fraction + F.logsigmoid(raw)
                    - torch.logaddexp(log_not_fraction, log_fraction + F.logsigmoid(-raw)))
        return raw

    def step(self, state, action, goal):
        next_state, reward, fraction = self._transition(state, action, goal)
        logits = self.terminal_logits(next_state, goal, state, fraction)
        if hasattr(self.spec, 'free_task'):
            reward = reward + self.spec.completion_reward(next_state, logits)
        return next_state, reward, logits

    def _fit_step_head(self, state, action, goal):
        next_state, reward, fraction = self._transition(state, action, goal)
        logits = self.terminal_logits(next_state.detach(), goal, state.detach(),
                                      None if fraction is None else fraction.detach())
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
        if self.spec.metadata.get('free_cut_task', {}).get('version') in ('free_cut_pose_v1', 'blademaster_pose_v1'):
            return done  # source-style task termination, not physical separation
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
        errors = []
        correct = torch.zeros((), dtype=torch.int64, device=states.device)
        for t in range(len(actions)):
            state, _, logits = self.fit_predict(state, actions[t], goal)
            sq = self.spec.error_vector(state, states[t + 1]).square()
            loss_state = loss_state + sq.sum(-1).mean()
            loss_term = loss_term + F.binary_cross_entropy_with_logits(logits, done[t])
            errors.append(sq.detach().mean())
            correct += ((logits.detach() > 0) == done[t].bool()).sum()
        k = len(actions)
        loss = (loss_state + loss_term) / k
        return loss, {'state_loss': loss_state.detach() / k, 'term_bce': loss_term.detach() / k,
                      'nse_1step': errors[0], 'nse_kstep': errors[-1],
                      'term_acc': correct / done.numel(), 'term_correct_count': correct,
                      'metric_transition_count': correct.new_tensor(done.numel())}

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
        correct = torch.zeros((), dtype=torch.int64, device=states.device)
        completion_metrics = {k: states.new_zeros(()) for k in
                              ('completion_brier_physical', 'completion_probability_excess')}
        completion_counts = {key: correct.clone() for key in RATE_COUNTS if key != 'term_acc'}
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
            if hasattr(self.spec, 'free_task') and not hasattr(self.spec, 'pose_success'):
                probability = logits.detach().sigmoid()
                rho = self.spec.field(predicted.detach(), 'cut_progress')[..., 0]
                completion_metrics['completion_brier_physical'] += torch.where(
                    valid, (probability - labels).square(), 0.).sum()
                completion_metrics['completion_probability_excess'] += torch.where(
                    valid, (probability - rho).clamp_min(0), 0.).sum()
                completion_counts['completion_false_positive_fraction'] += (
                    valid & (probability >= .5) & ~labels.bool()).sum()
                completion_counts['completion_incomplete_fraction'] += (valid & ~labels.bool()).sum()
                completion_counts['completion_missed_fraction'] += (
                    valid & (probability < .5) & labels.bool()).sum()
                completion_counts['completion_positive_fraction'] += (valid & labels.bool()).sum()
            error = sq.detach().mean(-1)
            last_error = torch.where(valid, error, last_error)
            if t == 0:
                first_error = error.mean()
            state = torch.where(valid[:, None], predicted, state)
        loss = (loss_state + loss_term) / count
        return loss, {'state_loss': loss_state.detach() / count,
                      'term_bce': loss_term.detach() / count,
                      'nse_1step': first_error, 'nse_kstep': last_error.mean(),
                      'term_acc': correct / count, 'term_correct_count': correct,
                      'metric_transition_count': count,
                      **({key: value.detach() / count for key, value in completion_metrics.items()}
                         if hasattr(self.spec, 'free_task') and not hasattr(self.spec, 'pose_success') else {}),
                      **({key: value / count for key, value in completion_counts.items()}
                         if hasattr(self.spec, 'free_task') and not hasattr(self.spec, 'pose_success') else {}),
                      **({RATE_COUNTS[key]: value for key, value in completion_counts.items()}
                         if hasattr(self.spec, 'free_task') and not hasattr(self.spec, 'pose_success') else {})}
