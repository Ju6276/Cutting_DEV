"""Source residual MLP, autoregressive loss and terminal head with cutting task interfaces."""
# Adapted from SuperMimic-Reproduced ba610534; MIT, third_party/supermimic/LICENSE.
import torch
from torch import nn
from torch.nn import functional as F

from third_party.supermimic.dynamics import DynamicsMLP, MLP
from .state_spec import CuttingStateSpec
from .control_state import ControlStateSpec


class CuttingWorldModel(nn.Module):
    def __init__(self, metadata, hidden=(2048, 2048, 2048)):
        super().__init__()
        layout = metadata.get('prediction_layout')
        if layout not in (None, 'control_history_v2'):
            raise ValueError('Unsupported prediction layout: ' + str(layout))
        spec_class = ControlStateSpec if layout == 'control_history_v2' else CuttingStateSpec
        self.spec = spec_class(metadata)
        self.dynamics = DynamicsMLP(self.spec.dynamics_dim, 6, 7, hidden, self.spec.delta_dim)
        self.term_head = MLP(26, (64, 64), 1)

    def step(self, state, action, goal):
        executed = self.spec.execute(action)
        delta = self.dynamics(self.spec.dynamics_features(state), executed, goal)
        fields = self.spec.integrate_physical(state, delta)
        next_state, reward, done = self.spec.derive(state, fields, executed, goal)
        feats = self.spec.terminal_features(next_state)
        logits = self.term_head(torch.cat((feats, feats.abs()), -1)).squeeze(-1)
        return next_state, reward, logits

    def enable_compile(self, mode="default"):
        self._step_fn = torch.compile(self.step, mode=mode, dynamic=False) if mode else self.step

    def predict(self, state, action, goal):
        return getattr(self, '_step_fn', self.step)(state, action, goal)

    def window_loss(self, states, actions, goal, done):
        state = states[0]
        loss_state = states.new_zeros(())
        loss_term = states.new_zeros(())
        errors, accuracies = [], []
        for t in range(len(actions)):
            state, _, logits = self.predict(state, actions[t], goal)
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
