"""World-model loss decomposition for gradient diagnostics (no training).

``window_loss_components`` repeats ``CuttingWorldModel.masked_window_loss`` but keeps the
state loss per physical field. Its components sum to the training loss numerator, so
gradients of the components sum to the training gradient.
"""
import torch
import torch.nn.functional as F


def field_slices(spec, state):
    """Columns of ``spec.error_vector`` that belong to each physical field."""
    slices, start = {}, 0
    for name in spec.physical_fields:
        if name in getattr(spec, 'pose_fields', ()):
            raise ValueError('Pose-field error widths differ from field widths: ' + name)
        width = spec.field(state, name).reshape(*state.shape[:-1], -1).shape[-1]
        slices[name] = slice(start, start + width)
        start += width
    return slices


def window_loss_components(wm, states, actions, goal, done, valid_steps):
    """Summed squared state error per physical field, summed terminal BCE and valid count.

    (sum of field sums + terminal sum) / count equals ``wm.window_loss`` with the mask.
    """
    if valid_steps.shape != done.shape or valid_steps.dtype != torch.bool:
        raise ValueError('Valid-step mask must be boolean and align with termination labels')
    spec = wm.spec
    slices = field_slices(spec, states[0])
    fields = {name: states.new_zeros(()) for name in slices}
    terminal = states.new_zeros(())
    state = states[0]
    for t in range(len(actions)):
        valid = valid_steps[t]
        input_state = torch.where(valid[:, None], state, states[0])
        action = torch.where(valid[:, None], actions[t], 0.)
        # Same route as the fitting loss (wm.terminal_gradient); set it to 'dynamics' to
        # measure the source BCE gradient on the dynamics.
        predicted, _, logits = wm.fit_predict(input_state, action, goal)
        target = torch.where(valid[:, None], states[t + 1], states[0])
        compared = torch.where(valid[:, None], predicted, target)
        sq = spec.error_vector(compared, target).square()
        if sq.shape[-1] != max(s.stop for s in slices.values()):
            raise ValueError('Field slices do not cover the error vector')
        for name, columns in slices.items():
            fields[name] = fields[name] + sq[:, columns].sum()
        logits = torch.where(valid, logits, 0.)
        labels = torch.where(valid, done[t], 0.)
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        terminal = terminal + torch.where(valid, bce, 0.).sum()
        state = torch.where(valid[:, None], predicted, state)
    return fields, terminal, valid_steps.sum()


def flat_grad(loss, parameters, retain_graph=True):
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for g, p in zip(grads, parameters)])


def adam_preconditioner(optimizer, parameters):
    """1 / (sqrt(bias-corrected second moment) + eps) per parameter, flattened."""
    parts = []
    for p in parameters:
        state = optimizer.state.get(p)
        if not state or 'exp_avg_sq' not in state:
            raise ValueError('Optimizer has no second-moment state for a parameter')
        group = next(g for g in optimizer.param_groups if any(q is p for q in g['params']))
        step = float(state['step'])
        v_hat = state['exp_avg_sq'] / (1 - group['betas'][1] ** step)
        parts.append((1 / (v_hat.sqrt() + group['eps'])).reshape(-1))
    return torch.cat(parts)


def cosine_matrix(vectors):
    names = list(vectors)
    stacked = torch.stack([vectors[n].double() for n in names])
    unit = stacked / stacked.norm(dim=1, keepdim=True).clamp_min(1e-30)
    cos = unit @ unit.T
    return {a: {b: float(cos[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)}
