import torch

from cutting_world_model.wm_diagnostics import (adam_preconditioner, cosine_matrix, flat_grad,
                                                window_loss_components)
from test_critic_target import control_learner


def test_components_sum_to_the_training_loss_and_gradient():
    learner, replay = control_learner()
    wm = learner.wm
    torch.manual_seed(3)
    batch = replay.sample_windows(8, 3)
    args = (batch.states, batch.a_exec, batch.xi, batch.done, batch.valid_steps)
    assert not batch.valid_steps.all()  # the fixture contains episode ends inside windows
    fields, terminal, count = window_loss_components(wm, *args)
    assert set(fields) == set(wm.spec.physical_fields)
    loss, metrics = wm.window_loss(*args)
    total = (sum(fields.values()) + terminal) / count
    torch.testing.assert_close(total, loss)
    torch.testing.assert_close(terminal / count, metrics['term_bce'])
    params = list(wm.dynamics.parameters())
    parts = [flat_grad(v / count, params) for v in (*fields.values(), terminal)]
    torch.testing.assert_close(sum(parts), flat_grad(loss, params, retain_graph=False), rtol=1e-5, atol=1e-7)


def test_adam_preconditioner_and_cosines():
    learner, replay = control_learner()
    learner.fit_world_model(replay)
    params = list(learner.wm.dynamics.parameters())
    scale = adam_preconditioner(learner.wopt, params)
    assert scale.numel() == sum(p.numel() for p in params) and bool((scale > 0).all())
    cos = cosine_matrix({'a': torch.tensor([1., 0.]), 'b': torch.tensor([1., 1.]), 'c': torch.tensor([-1., 0.])})
    assert abs(cos['a']['b'] - 2 ** -.5) < 1e-12 and cos['a']['c'] == -1 and cos['b']['b'] == 1
