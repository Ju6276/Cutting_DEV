"""WM-fitting variant 'head': the terminal BCE trains only the terminal head; policy BPTT is unchanged."""
import pytest
import torch

from cutting_world_model.wm_diagnostics import flat_grad, window_loss_components
from test_critic_target import control_learner


def components(learner, replay, route):
    learner.wm.terminal_gradient = route
    torch.manual_seed(3)
    batch = replay.sample_windows(8, 3)
    args = (batch.states, batch.a_exec, batch.xi, batch.done, batch.valid_steps)
    fields, terminal, count = window_loss_components(learner.wm, *args)
    loss, _ = learner.wm.window_loss(*args)
    return sum(fields.values()), terminal, count, loss


def test_head_route_removes_the_bce_gradient_from_the_dynamics_only():
    learner, replay = control_learner()
    wm = learner.wm
    dynamics, head = list(wm.dynamics.parameters()), list(wm.term_head.parameters())
    state_d, term_d, count_d, loss_d = components(learner, replay, 'dynamics')
    state_h, term_h, count_h, loss_h = components(learner, replay, 'head')
    # Forward values are identical; only the gradient route differs.
    torch.testing.assert_close(loss_h, loss_d)
    torch.testing.assert_close(term_h, term_d)
    assert float(flat_grad(term_d, dynamics).norm()) > 0
    assert float(flat_grad(term_h, dynamics).norm()) == 0
    torch.testing.assert_close(flat_grad(term_h, head), flat_grad(term_d, head))
    torch.testing.assert_close(flat_grad(state_h, dynamics), flat_grad(state_d, dynamics))
    # The training loss gradient on the dynamics equals the state-loss gradient alone.
    torch.testing.assert_close(flat_grad(loss_h, dynamics), flat_grad(state_h / count_h, dynamics))


def test_policy_imagination_keeps_the_terminal_probability_path():
    grads = {}
    for route in ('dynamics', 'head'):
        learner, replay = control_learner(wm_terminal_gradient=route)
        assert learner.wm.terminal_gradient == route
        torch.manual_seed(5)
        starts = replay.sample_starts(6, learner.pcfg.history)
        ro = learner.imagine(starts.state, starts.hist_states, starts.xi)
        params = list(learner.policy.parameters())
        grads[route] = flat_grad(ro.survival[-1].sum(), params)
        assert float(grads[route].norm()) > 0
    torch.testing.assert_close(grads['head'], grads['dynamics'])


def test_fit_updates_only_the_head_from_the_bce_with_route_head():
    learner, replay = control_learner(wm_terminal_gradient='head')
    before = {k: v.clone() for k, v in learner.wm.state_dict().items()}
    learner.fit_world_model(replay)
    after = learner.wm.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith('dynamics.'))
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith('term_head.'))


def test_checkpoint_records_the_route_and_a_change_must_be_explicit():
    source, _ = control_learner()
    saved = source.checkpoint()
    assert saved['wm_terminal_gradient'] == 'dynamics'
    target, _ = control_learner(wm_terminal_gradient='head')
    with pytest.raises(ValueError, match='terminal-gradient route differs'):
        target.restore(saved)
    target.restore(saved, change_wm_terminal_gradient=True)
    assert target.wm.terminal_gradient == 'head'
    assert target.checkpoint()['wm_terminal_gradient'] == 'head'
    # Checkpoints written before the option existed used the source route.
    del saved['wm_terminal_gradient']
    again, _ = control_learner()
    again.restore(saved)


def test_unknown_route_is_rejected():
    with pytest.raises(ValueError, match='terminal_gradient'):
        control_learner(wm_terminal_gradient='both')
