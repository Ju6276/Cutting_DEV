"""Route B force-risk constraint in the learner: head fit, constrained actor, dual update."""
import pytest
import torch

from test_critic_target import control_learner
from test_wm_terminal_gradient import perturbed


def risk_learner(**risk):
    learner, replay = control_learner(critic_target='real', actor_horizon=3,
                                      risk_constraint={'head_batch': 8, **risk})
    perturbed(learner)
    return learner, replay


def test_risk_head_fits_and_reports():
    learner, replay = risk_learner()
    before = [p.clone() for p in learner.risk_head.parameters()]
    metrics = learner.fit_risk_head(replay)
    assert torch.isfinite(torch.tensor(metrics['risk_pinball']))
    assert any(not torch.equal(a, b) for a, b in zip(before, learner.risk_head.parameters()))


def test_risk_penalty_reaches_the_policy_through_the_action():
    learner, replay = risk_learner(threshold=-10.)  # every step violates: penalty always active
    starts = replay.sample_starts(4, learner.pcfg.history)
    ro = learner.imagine(starts.state, starts.hist_states, starts.xi)
    assert ro.risk.shape == (3, 4)
    cost = (ro.survival[:-1].detach() * torch.relu(ro.risk + 10.)).sum()
    grads = torch.autograd.grad(cost, list(learner.policy.parameters()), allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_dual_update_and_checkpoint_roundtrip():
    learner, replay = risk_learner(lambda_init=1., lambda_lr=10., target_rate=.1)
    assert learner.update_risk_lambda(5, 10) == pytest.approx(1. + 10. * (.5 - .1))
    assert learner.update_risk_lambda(0, 10) == pytest.approx(5. - 1.)
    assert learner.update_risk_lambda(0, 0) == pytest.approx(4.)
    for _ in range(10):
        learner.update_risk_lambda(0, 1000)
    assert learner.risk_lambda == 0.
    learner.fit_risk_head(replay)
    saved = learner.checkpoint()
    other, _ = risk_learner(lambda_init=1., lambda_lr=10., target_rate=.1)
    other.restore(saved)
    assert other.risk_lambda == learner.risk_lambda and other.risk_head_updates == learner.risk_head_updates
    plain, _ = control_learner(critic_target='real', actor_horizon=3)
    with pytest.raises(ValueError):
        plain.restore(saved)


def ensemble_learner(**risk):
    learner, replay = control_learner(critic_target='real', actor_horizon=3,
                                      risk_constraint={'estimator': 'ensemble', 'ensemble_size': 3, **risk})
    perturbed(learner)
    return learner, replay


def test_ensemble_members_fit_without_touching_the_main_wm():
    learner, replay = ensemble_learner()
    assert len(learner.risk_members) == 2
    main = [p.clone() for p in learner.wm.parameters()]
    before = [p.clone() for p in learner.risk_members[0].parameters()]
    updates = learner.wm_updates
    metrics = learner.fit_risk_head(replay)
    assert set(metrics) == {'risk_member_loss_0', 'risk_member_loss_1'}
    assert all(torch.equal(a, b) for a, b in zip(main, learner.wm.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(before, learner.risk_members[0].parameters()))
    assert learner.wm_updates == updates


def test_ensemble_upper_bound_reaches_the_policy_and_checkpoints():
    learner, replay = ensemble_learner(threshold=-10.)
    starts = replay.sample_starts(4, learner.pcfg.history)
    ro = learner.imagine(starts.state, starts.hist_states, starts.xi)
    assert ro.risk.shape == (3, 4)
    grads = torch.autograd.grad(ro.risk.sum(), list(learner.policy.parameters()), allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    learner.fit_risk_head(replay)
    other, _ = ensemble_learner(threshold=-10.)
    other.restore(learner.checkpoint())
    for a, b in zip(learner.risk_members, other.risk_members):
        assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
