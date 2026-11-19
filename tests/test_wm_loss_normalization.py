"""Regression checks for separate dynamics/BCE sample populations."""
from types import SimpleNamespace
import copy
import pytest
import torch
from test_pose_depth_v3 import model, transitions
from cutting_world_model.policy import SliceItCritic
from cutting_world_model.trainer import OnlineLearner
from third_party.supermimic.trainer_config import WMTrainerCfg

def data(wm, s, g, flags, valid=None):
    steps = len(flags) - 1
    n = len(flags[0])
    return SimpleNamespace(states=transitions(wm.spec, s, flags),
        a_exec=torch.full((steps, n, 6), .3, dtype=s.dtype),
        xi=g.repeat(n, 1), done=torch.zeros(steps, n, dtype=s.dtype),
        valid_steps=torch.ones(steps, n, dtype=torch.bool) if valid is None else valid)

def loss(wm, b):
    return wm.window_loss(b.states, b.a_exec, b.xi, b.done, b.valid_steps)

def gradients(wm, b):
    wm.zero_grad(set_to_none=True)
    total, metrics = loss(wm, b)
    total.backward()
    return {k: p.grad.detach().clone() for k,p in wm.named_parameters() if p.grad is not None}, metrics

def test_frozen_rows_do_not_dilute_dynamics_loss_or_gradient():
    wm,s,g=model()
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight, std=.01)
    # Detach BCE's route only in this test to isolate dynamics-error gradients.
    wm.terminal_gradient='head'
    alone=data(wm,s,g,[(0,),(0,),(0,)])
    mixed=data(wm,s,g,[(0,1,1,1)]*3)
    grad1,m1=gradients(wm,alone)
    grad2,m2=gradients(wm,mixed)
    assert m1['state_loss'] > 0
    torch.testing.assert_close(m1['state_loss'],m2['state_loss'])
    for k in grad1:
        if k.startswith('dynamics.'):
            torch.testing.assert_close(grad1[k],grad2[k],rtol=1e-9,atol=1e-10)
    assert any(v.abs().sum()>0 for k,v in grad2.items() if k.startswith('term_head.'))
    assert m2['dynamics_fit_count']==2 and m2['dynamics_masked_count']==6

@pytest.mark.parametrize('microbatch',[1,2,3,4])
@pytest.mark.parametrize('route',['head','dynamics'])
def test_real_trainer_gradients_and_loss_match_whole_batch(microbatch, route):
    wm,s,g=model()
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight,std=.01)
    torch.nn.init.normal_(wm.term_head.net[-1].weight,std=.05)
    wm.terminal_gradient=route
    # Unequal valid prefixes, pure-latched and pure-fit microbatches, and first contact.
    valid=torch.tensor([[1,1,1,1],[1,1,1,0],[1,0,1,0]],dtype=torch.bool)
    b=data(wm,s,g,[(1,1,0,0),(1,1,0,1),(1,1,1,1),(1,1,1,1)],valid)
    expected_grad,metrics=gradients(wm,b)
    total=loss(wm,b)[0].detach()
    cfg=WMTrainerCfg(compile=None,lr=0.,window=3,batch_size=4,batches_per_update=1,grad_clip=1e12)
    learner=OnlineLearner(wm,torch.nn.Linear(1,1),SliceItCritic(wm.spec.state_dim,8),
        microbatch=microbatch,wm_cfg=cfg,wm_terminal_gradient=route)
    result=learner.fit_world_model(SimpleNamespace(sample_windows=lambda *a:b))
    assert result['state_loss']==pytest.approx(float(metrics['state_loss']),rel=1e-9)
    assert result['term_bce']==pytest.approx(float(metrics['term_bce']),rel=1e-9)
    assert result['loss']==pytest.approx(float(total),rel=1e-9)
    for k,p in wm.named_parameters():
        if k in expected_grad:
            torch.testing.assert_close(p.grad,expected_grad[k],rtol=1e-8,atol=1e-10)

def test_no_latch_matches_old_valid_count_formula():
    wm,s,g=model('pose_depth_v2')
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight,std=.01)
    states=torch.stack([s.repeat(2,1)]*3)
    valid=torch.tensor([[1,1],[1,0]],dtype=torch.bool)
    actions=torch.zeros(2,2,6,dtype=s.dtype)
    goals=g.repeat(2,1); done=torch.zeros(2,2,dtype=s.dtype)
    total, m=wm.window_loss(states,actions,goals,done,valid)
    torch.testing.assert_close(total.detach(),m['state_loss']+m['term_bce'])

def test_checkpoint_records_rule_and_old_v3_resume_warns():
    wm,_,_=model()
    def learner():
        candidate=copy.deepcopy(wm)
        return OnlineLearner(candidate,torch.nn.Linear(1,1),
            SliceItCritic(candidate.spec.state_dim,8),
            wm_cfg=WMTrainerCfg(compile=None))
    saved=learner().checkpoint()
    assert saved['wm_loss_normalization']=='separate_fit_valid_v1'
    learner().restore(saved)
    old=copy.deepcopy(saved)
    old.pop('wm_loss_normalization')
    with pytest.warns(RuntimeWarning,match='changes the training loss'):
        learner().restore(old)
    saved['wm_loss_normalization']='unknown'
    with pytest.raises(ValueError,match='normalization differs'):
        learner().restore(saved)

def test_trainer_accepts_unpadded_v3_windows_without_explicit_mask():
    wm,s,g=model()
    b=data(wm,s,g,[(0,1),(0,1),(0,1)])
    reference=loss(wm,b)[0].detach()
    del b.valid_steps
    cfg=WMTrainerCfg(compile=None,lr=0.,window=2,batch_size=2,batches_per_update=1)
    learner=OnlineLearner(wm,torch.nn.Linear(1,1),SliceItCritic(wm.spec.state_dim,8),
                          microbatch=1,wm_cfg=cfg)
    m=learner.fit_world_model(SimpleNamespace(sample_windows=lambda *a:b))
    assert m['loss']==pytest.approx(float(reference),rel=1e-9)
    assert m['dynamics_masked_fraction']==.5

@pytest.mark.parametrize('microbatch',[1,2,4])
def test_diagnostic_decomposition_matches_training_loss_and_gradient(microbatch):
    from cutting_world_model.wm_diagnostics import window_loss_components
    wm,s,g=model()
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight,std=.01)
    b=data(wm,s,g,[(1,1,0,0),(1,1,0,1),(1,1,1,1)])
    reference=loss(wm,b)[0]
    params=list(wm.parameters())
    expected=torch.autograd.grad(reference,params)
    count=b.valid_steps.sum()
    fit=(b.valid_steps & ~wm.spec.contact(b.states[:-1])[...,0]).sum()
    parts=[]
    for start in range(0,4,microbatch):
        part=slice(start,min(start+microbatch,4))
        fields,bce,_=window_loss_components(wm,b.states[:,part],b.a_exec[:,part],
            b.xi[part],b.done[:,part],b.valid_steps[:,part],
            normalization_counts=(fit,count))
        parts.append((sum(fields.values())+bce)/count)
    result=sum(parts)
    torch.testing.assert_close(result,reference)
    actual=torch.autograd.grad(result,params)
    for a,e in zip(actual,expected):
        torch.testing.assert_close(a,e,rtol=1e-8,atol=1e-10)
    fields,bce,local_count=window_loss_components(wm,b.states,b.a_exec,b.xi,b.done,b.valid_steps)
    torch.testing.assert_close((sum(fields.values())+bce)/local_count,reference)
