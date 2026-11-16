import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from cutting_world_model.object_slip import validate, planar_rms, penalty, observe
from cutting_world_model.free_cut import FreeCutTask
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.world_model import CuttingWorldModel
from test_pose_depth import model as depth_model

CFG = dict(revision='planar_mass_rms_v1', free_m=.025, failure_m=.06, weight=20.)

def model(backend='mujoco_warp_disect'):
    old, state, goal = depth_model(backend)
    meta = copy.deepcopy(old.spec.metadata)
    meta['free_cut_task']['object_slip'] = dict(CFG)
    frame = {k: old.spec.field(state,k)[0].numpy() for k in old.spec.base_fields}
    frame['object_slip'] = np.array([0.])
    meta = ControlStateSpec.metadata_from_frame(meta,frame)
    new = CuttingWorldModel(meta,hidden=(12,12)).double()
    return new,new.spec.pack(frame,dtype=torch.double)[None],goal

def test_motion_metric_preserves_translation_and_opposing_fragment_motion():
    reference=torch.zeros(1,2,3,dtype=torch.double)
    position=reference.clone();position[:,:,0]=.04;position[:,:,1]=100.
    mass=torch.tensor([[1.,3.]],dtype=torch.double)
    assert planar_rms(position,reference,mass,(0,2)).item()==pytest.approx(.04)
    position[:,0,0]=-.04
    assert planar_rms(position,reference,mass,(0,2)).item()==pytest.approx(.04)
    position[:,1,0]=0
    assert planar_rms(position,reference,mass,(0,2)).item()==pytest.approx(.02)

def test_reset_references_are_episode_local_and_not_group_centroid():
    positions=torch.zeros(2,2,3)
    env=SimpleNamespace(worlds=2,step_count=torch.zeros(2,dtype=torch.long),
        metadata=dict(physical_backend='mujoco_warp_disect',free_cut_task=dict(object_slip=CFG)),
        physics=SimpleNamespace(material=SimpleNamespace(model=SimpleNamespace(particle_mass=torch.ones(4)))))
    frame={'particle_q':positions};observe(env,frame)
    env.step_count[:]=1;positions[0,:,0]=.04;positions[1,:,2]=.02
    observe(env,frame)
    torch.testing.assert_close(frame['object_slip'].flatten(),torch.tensor([.04,.02]))
    env.step_count[0]=0;observe(env,frame)
    torch.testing.assert_close(frame['object_slip'].flatten(),torch.tensor([0.,.02]))
    assert frame['object_reference_q'][1].eq(0).all()

@pytest.mark.parametrize('key,value', [('free_m',.06),('failure_m',0.),('weight',-1.),
                                      ('free_m',float('nan')),('weight',True)])
def test_invalid_constraint_rejected(key,value):
    config={**CFG,key:value}
    with pytest.raises(ValueError):validate(config)

def test_free_zone_soft_penalty_and_bounded_episode_cost():
    values=torch.tensor([0.,.025,.0425,.06,.09],dtype=torch.double,requires_grad=True)
    cost=penalty(values,CFG,180)
    torch.testing.assert_close(cost,torch.tensor([0.,0.,5/180,20/180,20/180],dtype=torch.double))
    assert torch.autograd.grad(cost.sum(),values)[0][2]>0
    assert cost.max()*180<=20.

@pytest.mark.parametrize('backend',['mujoco_warp_disect','mujoco_warp_blademaster'])
@pytest.mark.parametrize('distance',[0.,.0425,.061])
def test_physical_and_wm_reward_termination_agree(backend,distance):
    wm,s,g=model(backend);sp=wm.spec
    s[:,sp.layout['success_counter']]=sp.heads.env['successes_threshold']
    action=torch.zeros(1,6,dtype=s.dtype)
    fields=sp.integrate_physical(s,torch.zeros(1,sp.delta_dim,dtype=s.dtype))
    fields['object_slip']=torch.tensor([[distance]],dtype=s.dtype,requires_grad=True)
    nxt,reward,done=sp.derive(s,fields,action,g)
    obs=sp.field(nxt,'observation');peak=sp.field(nxt,'peak_wrench')
    events=sp.free_task.events(obs,peak,None,sp.field(nxt,'step_count')[...,0],
        pose=sp.fk(fields['joint_q']),board=dict(knife_min_board_gap=torch.ones(1,1)*1e-6,
        knife_board_contact=torch.ones(1,1,dtype=torch.bool)),
        success_counter=sp.field(s,'success_counter'),object_slip=fields['object_slip'][...,0])
    actual=sp.free_task.reward(sp.field(s,'cut_progress')[...,0],fields['cut_progress'][...,0],
        obs,peak,g[:,-1],events)
    torch.testing.assert_close(reward,actual)
    torch.testing.assert_close(done,events['terminated'])
    assert events['object_slip_failure'].item()==(distance>.06)
    assert events['success'].item()==(distance<=.06)
    assert sp.pose_success(nxt).item()==(distance<=.06)
    assert sp.terminal_features(nxt).shape[-1]==15
    if .025<distance<.06:
        assert torch.autograd.grad(reward.sum(),fields['object_slip'])[0].item()<0
    if distance>.06:
        assert reward.item() < -100 and not events['timeout'].item()

def test_failure_settlement_once_with_simultaneous_force_and_arrival():
    wm,s,g=model();task=wm.spec.free_task
    obs=torch.zeros(1,25);peak=torch.tensor([[100.,0,0,0,0,0]])
    ev=task.events(obs,peak,None,torch.tensor([180]),pose=torch.zeros(1,7),
        board=dict(knife_min_board_gap=torch.ones(1,1)*1e-6,
        knife_board_contact=torch.ones(1,1,dtype=torch.bool)),
        success_counter=torch.tensor([99.]),object_slip=torch.tensor([.07]))
    r=task.reward(torch.zeros(1),torch.zeros(1),obs,peak,torch.tensor([.03]),ev)
    raw=task.step_reward(torch.zeros(1),torch.zeros(1),obs,peak,torch.tensor([.03]),torch.tensor([.07]))
    assert (r-raw).item()==pytest.approx(-100)
    assert not ev['success'].item() and not ev['timeout'].item()

def test_learning_checkpoint_and_old_layout_incompatibility():
    wm,s,g=model();sp=wm.spec
    s[:,sp.layout['object_slip']]=.04
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight,std=1e-4)
    action=torch.full((1,6),.1,dtype=s.dtype,requires_grad=True)
    nxt,r,logits=wm.step(s,action,g)
    grad=torch.autograd.grad(r.sum()+logits.sum(),action,retain_graph=True)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    loss,_=wm.window_loss(torch.stack((s,nxt.detach())),action.detach()[None],g,torch.zeros(1,1))
    opt=torch.optim.Adam(wm.parameters(),lr=1e-4);opt.zero_grad();loss.backward();opt.step()
    assert all(torch.isfinite(p).all() for p in wm.parameters())
    restored,_,_=model();restored.load_state_dict(wm.state_dict())
    for a,b in zip(wm.step(s,action,g),restored.step(s,action,g)):torch.testing.assert_close(a,b)
    old,_,_=depth_model()
    with pytest.raises(RuntimeError):wm.load_state_dict(old.state_dict())
    assert sp.shapes['observation']==old.spec.shapes['observation']

def test_logging_keeps_new_failure_cause():
    from cutting_world_model.episode_log import ending
    assert ending(False,False,False,False,False,True)=='object_motion'
