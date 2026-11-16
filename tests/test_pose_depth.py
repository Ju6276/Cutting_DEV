import numpy as np
import pytest
import torch
from test_pose_cut import config
from test_state_spec import fixture
from cutting_world_model.pose_depth import depth_progress, PROGRESS_SEMANTICS
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.world_model import CuttingWorldModel

def model(backend='mujoco_warp_disect'):
    meta,frame=fixture()
    task=config();task['version']='pose_depth_v2';task.pop('initial_progress_fraction')
    frame.update(cut_progress=np.array([0.]),depth_span=np.array([.1-1e-6]),
                 step_count=np.array([0.]),observation=np.zeros(25))
    meta.update(free_cut_task=task,goal_dim=8,physical_backend=backend,
        progress_semantics=PROGRESS_SEMANTICS,
        board_pose_geometry=dict(vertices_in_ee=[[0,0,0]],board_origin_base=[0,0,-.1],
                                 board_normal_base=[0,0,1],stop_gap_m=1e-6))
    meta=ControlStateSpec.metadata_from_frame(meta,frame)
    wm=CuttingWorldModel(meta,hidden=(12,12)).double()
    state=wm.spec.pack(frame,dtype=torch.double)[None]
    goal=torch.tensor([[.06,0,0,0,0,0,1,.03]],dtype=torch.double)
    return wm,state,goal

def test_progress_is_bounded_monotone_and_cannot_farm_reentry():
    geo=dict(vertices_in_ee=[[0,0,0]],board_origin_base=[0,0,0],
             board_normal_base=[0,0,1],stop_gap_m=1e-6)
    pose=torch.tensor([[0,0,.1,0,0,0,1]],dtype=torch.double)
    span=torch.tensor([[.1-1e-6]],dtype=torch.double)
    old=torch.zeros_like(span);increments=[]
    for h in [.1,.05,.09,.05,1e-6,-.01]:
        pose[:,2]=h
        new=depth_progress(pose,geo,span,old);increments.append((new-old).item());old=new
    assert increments[0]==0 and increments[2]==0 and increments[3]==0 and increments[-1]==0
    assert sum(increments)==pytest.approx(1)
    assert old.item()==1

@pytest.mark.parametrize('backend',['mujoco_warp_disect','mujoco_warp_blademaster'])
def test_wm_has_no_spring_or_learned_progress_input_output(backend):
    wm,s,g=model(backend);sp=wm.spec
    assert 'cut_progress' not in sp.delta_slices
    assert 'cut_spring_ke' not in sp.layout
    indices=sp.dynamics_indices.tolist()
    assert not set(range(sp.layout['cut_progress'].start,sp.layout['cut_progress'].stop)) & set(indices)
    fields=sp.integrate_physical(s,torch.zeros(1,sp.delta_dim,dtype=s.dtype))
    assert fields['cut_progress'].item()==pytest.approx(0,abs=1e-12)
    torch.testing.assert_close(fields['depth_span'],sp.field(s,'depth_span'))
    nxt,reward,done=sp.derive(s,fields,torch.zeros(1,6,dtype=s.dtype),g)
    assert torch.isfinite(nxt).all() and torch.isfinite(reward).all()
    torch.testing.assert_close(wm.completion_labels(torch.stack((s,nxt)),torch.ones(1,1)),torch.ones(1,1))

def test_shared_backends_reward_and_model_are_identical_for_same_states():
    a,s,g=model();b,_,_=model('mujoco_warp_blademaster');b.load_state_dict(a.state_dict())
    action=torch.full((1,6),.1,dtype=s.dtype)
    for x,y in zip(a.step(s,action,g),b.step(s,action,g)):
        torch.testing.assert_close(x,y,atol=0,rtol=0)

def test_progress_gradient_matches_fk_and_learning_updates_remain_finite():
    wm,s,g=model();sp=wm.spec
    sp.board_geometry=dict(vertices_in_ee=[[0,0,0]],board_origin_base=[0,-.1,0],
                           board_normal_base=[0,1,0],stop_gap_m=1e-6)
    delta=torch.zeros(1,sp.delta_dim,dtype=s.dtype)
    delta[:,sp.delta_slices['joint_q']]=-.01
    delta.requires_grad_(True)
    fun=lambda d: sp.integrate_physical(s,d)['cut_progress']
    assert fun(delta).item()>0
    assert torch.autograd.gradcheck(fun,(delta,),eps=1e-6,atol=1e-5,rtol=1e-3)
    action=torch.full((1,6),.1,dtype=s.dtype,requires_grad=True)
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight,std=1e-4)
    nxt,reward,logits=wm.step(s,action,g)
    grad=torch.autograd.grad(reward.sum()+logits.sum(),action,retain_graph=True)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    loss,_=wm.window_loss(torch.stack((s,nxt.detach())),action.detach()[None],g,torch.zeros(1,1))
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in wm.parameters())

def test_depth_is_not_a_success_or_separation_label():
    wm,s,g=model();sp=wm.spec;task=sp.free_task
    obs=torch.zeros(2,25);obs[:,1]=1. # not at target even if depth=1
    board=dict(knife_min_board_gap=torch.full((2,1),1e-6),knife_board_contact=torch.ones(2,1,dtype=torch.bool))
    ev=task.events(obs,torch.zeros(2,6),torch.tensor([0.,1.]),torch.ones(2,dtype=torch.long),
                   pose=torch.zeros(2,7),board=board,success_counter=torch.zeros(2))
    assert not ev['success'].any()
    # Separation remains an independent *physical* diagnostic; changing it
    # cannot alter settlement/reward with identical pose/depth/force.
    for k in ('success','collision','outside','terminated'):
        assert ev[k][0]==ev[k][1]
    reward=task.reward(torch.zeros(2),torch.full((2,),.7),obs,torch.zeros(2,6),
                       torch.full((2,),.03),ev)
    assert reward[0]==reward[1]

def test_old_progress_model_cannot_silently_load_new_model():
    from test_pose_cut import model as old_model
    new,_,_=model();old,_,_=old_model()
    with pytest.raises(RuntimeError):
        new.load_state_dict(old.state_dict())

def test_missing_physical_progress_does_not_invent_separation():
    wm,s,g=model('mujoco_warp_blademaster')
    ev=wm.spec.free_task.events(torch.zeros(1,25),torch.zeros(1,6),None,
        torch.ones(1,dtype=torch.long),pose=torch.zeros(1,7),
        board=dict(knife_min_board_gap=torch.ones(1,1)*1e-6,
                   knife_board_contact=torch.ones(1,1,dtype=torch.bool)),
        success_counter=torch.zeros(1))
    assert 'separated' not in ev
