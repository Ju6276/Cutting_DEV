"""Current V3 has a per-step board boundary, never an absorbing physical latch."""
import copy
import pytest
import torch
from test_pose_depth_v3 import model, put, joints_for, joint_delta, gap
from cutting_world_model.world_model import CuttingWorldModel

def continuing():
    old, state, goal = model()
    meta=copy.deepcopy(old.spec.metadata)
    meta['free_cut_task']['board']['motion'] = 'project_continue_v1'
    meta['free_cut_task']['tool_revision'] = 'disect_knife_rigid_mount_v1'
    wm=CuttingWorldModel(meta, hidden=(12,12)).double()
    wm.load_state_dict(old.state_dict())
    return wm,state,goal

def test_contact_state_can_lift_and_fields_are_predicted():
    wm,s,g=continuing(); sp=wm.spec
    assert not sp.board_latch
    s=put(sp,s,board_contact=[1.],last_wrench=[1.,2.,3.,.1,.2,.3])
    q=joints_for(sp,sp.field(s,'joint_q'),0.,.001)
    fields=sp.integrate_physical(s,joint_delta(sp,s,q))
    torch.testing.assert_close(fields['joint_q'],q)
    assert fields['board_contact'].item()==0
    assert gap(sp,sp.assemble(fields)).item() > .001
    # No model transition mask is enabled, including records starting at contact.

def test_downward_plus_lateral_motion_is_not_permanently_blocked():
    wm,s,g=continuing(); sp=wm.spec
    q=joints_for(sp,sp.field(s,'joint_q'),0.,-.002)
    fields=sp.integrate_physical(s,joint_delta(sp,s,q))
    assert fields['board_contact'].item()==1
    assert gap(sp,sp.assemble(fields)).item() >= sp.board_geometry['stop_gap_m']-1e-9
    s=sp.assemble(fields)
    old_pose=sp.fk(sp.field(s,'joint_q'))
    q=joints_for(sp,sp.field(s,'joint_q'),.001,-.0001)
    out=sp.integrate_physical(s,joint_delta(sp,s,q))
    assert (sp.fk(out['joint_q'])-old_pose)[0,0].abs() > .0005
    assert gap(sp,sp.assemble(out)).item() >= sp.board_geometry['stop_gap_m']-1e-9

def test_new_config_is_explicit_and_old_checkpoint_still_latches():
    from cutting_world_model.free_cut import validate_task
    wm,_,_=continuing()
    validate_task(wm.spec.metadata['free_cut_task'])
    assert model()[0].spec.board_latch
    bad=copy.deepcopy(wm.spec.metadata['free_cut_task'])
    bad['board']['motion']='unknown'
    with pytest.raises(ValueError): validate_task(bad)

def test_all_new_predictions_respect_peak_last_relation():
    wm,s,g=continuing(); sp=wm.spec
    s=put(sp,s,board_contact=[1.])
    fields=sp.integrate_physical(s,torch.zeros(1,sp.delta_dim,dtype=s.dtype))
    fields['last_wrench']=s.new_tensor([[55.,0.,0.,0.,0.,0.]])
    fields['peak_wrench']=torch.zeros_like(fields['last_wrench'])
    out=sp.apply_board_latch(s,fields,torch.tensor([False]),s.new_tensor([.001]))
    assert out['peak_wrench'][0,0]==55
    assert out['last_wrench'][0,0]==55
    assert not out['board_contact'].any()
    fields['peak_wrench'][0,0]=-100.
    out=sp.apply_board_latch(s,fields,torch.tensor([False]),s.new_tensor([.001]))
    assert out['peak_wrench'][0,0]==100, 'Projection must not erase a predicted force violation'


def test_projection_gradient_and_inactive_world():
    from cutting_gpu_backend.tool_mount import project_joint_endpoint, continuing_joint_step
    def fk(q):
        quat=torch.cat((q[...,3:]*0,torch.ones_like(q[...,:1])),-1)
        return torch.cat((q[...,:3],quat),-1)
    geometry=dict(vertices_in_ee=[[0.,0.,0.],[.02,0.,0.]],board_origin_base=[0.,0.,0.],
                  board_normal_base=[0.,0.,1.],stop_gap_m=1e-6)
    q=torch.tensor([[.01,.02,-.001,0.,0.,0.]],dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:project_joint_endpoint(x,fk,geometry)[0],(q,),
                                   eps=1e-7,atol=2e-5,rtol=1e-3)
    old=q.detach().repeat(2,1);old[:,2]=1.2e-6
    new=old.clone();new[:,0]+=.001;new[:,2]-=.001
    out,hit=continuing_joint_step(old,new,fk,geometry,0.,torch.tensor([True,False]))
    assert hit.tolist()==[True,False]
    torch.testing.assert_close(out[1],old[1])
    assert out[0,0]==new[0,0] and out[0,2]>=geometry['stop_gap_m']
    lifted=old.clone();lifted[:,2]+=.002
    out,hit=continuing_joint_step(old,lifted,fk,geometry,1.,torch.ones(2,dtype=torch.bool))
    torch.testing.assert_close(out,lifted)
    assert not hit.any()

def test_contact_transitions_are_fitted_not_masked():
    wm,s,g=continuing();sp=wm.spec
    s=put(sp,s,board_contact=[1.])
    a=torch.zeros(1,6,dtype=s.dtype)
    nxt,_,_=wm.step(s,a,g)
    target=nxt.detach().clone()
    target[:,sp.layout['last_wrench']]+=1
    loss,m=wm.masked_window_loss(torch.stack((s,target)),a[None],g,
                                  torch.zeros(1,1,dtype=s.dtype),torch.ones(1,1,dtype=torch.bool))
    assert m['nse_1step_count']==1
    assert m['dynamics_fit_count']==1
    assert m['dynamics_masked_count']==0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in wm.dynamics.parameters())


def test_cumulative_arrival_counter_and_failure_priority_retained():
    wm,s,g=continuing();sp=wm.spec
    sp.heads.env['successes_threshold']=10
    # Counter 0 cannot succeed simply because pose already equals goal.
    fields=sp.integrate_physical(s,torch.zeros(1,sp.delta_dim,dtype=s.dtype))
    a=torch.zeros(1,6,dtype=s.dtype)
    nxt,reward,done=sp.derive(s,fields,a,g)
    assert sp.field(nxt,'success_counter').item()==1
    assert not done.item()
    s=put(sp,s,success_counter=[10.])
    fields=sp.integrate_physical(s,torch.zeros(1,sp.delta_dim,dtype=s.dtype))
    nxt,reward,done=sp.derive(s,fields,a,g)
    assert done.item() and sp.pose_success(nxt).item()
    fields['peak_wrench']=s.new_full((1,6),1000.)
    nxt,reward,done=sp.derive(s,fields,a,g)
    assert done.item() and not sp.pose_success(nxt).item() and reward.item()<0

def test_new_and_historical_metadata_cannot_be_paired():
    from cutting_world_model.checkpoint import check_checkpoint_pair
    wm,_,_=continuing();old,_,_=model()
    with pytest.raises(ValueError,match='metadata differ'):
        check_checkpoint_pair({'metadata':wm.spec.metadata},{'metadata':old.spec.metadata})


def test_position_command_does_not_invent_joint_velocity():
    from cutting_gpu_backend.tool_mount import project_joint_velocity
    def fk(q):
        return torch.cat((q[...,:3],q[...,3:]*0,torch.ones_like(q[...,:1])),-1)
    geometry=dict(vertices_in_ee=[[0,0,0]],board_origin_base=[0,0,0],
                  board_normal_base=[0,0,1],stop_gap_m=1e-6)
    q=torch.zeros(3,6,dtype=torch.double);q[:,2]=1.2e-6
    v=torch.zeros_like(q);v[:,0]=.1;v[:,2]=torch.tensor([-.02,.02,0.])
    result=project_joint_velocity(q,v,fk,geometry,torch.ones(3,dtype=torch.bool))
    torch.testing.assert_close(result[:,0],v[:,0])
    torch.testing.assert_close(result[:,2],q.new_tensor([0.,.02,0.]),atol=1e-9,rtol=1e-7)


def test_wm_boundary_projection_has_finite_useful_gradients():
    wm,s,g=continuing();sp=wm.spec
    q=joints_for(sp,sp.field(s,'joint_q'),.0003,-.002)
    delta=joint_delta(sp,s,q).requires_grad_(True)
    fields=sp.integrate_physical(s,delta)
    value=sp.fk(fields['joint_q'])[...,0].sum()+fields['last_wrench'].sum()
    grad=torch.autograd.grad(value,delta)[0]
    assert torch.isfinite(grad).all()
    assert grad[:,sp.delta_slices['joint_q']].norm()>0
    assert grad[:,sp.delta_slices['last_wrench']].norm()>0


def test_zero_rotation_second_derivative_for_projection():
    from cutting_world_model.kinematics import quat_exp_second_order
    for x in (0.,1e-5,.1):
        v=torch.full((1,3),x,dtype=torch.double,requires_grad=True)
        assert torch.autograd.gradgradcheck(quat_exp_second_order,(v,),atol=1e-6,rtol=1e-4)


def test_current_diagnostic_replays_through_training_imagination():
    from test_pose_depth_v3 import FixedPolicy
    from cutting_world_model.policy import SliceItCritic
    from cutting_world_model.trainer import OnlineLearner
    from third_party.supermimic.trainer_config import WMTrainerCfg
    from scripts.diagnose_pose_predictions import analyse,event_flags,EVENTS
    wm,s,g=continuing();sp=wm.spec
    s=put(sp,s,board_contact=[1.])
    learner=OnlineLearner(wm,FixedPolicy().double(),SliceItCritic(sp.state_dim,8).double(),
                          actor_horizon=2,wm_cfg=WMTrainerCfg(compile=None))
    a=torch.zeros(1,6,dtype=s.dtype)
    with torch.no_grad():
        nxt,r,_=wm.predict(s,a,g)
        flags,terminal=event_flags(sp,nxt)
        rec=dict(alive=torch.ones(1,1,dtype=torch.bool),state=s[None],next_state=nxt[None],
                 goal=g[None],history_states=s.repeat(1,10,1,1),action=a[None],
                 depth_span=sp.field(s,'depth_span').reshape(1,1),terminated=terminal[None],
                 contact_start=torch.ones(1,1,dtype=torch.bool),
                 contact_next=sp.contact(nxt).reshape(1,1),reward=r[None])
        rec.update({'real_'+e:flags[e][None] for e in EVENTS})
        result=analyse(learner,rec,k=1)
    assert result['latch']['persistent_latch'] is False
    assert result['replay_first_step_max_normalized_gap'] < 1e-6
