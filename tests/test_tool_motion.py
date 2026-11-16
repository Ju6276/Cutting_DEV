import copy
import numpy as np
import pytest
import torch
from test_pose_depth import model as depth_model
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.world_model import CuttingWorldModel


def model(backend, enabled):
    old,s,g=depth_model(backend)
    meta=copy.deepcopy(old.spec.metadata)
    meta['free_cut_task']['completion_position']=dict(revision='robot_base_xy_goal_v1',radius_m=.04)
    meta['free_cut_task']['tool_motion']=dict(revision='robot_base_xy_goal_v1',enabled=enabled,
                                            free_m=.025,failure_m=.06,weight=20.)
    frame={k:old.spec.field(s,k)[0].numpy() for k in old.spec.base_fields}
    frame['completion_xy_error']=np.array([0.])
    meta=ControlStateSpec.metadata_from_frame(meta,frame)
    wm=CuttingWorldModel(meta,hidden=(12,12)).double()
    return wm,wm.spec.pack(frame,dtype=torch.double)[None],g


@pytest.mark.parametrize('backend',['mujoco_warp_disect','mujoco_warp_blademaster'])
@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('distance',[0.,.025,.035,.039,.04,.05,.06,.061])
def test_completion_only_and_path_modes_share_physical_and_wm_rules(backend,enabled,distance):
    wm,state,goal=model(backend,enabled);sp=wm.spec
    goal[:,1]=distance
    state[:,sp.layout['success_counter']]=sp.heads.env['successes_threshold']
    fields=sp.integrate_physical(state,torch.zeros(1,sp.delta_dim,dtype=state.dtype))
    action=torch.zeros(1,6,dtype=state.dtype)
    nxt,reward,done=sp.derive(state,fields,action,goal)
    ev=sp.free_task.events(sp.field(nxt,'observation'),fields['peak_wrench'],None,
        sp.field(nxt,'step_count')[...,0],pose=sp.fk(fields['joint_q']),target_pose=goal[:,:7],
        board=dict(knife_min_board_gap=torch.ones(1,1)*1e-6,knife_board_contact=torch.ones(1,1,dtype=torch.bool)),
        success_counter=sp.field(state,'success_counter'))
    physical=sp.free_task.reward(sp.field(state,'cut_progress')[...,0],fields['cut_progress'][...,0],
                                sp.field(nxt,'observation'),fields['peak_wrench'],goal[:,-1],ev)
    torch.testing.assert_close(reward,physical)
    torch.testing.assert_close(done,ev['terminated'])
    assert ev['success'].item()==(distance<.04)
    assert ev['tool_motion_failure'].item()==(enabled and distance>.06)
    assert done.item()==((distance<.04) or (enabled and distance>.06))
    assert sp.pose_success(nxt).item()==(distance<.04)
    assert sp.field(nxt,'success_counter').item()==10+int(distance<.04)
    assert sp.field(nxt,'completion_xy_error').item()==pytest.approx(distance)
    assert 'completion_xy_error' not in sp.delta_slices
    assert 'object_slip' not in sp.layout
    assert sp.terminal_features(nxt).shape[-1]==15


def test_process_cost_is_optional_and_position_gradient_is_analytic():
    for enabled in (False,True):
        wm,s,g=model('mujoco_warp_disect',enabled);sp=wm.spec
        obs=torch.zeros(1,25,dtype=s.dtype);peak=torch.zeros(1,6,dtype=s.dtype)
        pose=torch.tensor([[0.,.04,0,0,0,0,1]],dtype=s.dtype,requires_grad=True)
        target=torch.tensor([[0.,0.,0,0,0,0,1]],dtype=s.dtype)
        distance=sp.free_task.completion_distance(pose,target)
        zero=sp.free_task.step_reward(torch.zeros(1),torch.zeros(1),obs,peak,g[:,-1],tool_distance=torch.zeros(1))
        moved=sp.free_task.step_reward(torch.zeros(1),torch.zeros(1),obs,peak,g[:,-1],tool_distance=distance)
        expected=-(20/180)*((.04-.025)/(.06-.025))**2 if enabled else 0.
        assert (moved-zero).item()==pytest.approx(expected)
        if enabled:assert torch.autograd.grad(moved.sum(),pose)[0][0,1]<0


def test_radial_final_criterion_covers_both_horizontal_axes():
    wm,_,_=model('mujoco_warp_disect',False)
    target=torch.zeros(1,7)
    pose=target.clone();pose[0,:2]=.03
    assert not wm.spec.free_task.arrival_mask(pose,target).item()


def test_tool_failure_logged_separately():
    from cutting_world_model.episode_log import ending
    assert ending(False,False,False,False,False,False,True)=='tool_motion'


@pytest.mark.parametrize('enabled',[False,True])
def test_new_layout_supports_world_model_fit_bptt_and_restore(enabled):
    wm,s,g=model('mujoco_warp_blademaster',enabled);sp=wm.spec
    g[:,1]=.035
    action=torch.full((1,6),.1,dtype=s.dtype,requires_grad=True)
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight,std=1e-4)
    nxt,reward,logits=wm.step(s,action,g)
    grad=torch.autograd.grad(reward.sum()+logits.sum(),action,retain_graph=True)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    loss,_=wm.window_loss(torch.stack((s,nxt.detach())),action.detach()[None],g,torch.zeros(1,1))
    opt=torch.optim.Adam(wm.parameters(),lr=1e-4);opt.zero_grad();loss.backward();opt.step()
    assert all(torch.isfinite(p).all() for p in wm.parameters())
    restored,_,_=model('mujoco_warp_blademaster',enabled);restored.load_state_dict(wm.state_dict())
    for a,b in zip(wm.step(s,action,g),restored.step(s,action,g)):torch.testing.assert_close(a,b)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_boundary_roundoff_after_world_frame_translation(dtype):
    from cutting_world_model.motion_limits import inside,outside
    pose=torch.tensor([[.138,.505,0,0,0,0,1]],dtype=dtype)
    for d in (.0399,.04,.0599,.06,.0601):
        target=pose.clone();target[:,1]+=d
        distance=torch.linalg.vector_norm(pose[:,:2]-target[:,:2],dim=-1)
        assert inside(distance,.04).item()==(d<.04)
        assert outside(distance,.06).item()==(d>.06)
