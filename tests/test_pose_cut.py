import numpy as np
import torch
import pytest
from cutting_world_model.free_cut import FreeCutTask
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.world_model import CuttingWorldModel
from cutting_world_model.pose_cut import gap_from_pose
from test_task_heads import heads
from test_state_spec import fixture

def config():
    return dict(version='free_cut_pose_v1',horizon_steps=180,gamma=1.,
      initial_progress_fraction=.001,
      workspace=dict(frame='robot_base',point='ee',lower=[-1,-1,-1],upper=[1,1,1]),
      board=dict(height_m=0.,stop_gap_m=1e-6),
      weights=dict(progress=10.,speed=20.,load=1.,jerk=.1,success=100.,failure=100.))

def test_pose_reward_needs_arrival_without_failure_not_spring_release():
    h=heads();h.env['agent_control_dt']=.05
    task=FreeCutTask(config(),h)
    obs=torch.zeros(6,25);obs[1,1]=1.;obs[4,1]=1.
    peak=torch.zeros(6,6);peak[2,0]=100
    pose=torch.zeros(6,7);pose[3,0]=2.
    progress=torch.tensor([0.,1.,1.,1.,1.,0.])
    counter=torch.full((6,),float(h.env['successes_threshold']))
    counter[5]=0.
    board=dict(knife_min_board_gap=torch.ones(6,1)*1e-6,knife_board_contact=torch.ones(6,1,dtype=torch.bool))
    ev=task.events(obs,peak,progress,torch.tensor([1,1,1,1,180,1]),pose=pose,board=board,success_counter=counter)
    assert ev['success'].tolist()==[True,False,False,False,False,False]
    assert ev['timeout'].tolist()==[False,False,False,False,True,False]
    raw=task.step_reward(progress,progress,obs,peak,torch.full((6,),.03))
    total=task.reward(progress,progress,obs,peak,torch.full((6,),.03),ev)
    torch.testing.assert_close(total-raw,torch.tensor([100.,0.,-100.,-100.,0.,0.]))
    board['knife_min_board_gap'][0,0]=-.001
    with pytest.raises(FloatingPointError):
        task.events(obs,peak,progress,torch.ones(6),pose=pose,board=board,success_counter=counter)

def model():
    meta,frame=fixture()
    frame.update(cut_progress=np.array([.2]),step_count=np.array([0.]),observation=np.zeros(25))
    meta.update(free_cut_task=config(),goal_dim=8,
                board_pose_geometry=dict(vertices_in_ee=[[0.,0.,0.]],board_origin_base=[0.,0.,-.1],
                                         board_normal_base=[0.,0.,1.],stop_gap_m=1e-6))
    meta=ControlStateSpec.metadata_from_frame(meta,frame)
    wm=CuttingWorldModel(meta,hidden=(12,12)).double()
    s=wm.spec.pack(frame,dtype=torch.float64)[None]
    goal=torch.tensor([[.06,0.,0.,0.,0.,0.,1.,.03]],dtype=torch.float64)
    return wm,s,goal

def test_model_pose_reward_paid_once_and_terminal_head_cannot_create_success_bonus():
    wm,s,g=model();spec=wm.spec
    s[:,spec.layout['success_counter']]=spec.heads.env['successes_threshold']
    action=torch.zeros(1,6,dtype=s.dtype)
    fields=spec.integrate_physical(s,torch.zeros(1,spec.delta_dim,dtype=s.dtype))
    nxt,r,done=spec.derive(s,fields,action,g)
    assert done.item() and spec.pose_success(nxt).item()
    assert spec.field(nxt,'cut_progress').item()<1
    torch.testing.assert_close(spec.completion_reward(nxt,torch.tensor([100.])),torch.zeros(1))
    assert spec.terminal_features(nxt).shape[-1]==14
    # Learned terminal target is task termination, not all-springs-zero.
    target=torch.stack((s,nxt))
    torch.testing.assert_close(wm.completion_labels(target,torch.ones(1,1)),torch.ones(1,1))
    from cutting_world_model.trainer import OnlineLearner
    from types import SimpleNamespace
    l=object.__new__(OnlineLearner);l.wm=wm;l.policy=lambda hist,goal:action;l.pcfg=SimpleNamespace();l.actor_horizon=3
    ro=l.imagine(s,s[None].repeat(10,1,1),g)
    assert ro.survival[1:].eq(0).all() and ro.rewards[1:].eq(0).all()

def test_pose_world_model_real_gradient_update():
    wm,s,g=model();action=torch.full((1,6),.1,dtype=s.dtype,requires_grad=True)
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight,std=1e-4)
    nxt,reward,logits=wm.step(s,action,g)
    grad=torch.autograd.grad(reward.sum()+logits.sum(),action,retain_graph=True)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    loss,_=wm.window_loss(torch.stack((s,nxt.detach())),action.detach()[None],g,torch.zeros(1,1,dtype=s.dtype))
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in wm.parameters())

def test_finite_blade_geometry_counts_rotated_tip():
    geom=dict(vertices_in_ee=[[0.,0.,0.],[1.,0.,0.]],board_origin_base=[0.,0.,0.],board_normal_base=[0.,1.,0.])
    pose=torch.tensor([[0.,.1,0.,0.,0.,-2**-.5,2**-.5]])
    assert gap_from_pose(pose,geom).item()==pytest.approx(-.9,abs=1e-6)

def test_pose_model_projects_unsafe_endpoint_and_has_no_residual_success_bonus():
    wm,s,g=model()
    spec=wm.spec
    # This fixture is a planar arm. Put the board just below its initial y position.
    spec.board_geometry=dict(vertices_in_ee=[[0.,0.,0.]],board_origin_base=[0.,-.001,0.],
                             board_normal_base=[0.,1.,0.],stop_gap_m=1e-6)
    delta=torch.zeros(1,spec.delta_dim,dtype=s.dtype)
    delta[:,spec.delta_slices['joint_q']]=-10.
    fields=spec.integrate_physical(s,delta)
    gap=gap_from_pose(spec.fk(fields['joint_q']),spec.board_geometry)
    assert gap.item() >= 1e-6-1e-9
    assert fields['joint_qd'].eq(0).all()
    # Full separation at a distant pose still does not settle the task as success.
    s[:,spec.layout['cut_progress']]=1.
    s[:,spec.layout['success_counter']]=0.
    assert not spec.pose_success(s).item()

def test_board_projection_gradient_matches_finite_difference():
    wm,s,g=model();spec=wm.spec
    spec.board_geometry=dict(vertices_in_ee=[[0.,0.,0.]],board_origin_base=[0.,-.001,0.],
                             board_normal_base=[0.,1.,0.],stop_gap_m=1e-6)
    delta=torch.zeros(1,spec.delta_dim,dtype=s.dtype)
    delta[:,spec.delta_slices['joint_q']]=-10.
    delta.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda d: spec.integrate_physical(s,d)['joint_q'],
                                   (delta,),eps=1e-5,atol=2e-5,rtol=2e-3)

def test_historical_separation_report_rejects_pose_task():
    from cutting_world_model.wm_diagnostics import completion_progress_report
    wm,s,g=model()
    with pytest.raises(ValueError,match='do not apply'):
        completion_progress_report(wm,torch.stack((s,s)),torch.zeros(1,1,6),g)
