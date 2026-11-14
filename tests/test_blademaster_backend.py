"""Backend-independent checks for the BladeMaster pose state and wrench interface."""
import numpy as np
import pytest
import torch
from test_pose_cut import config
from test_state_spec import fixture
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.world_model import CuttingWorldModel

def model():
    meta, frame = fixture()
    task = config(); task['version']='blademaster_pose_v1'
    frame.update(cut_progress=np.array([.2]), step_count=np.array([0.]), observation=np.zeros(25))
    meta.update(free_cut_task=task, goal_dim=8, physical_backend='mujoco_warp_blademaster',
                depth_span_m=.2, progress_semantics='maximum_board_normal_blade_depth_v1',
                board_pose_geometry=dict(vertices_in_ee=[[0,0,0]],board_origin_base=[0,0,-.1],
                                         board_normal_base=[0,0,1],stop_gap_m=1e-6))
    meta=ControlStateSpec.metadata_from_frame(meta,frame)
    wm=CuttingWorldModel(meta,hidden=(12,12)).double()
    state=wm.spec.pack(frame,dtype=torch.double)[None]
    goal=torch.tensor([[.06,0,0,0,0,0,1,.03]],dtype=torch.double)
    return wm,state,goal

def test_blademaster_progress_is_analytic_not_learned_fracture():
    wm,state,goal=model(); spec=wm.spec
    assert 'cut_progress' not in spec.delta_slices
    assert 'cut_progress' not in spec.physical_fields
    fields=spec.integrate_physical(state,torch.zeros(1,spec.delta_dim,dtype=state.dtype))
    torch.testing.assert_close(fields['cut_progress'],torch.tensor([[.5]],dtype=state.dtype))
    state[:,spec.layout['cut_progress']]=.8
    fields=spec.integrate_physical(state,torch.zeros(1,spec.delta_dim,dtype=state.dtype))
    assert fields['cut_progress'].item()==pytest.approx(.8)
    nxt,reward,done=spec.derive(state,fields,torch.zeros(1,6,dtype=state.dtype),goal)
    assert torch.isfinite(nxt).all() and torch.isfinite(reward).all()
    assert spec.terminal_features(nxt).shape==(1,14)
    torch.testing.assert_close(wm.completion_labels(torch.stack((state,nxt)),torch.ones(1,1)),torch.ones(1,1))

def test_blademaster_gradient_and_backend_identity():
    wm,state,goal=model()
    action=torch.full((1,6),.1,dtype=state.dtype,requires_grad=True)
    torch.nn.init.normal_(wm.dynamics.mlp.net[-1].weight,std=1e-4)
    nxt,reward,logits=wm.step(state,action,goal)
    gradient=torch.autograd.grad(reward.sum()+logits.sum(),action)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    metadata=dict(wm.spec.metadata,physical_backend='mujoco_warp_disect')
    with pytest.raises(ValueError,match='backend identity'):
        CuttingWorldModel(metadata,hidden=(12,12))

def test_bm_episode_log_does_not_invent_spring_or_separation_data():
    from cutting_world_model.episode_log import episode_row, episode_metrics
    b=torch.tensor([False]);f=torch.tensor([.2]);n=torch.tensor([2])
    info=dict(success=b,collision=b,outside=b,timeout=~b,depth_progress=f)
    row=episode_row(0,0,1,f,n,info,b,f,f)
    assert row['ending']=='time_limit' and row['depth_progress']==pytest.approx(.2)
    assert 'released_springs' not in row and 'separated' not in row
    metrics=episode_metrics('',~b,f,n,info,b,f,f)
    assert metrics['timeout_rate']==1 and 'separated_rate' not in metrics

def test_sensor_wrench_shift_preserves_virtual_work():
    pytest.importorskip('cutting_gpu_backend')
    from cutting_world_model.blademaster_backend import reaction_at_sensor
    f=torch.tensor([[2.,3.,4.]],dtype=torch.double)
    tau=torch.tensor([[.1,.2,.3]],dtype=torch.double)
    base=torch.tensor([.3,-.2,.4,0,0,0,1.],dtype=torch.double)
    sensor=torch.tensor([[.4,.1,.6,0,0,0,1.]],dtype=torch.double)
    w=reaction_at_sensor(f,tau,base,sensor)
    expected=tau+torch.linalg.cross(base[:3]-sensor[:,:3],f)
    torch.testing.assert_close(w[:,:3],f)
    torch.testing.assert_close(w[:,3:],expected)
    omega=torch.tensor([[.2,-.1,.3]],dtype=torch.double);v=torch.tensor([[.4,.2,-.5]],dtype=torch.double)
    material_v=v+torch.linalg.cross(omega,base[:3]-sensor[:,:3])
    torch.testing.assert_close((w[:,:3]*v+w[:,3:]*omega).sum(),(f*material_v+tau*omega).sum())


def test_swept_guard_catches_safe_endpoints_but_mid_arc_below_board():
    pytest.importorskip('cutting_gpu_backend')
    from cutting_world_model.blademaster_backend import swept_joint_fraction
    from cutting_world_model.pose_cut import gap_from_pose
    def fk(q):
        angle=q[:,0]
        out=q.new_zeros((len(q),7))
        out[:,2]=.1-torch.cos(angle)
        out[:,6]=1
        return out
    old=torch.tensor([[-torch.pi]],dtype=torch.double)
    new=torch.tensor([[torch.pi]],dtype=torch.double)
    geo=dict(vertices_in_ee=[[0,0,0]],board_origin_base=[0,0,0],
             board_normal_base=[0,0,1],stop_gap_m=1e-6)
    assert gap_from_pose(fk(old),geo)>0 and gap_from_pose(fk(new),geo)>0
    fraction,hit=swept_joint_fraction(old,new,fk,geo,1.,torch.tensor([True]))
    assert hit.item() and 0<fraction.item()<.5
    path=old+torch.linspace(0,fraction.item(),1001,dtype=torch.double)[:,None]*(new-old)
    assert gap_from_pose(fk(path),geo).min()>=geo['stop_gap_m']

def test_training_rejects_ignored_legacy_reward_override(tmp_path, monkeypatch):
    pytest.importorskip('cutting_gpu_backend')
    from types import SimpleNamespace
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from scripts.train_gpu import run
    args = SimpleNamespace(backend_root=tmp_path, seed=0, force_penalty='linear',
                           force_penalty_scale=1., conditions=None,
                           blademaster_config=tmp_path/'unused.json')
    with pytest.raises(ValueError, match='legacy force-penalty'):
        run(args, tmp_path, {}, 0.)
