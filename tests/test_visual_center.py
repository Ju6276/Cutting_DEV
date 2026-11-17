import copy
import numpy as np
import pytest
import torch
from cutting_world_model.visual_center import CenterNet,center_plan,load_center,REVISION,SCALE_M,POINT_COUNT
from cutting_world_model.visual_center_features import cloud_input,heading
from cutting_world_model.visual_cut import blade_points,observe_depth
from test_visual_cut import example

def sample():
    d,c,cfg=example()
    cfg["tool_orientation_xyzw"]=cfg.pop("prescribed_ee_goal")[3:]
    return d,c,cfg

def test_input_does_not_encode_absolute_xy_and_tracks_observed_origin():
    rng=np.random.default_rng(3);p=rng.normal(size=(800,3))*.01+[.1,.5,.15]
    f,o=cloud_input(p,.126)
    g,q=cloud_input(p+[.013,-.012,0],.126)
    np.testing.assert_allclose(f,g,atol=1e-6)
    np.testing.assert_allclose(q-o,[.013,-.012,0])
    assert f.shape==(POINT_COUNT,3)
    # Normalization retains size: no true mesh center or extent supplied.
    large,_=cloud_input((p-o)*2+o,.126)
    assert np.ptp(large[:,0])==pytest.approx(2*np.ptp(f[:,0]))

def test_point_network_is_permutation_invariant_and_differentiable():
    torch.manual_seed(3);net=CenterNet()
    torch.nn.init.normal_(net.head[-1].weight,std=.1)
    x=torch.randn(2,80,3,requires_grad=True)
    torch.testing.assert_close(net(x),net(x[:,torch.randperm(80)]),atol=1e-6,rtol=1e-5)
    net(x).sum().backward()
    assert x.grad.isfinite().all() and x.grad.abs().sum()>0

@pytest.mark.parametrize("learned",[False,True])
def test_goal_uses_selected_plane_and_preserves_board_and_tool(learned):
    d,c,cfg=sample()
    model=CenterNet().eval() if learned else None
    p=center_plan(d,c,cfg,model=model)
    goal=np.asarray(p["goal_ee_pose"])
    blade=blade_points(goal,cfg["vertices_in_ee"])
    assert blade[:,2].min()==pytest.approx(p["board_offset_m"]+1e-6)
    edge=blade[blade[:,2]<blade[:,2].min()+.0002].mean(0)
    np.testing.assert_allclose(edge[:2],p["selected_plane_point_base"][:2],atol=1e-7)
    assert not p["prescribed_cut_pose_used"]
    assert len(p["policy_goal"])==8
    if learned:
        seen=observe_depth(d,c,cfg)
        assert p["selected_plane_point_base"][1]==pytest.approx(seen["points"][:,1].mean())

def test_learned_path_rejects_fake_goal_and_wrong_heading():
    d,c,cfg=sample();cfg["prescribed_ee_goal"]=[0,0,0,0,0,0,1]
    with pytest.raises(ValueError,match="prescribed"):center_plan(d,c,cfg)
    cfg.pop("prescribed_ee_goal");cfg["tool_orientation_xyzw"]=[0,0,np.sin(.1),np.cos(.1)]
    with pytest.raises(ValueError,match="fixed"):heading(cfg)

def test_quaternion_sign_and_frozen_checkpoint(tmp_path):
    d,c,cfg=sample();expected=center_plan(d,c,cfg)
    cfg["tool_orientation_xyzw"]=[0,0,0,-1];cfg["policy_quaternion_reference_xyzw"]=[0,0,0,1]
    actual=center_plan(d,c,cfg)
    np.testing.assert_allclose(actual["policy_goal"],expected["policy_goal"])
    net=CenterNet();path=tmp_path/"model.pt"
    payload=dict(revision=REVISION,scale_m=SCALE_M,points=POINT_COUNT,model=net.state_dict())
    torch.save(payload,path);loaded,_=load_center(path)
    assert not any(p.requires_grad for p in loaded.parameters())
    payload["scale_m"]=1;torch.save(payload,path)
    with pytest.raises(ValueError,match="contract"):load_center(path)

def test_outside_predicted_plane_is_rejected_not_clipped():
    d,c,cfg=sample()
    class BadCenter(torch.nn.Module):
        def forward(self,x):return torch.ones(len(x))*10
    with pytest.raises(ValueError,match="Invalid predicted"):center_plan(d,c,cfg,model=BadCenter())
