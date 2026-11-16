import copy
import numpy as np
import pytest
from cutting_world_model.visual_cut import DepthCamera,front_camera,estimate_plan,blade_points

def example(board_shift=0.,object_y=0.):
    camera=front_camera([0,0,0],width=160,height=120,focal_px=180)
    rays=camera.points(np.ones((120,160)))-camera.base_from_camera[:3,3]
    eye=camera.base_from_camera[:3,3]
    depth=(board_shift-eye[2])/rays[:,:,2]
    top_depth=(.06+board_shift-eye[2])/rays[:,:,2]
    top=camera.points(top_depth)
    mask=(np.abs(top[:,:,0])<.025)&(np.abs(top[:,:,1]-object_y)<.02)
    depth[mask]=top_depth[mask]
    vertices=[[x,y,z] for x in [-.075,.075] for y in [-.001,.001] for z in [-.02,.03]]
    config=dict(board_origin_base=[0,0,0],board_normal_base=[0,0,1],
        board_half_extent_m=[.12,.08],prescribed_ee_goal=[0,0,.02,0,0,0,1],
        vertices_in_ee=vertices,stop_gap_m=1e-6,clearance_m=.02,clearance_range_m=[.005,.03],
        target_speed_mps=.03,target_speed_range_mps=[.01,.05],
        workspace=dict(lower=[-.2,-.2,-.01],upper=[.2,.2,.3]))
    return depth,camera,config

def test_camera_roundtrip_and_metric_depth():
    depth,cam,_=example()
    points=cam.points(depth)
    pixels=cam.project(points)
    u,v=np.meshgrid(np.arange(cam.width),np.arange(cam.height))
    np.testing.assert_allclose(pixels,np.stack((u,v),-1),atol=1e-10)
    camera_coords=(points-cam.base_from_camera[:3,3])@cam.base_from_camera[:3,:3]
    np.testing.assert_allclose(camera_coords[:,:,2],depth)

@pytest.mark.parametrize("shift",[0.,.001,-.001])
def test_prescribed_goal_board_mount_and_start(shift):
    depth,cam,cfg=example(shift)
    plan=estimate_plan(depth,cam,cfg)
    assert plan["surface_top_m"]==pytest.approx(.06,abs=1e-6)
    assert plan["board_offset_m"]==pytest.approx(shift,abs=1e-6)
    goal=np.array(plan["goal_ee_pose"]);start=np.array(plan["start_ee_pose"])
    assert blade_points(goal,cfg["vertices_in_ee"])[:,2].min()==pytest.approx(shift+1e-6)
    assert blade_points(start,cfg["vertices_in_ee"])[:,2].min()==pytest.approx(shift+.081)
    np.testing.assert_allclose(goal[[0,1,3,4,5,6]],cfg["prescribed_ee_goal"][0:2]+cfg["prescribed_ee_goal"][3:])
    assert len(plan["policy_goal"])==8 and plan["policy_goal"][-1]==.03

def test_prescribed_plane_cannot_silently_move_to_food():
    d,c,cfg=example(object_y=.05)
    with pytest.raises(ValueError,match="misses"):estimate_plan(d,c,cfg)

def test_missing_depth_and_robot_occlusion_reject():
    d,c,cfg=example()
    with pytest.raises(ValueError,match="Board"):estimate_plan(np.zeros_like(d),c,cfg)
    with pytest.raises(ValueError,match="Board"):estimate_plan(d,c,cfg,np.ones_like(d,dtype=bool))

@pytest.mark.parametrize("key,value",[("clearance_m",.1),("target_speed_mps",.5)])
def test_out_of_distribution_commands_reject(key,value):
    d,c,cfg=example();cfg[key]=value
    with pytest.raises(ValueError,match="range"):estimate_plan(d,c,cfg)

def test_camera_and_workspace_validation():
    d,c,cfg=example()
    bad=c.to_dict();bad["base_from_camera"][0][0]*=2
    with pytest.raises(ValueError):DepthCamera(**bad)
    cfg["workspace"]["upper"][2]=.05
    with pytest.raises(ValueError,match="workspace"):estimate_plan(d,c,cfg)

def test_unknown_depth_units_do_not_create_plan():
    d,c,cfg=example()
    with pytest.raises(ValueError):estimate_plan(d*1000,c,cfg)

def test_height_margin_cannot_exceed_training_clearance():
    d,c,cfg=example();cfg["clearance_m"]=.03
    with pytest.raises(ValueError,match="range"):estimate_plan(d,c,cfg)

def test_policy_goal_rejects_wrong_frame_and_tool():
    from cutting_world_model.visual_cut import validate_policy_plan
    d,c,cfg=example();plan=estimate_plan(d,c,cfg)
    meta=dict(goal_dim=8,state_shapes={"observation":[34]},
        environment={"relative_to_ee":False},free_cut_task={"version":"pose_depth_v2"},
        board_pose_geometry={k:copy.deepcopy(cfg[k]) for k in
          ["vertices_in_ee","board_origin_base","board_normal_base","stop_gap_m"]})
    assert validate_policy_plan(plan,meta,cfg).shape==(8,)
    meta["environment"]["relative_to_ee"]=True
    with pytest.raises(ValueError,match="base"):validate_policy_plan(plan,meta,cfg)
    meta["environment"]["relative_to_ee"]=False
    meta["board_pose_geometry"]["vertices_in_ee"][0][0]+=.01
    with pytest.raises(ValueError,match="geometry"):validate_policy_plan(plan,meta,cfg)
