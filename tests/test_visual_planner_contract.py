import copy
import pytest
import torch
from cutting_world_model.shared_baselines import validate_current_interface
from cutting_world_model.visual_execution import validate_paired_visual_checkpoint

def metadata():
    return dict(free_cut_task=dict(version="pose_depth_v2"),
        physical_backend="mujoco_warp_disect",controller={"gain":1},
        contact_discretization_revision="edge_coordinate_consistent_v2",
        board_pose_geometry=dict(vertices_in_ee=[[0,0,0],[1,0,0],[0,1,0]],
            board_origin_base=[0,0,.126],board_normal_base=[0,0,1],stop_gap_m=1e-6))

def target(s):
    t=copy.deepcopy(s);t["physical_backend"]="mujoco_warp_blademaster"
    t["contact_discretization_revision"]=None
    t["board_pose_geometry"]["vertices_in_ee"].reverse()
    t["board_pose_geometry"]["board_origin_base"][0]=.01
    return t

def test_transfer_is_explicit_and_only_changes_physics_backend():
    s=metadata();t=target(s)
    with pytest.raises(ValueError):validate_current_interface(s,t)
    validate_current_interface(s,t,True)
    t["controller"]["gain"]=2
    with pytest.raises(ValueError,match="controller"):validate_current_interface(s,t,True)

@pytest.mark.parametrize("kind",["blade","plane","normal","gap","task"])
def test_transfer_rejects_incompatible_task_and_geometry(kind):
    s=metadata();t=target(s);b=t["board_pose_geometry"]
    if kind=="blade":b["vertices_in_ee"][0][0]+=.001
    elif kind=="plane":b["board_origin_base"][2]+=.001
    elif kind=="normal":b["board_normal_base"]=[0,1,0]
    elif kind=="gap":b["stop_gap_m"]*=2
    else:t["free_cut_task"]["gamma"]=.99
    with pytest.raises(ValueError):validate_current_interface(s,t,True)

def test_visual_wm_pair_is_tensor_checked():
    p={"metadata":metadata(),"policy":{"weight":torch.ones(2)}}
    m={"metadata":copy.deepcopy(p["metadata"]),"learner":{"policy":{"weight":torch.ones(2)}}}
    validate_paired_visual_checkpoint(p,m)
    m["learner"]["policy"]["weight"][0]=0
    with pytest.raises(ValueError,match="actor weights"):validate_paired_visual_checkpoint(p,m)
