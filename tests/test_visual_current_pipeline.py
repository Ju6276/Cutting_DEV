"""Current visual execution must select a food explicitly and use its platform."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts.evaluate_visual_cut import select_visual_group, source_actor, observation_pose
from cutting_world_model.camera_sim import nano_camera, validate_sensor_capture

FOODS=["cucumber_003","potato_half","banana","tofu","luncheon"]

@pytest.mark.parametrize("food",FOODS)
def test_visual_group_is_explicit_and_source_is_preserved(food):
    source=dict(worlds=320,groups=[dict(name=n,worlds=64) for n in FOODS],
                start_clearance_m=[.01]*320,episode_conditions={})
    before=copy.deepcopy(source)
    actual=select_visual_group(source,food,12)
    assert actual["worlds"]==1 and actual["groups"]==[dict(name=food,worlds=1)]
    assert actual["start_clearance_m"]==[.02]
    assert actual["episode_conditions"]["seed"]==12
    assert source==before

def test_unknown_or_duplicate_food_is_not_silently_replaced():
    source=dict(groups=[dict(name="tofu")])
    with pytest.raises(ValueError):select_visual_group(source,"banana",1)
    source["groups"]*=2
    with pytest.raises(ValueError):select_visual_group(source,"tofu",1)

def test_current_full_checkpoint_actor_is_used():
    weights={"weight":torch.ones(2)}
    assert source_actor({"learner":{"policy":weights}}) is weights
    assert source_actor({"policy":weights}) is weights

def test_observation_pose_tracks_ur10e_workspace_and_keeps_orientation():
    frame={"ee_pose":torch.tensor([[.6,0.,.45,0.,0.,0.,1.]])}
    box=dict(lower=[.528,-.10,.37],upper=[.728,.10,.543])
    p=observation_pose(frame,box)
    assert (p[:3]>box["lower"]).all() and (p[:3]<box["upper"]).all()
    np.testing.assert_array_equal(p[3:],frame["ee_pose"][0,3:])
    assert p[0]>.5 and p[2]>.5

def test_nano_v4_uses_measured_intrinsics_and_declared_extrinsics():
    root=Path(__file__).resolve().parents[1]
    profile=json.loads((root/"configs/vision/mecheye_nano_v4_550_lab_v1.json").read_text())
    camera=nano_camera(profile,[.65,0.,.37])
    validate_sensor_capture(camera,profile,np.eye(4))
    assert camera.fx==1728.890384 and camera.fy==1728.884610
    assert profile["hardware_version"]=="4.0.0"
    assert "NOT measured" in profile["extrinsics_source"]
    assert np.isclose(np.linalg.norm(camera.base_from_camera[:3,3]-[.65,0.,.37]),.55)
