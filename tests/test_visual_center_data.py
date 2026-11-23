import copy
import numpy as np
import pytest
from cutting_world_model.visual_center_features import cloud_input,table_cloud_input,TABLE_FRAME
from cutting_world_model.visual_center_data import placement_splits,camera_views,validate_dataset
from cutting_world_model.visual_cut import connected_components,DepthCamera

def test_table_cloud_roundtrip_matches_normalization_without_label_information():
    p=np.random.default_rng(3).normal(size=(800,3))*.01+[.6,.04,.42]
    config=dict(board_origin_base=[.6,.02,.35],board_normal_base=[0,0,1])
    f,o,t=table_cloud_input(p,config,.002)
    old,base=cloud_input(p,.352)
    np.testing.assert_allclose(f,old,atol=1e-6)
    np.testing.assert_allclose(o+t,base,atol=1e-12)
    assert (o[1]+.012+t[1])==pytest.approx(base[1]+.012)
    config["board_x_base"]=[0,1,0]
    with pytest.raises(ValueError,match="align"):table_cloud_input(p,config)

def dataset():
    split=placement_splits(12,7)
    cases=[];objects=[];splits=[]
    for obj in range(5):
        for placement in range(12):
            for view in range(3):
                cases.append(dict(case=f"food{obj}_{placement}_view{view}",placement_id=placement,split=str(split[placement])))
                objects.append(obj);splits.append(split[placement])
    n=len(cases)
    data=dict(cloud=np.zeros((n,512,3)),origin=np.zeros((n,3)),label=np.zeros(n),
              center_y=np.zeros(n),geometric_y=np.zeros(n),case=np.array([r["case"] for r in cases]),
              object=np.array(objects),split=np.array(splits))
    manifest=dict(input_frame=TABLE_FRAME,cases=cases,sources=[{"object":f"food{i}"} for i in range(5)],
                  split_counts={s:int(np.sum(data["split"]==s)) for s in ["train","val","test"]})
    return data,manifest

def test_five_food_grouped_views_not_hardcoded_to_three_foods():
    d,m=dataset()
    assert validate_dataset(d,m)==[f"food{i}" for i in range(5)]
    assert sum(m["split_counts"].values())==180

def test_same_placement_view_cannot_leak_into_other_split():
    d,m=dataset()
    new="val" if d["split"][0]=="train" else "train"
    d["split"][0]=new;m["cases"][0]["split"]=new
    with pytest.raises(ValueError,match="leakage"):validate_dataset(d,m)

def test_dataset_rejects_wrong_label_coordinate_and_nonfinite_input():
    d,m=dataset();d["center_y"][0]=.01
    with pytest.raises(ValueError,match="coordinate"):validate_dataset(d,m)
    d,m=dataset();d["cloud"][0,0,0]=np.nan
    with pytest.raises(ValueError,match="Nonfinite"):validate_dataset(d,m)

def test_camera_orbit_preserves_range_and_intrinsics():
    t=np.eye(4);t[:3,3]=[.3,.0,.5]
    c=DepthCamera(width=10,height=10,fx=20,fy=20,cx=5,cy=5,base_from_camera=t)
    views=list(camera_views(c,[0,0,0],[-10,0,10]))
    for view in views:
        assert np.linalg.norm(view.base_from_camera[:3,3])==pytest.approx(np.linalg.norm(t[:3,3]))
        np.testing.assert_allclose(view.base_from_camera[:3,:3].T@view.base_from_camera[:3,:3],np.eye(3),atol=1e-12)
    np.testing.assert_allclose(views[1].base_from_camera,t)

def test_components_are_four_connected():
    labels,count=connected_components(np.array([[1,0,1],[0,1,1],[1,0,0]],bool))
    assert count==3
    assert labels[0,2]==labels[1,1]==labels[1,2]
    assert labels[0,0]!=labels[1,1] and labels[2,0]!=labels[0,0]

def test_table_checkpoint_goal_matches_base_goal_and_unknown_frame_rejected(tmp_path):
    import torch
    from cutting_world_model.visual_center import CenterNet,center_plan,load_center,REVISION,SCALE_M,POINT_COUNT
    from test_visual_center import sample
    d,c,cfg=sample()
    model=CenterNet().eval()
    base=center_plan(d,c,cfg,model=model)
    model.input_frame=TABLE_FRAME
    table=center_plan(d,c,cfg,model=model)
    np.testing.assert_allclose(table["policy_goal"],base["policy_goal"],atol=1e-7)
    path=tmp_path/"center.pt"
    payload=dict(revision=REVISION,scale_m=SCALE_M,points=POINT_COUNT,model=model.state_dict(),input_frame=TABLE_FRAME)
    torch.save(payload,path)
    loaded,_=load_center(path)
    assert loaded.input_frame==TABLE_FRAME
    np.testing.assert_allclose(center_plan(d,c,cfg,model=loaded)["policy_goal"],table["policy_goal"],atol=1e-7)
    payload["input_frame"]="unknown"
    torch.save(payload,path)
    with pytest.raises(ValueError,match="input frame"):load_center(path)
