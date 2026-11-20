import json
import numpy as np
import pytest
from cutting_world_model.render_contract import camera,load_profile,validate_platform,appearance_reference,_fixed_transform

def test_fixed_camera_has_no_object_or_rollout_fit():
    p,_=load_profile();center=np.array([.6,0,.37])
    a=camera(p,center,"close");b=camera(p,center,"close")
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    assert a[2]==p["camera"]["close"]["ortho_scale_m"]

def test_platform_mismatch_is_not_a_visual_fallback():
    with pytest.raises(ValueError,match="platform"):
        validate_platform({"robot_base":"b_bot_base_link"},"ur10e")

def test_missing_appearance_rejected(tmp_path):
    p={"appearance":{"cucumber_003":{"directory":str(tmp_path),"sha256":{"appearance.npz":"missing"}}}}
    with pytest.raises(ValueError,match="appearance"):appearance_reference(p,"cucumber_003")
    with pytest.raises(ValueError,match="approved"):appearance_reference(p,"prism")

def test_fixed_material_transform_does_not_use_pca():
    xml='<robot><link name="world"/><link name="base"/><link name="cutting_board_disect"/><joint name="a" type="fixed"><parent link="world"/><child link="base"/><origin xyz="1 0 0"/></joint><joint name="b" type="fixed"><parent link="world"/><child link="cutting_board_disect"/><origin xyz="1 0 2"/></joint></robot>'
    t=_fixed_transform(xml,"base","cutting_board_disect")
    np.testing.assert_array_equal(t[:3,3],[0,0,2])

def test_recorded_base_inertia_frame_restores_material_coordinates():
    from cutting_world_model.render_contract import material_coordinates
    xml='<robot><link name="base"/><link name="inertia"/><link name="cutting_board_disect"/><joint type="fixed"><parent link="base"/><child link="inertia"/><origin xyz="0 0 .2"/></joint><joint type="fixed"><parent link="base"/><child link="cutting_board_disect"/><origin xyz=".1 0 .3"/></joint></robot>'
    d=dict(geom_body=np.array(["inertia"]),body_pos=np.array([[[1.,2.,3.2]]]),body_mat=np.eye(3).reshape(1,1,9))
    result=material_coordinates(d,dict(robot_xml=xml,robot_base="base"),np.array([[1.1,2.,3.3]]))
    np.testing.assert_allclose(result,0,atol=1e-14)

def test_synthetic_food_colors_are_deterministic_and_not_camera_dependent():
    from cutting_world_model.render_contract import procedural_colors
    x=np.array([[.01,.02,.03],[.04,.02,.07]])
    for name in ("banana","tofu","luncheon_meat"):
        a=procedural_colors(name,x)
        np.testing.assert_array_equal(a,procedural_colors(name,x[::-1])[::-1])
        assert np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
    assert not np.array_equal(procedural_colors("tofu",x),procedural_colors("luncheon_meat",x))

def test_compliance_reparenting_preserves_all_physical_poses():
    from cutting_world_model.compliance_frame import configure
    from cutting_world_model.robot_platform import frame_poses
    xml='<robot name="test"><link name="base"/><link name="knife_mount"/><link name="tcp"/><link name="sensor"/><joint name="m" type="fixed"><parent link="base"/><child link="knife_mount"/><origin xyz=".2 .1 .3" rpy=".2 .3 .4"/></joint><joint name="t" type="fixed"><parent link="knife_mount"/><child link="tcp"/><origin xyz=".1 .02 .03" rpy=".1 .2 .3"/></joint><joint name="s" type="fixed"><parent link="knife_mount"/><child link="sensor"/><origin xyz=".03 .02 .1"/></joint></robot>'
    m=dict(robot_base="base",robot_tip="tcp",joint_names=[])
    before=frame_poses(dict(m,robot_xml=xml),[])
    out,_=configure(xml,m,[])
    after=frame_poses(dict(m,robot_xml=out),[])
    again,_=configure(out,m,[])
    after2=frame_poses(dict(m,robot_xml=again),[])
    for name in before:
        np.testing.assert_allclose(before[name],after[name],atol=1e-14)
        np.testing.assert_allclose(before[name],after2[name],atol=1e-14)


def test_prepared_split_does_not_create_an_exterior_normal_seam():
    from scripts.render_episode_still import exterior_vertex_normals
    x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                [0.,0.,0.],[0.,1.,0.],[-1.,0.,0.]])
    before=x.copy()
    n=exterior_vertex_normals(x,np.array([[0,1,2],[3,4,5]]))
    np.testing.assert_allclose(n,np.tile([0,0,1.],(6,1)))
    np.testing.assert_array_equal(x,before)

def test_old_presentation_export_is_rejected():
    from scripts.render_episode_still import export
    with pytest.raises(ValueError,match="render_scene.py"):export(None)
