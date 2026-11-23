import copy
import pytest
from scripts.evaluate_pose_policy import summarize, validate_interfaces

def test_pose_success_does_not_imply_separation():
    row=dict(success=True,collision=False,outside=False,timeout=False,separated=False,
             **{"return":103., "steps":20, "peak_material_force_n":7.,
                "peak_limit_ratio":.2,"material_impulse_ns":2.,"speed_error_mps":.01})
    s=summarize([row],2)
    assert s["success_rate"]==1 and s["separated_rate"]==0
    assert not s["complete"] and s["completed_cases"]==1

def test_native_components_are_not_declared_slice_separation():
    row=dict(success=True,collision=False,outside=False,timeout=False,material_components=2,
             **{"return":103.,"steps":20,"peak_material_force_n":7.,
                "peak_limit_ratio":.2,"material_impulse_ns":2.,"speed_error_mps":.01})
    s=summarize([row],1)
    assert s["complete"] and s["multiple_components_rate"]==1
    assert "separated_rate" not in s
    assert "object_slip_failure_rate" not in s

def test_interface_mismatch_rejected():
    meta=dict(free_cut_task={"version":"free_cut_pose_v1"},environment={"force_limits":[50]*6},
              controller={},policy_observation_revision="sensor_wrench6_v1",
              state_shapes={"observation":[34]})
    actual=copy.deepcopy(meta)
    validate_interfaces(meta,actual,{"task":meta["free_cut_task"]})
    actual["environment"]["force_limits"][0]=100
    with pytest.raises(ValueError,match="environment"):
        validate_interfaces(meta,actual,{"task":meta["free_cut_task"]})


def test_motion_failure_is_reported_in_summary():
    row=dict(success=False,collision=False,outside=False,timeout=False,
             object_slip_failure=True,object_slip_m=.061,
             **{"return":-100.,"steps":12,"peak_material_force_n":3.,
                "peak_limit_ratio":.1,"material_impulse_ns":1.,"speed_error_mps":.01})
    result=summarize([row],1)
    assert result["object_slip_failure_rate"]==1
    assert result["mean_terminal_object_slip_m"]==pytest.approx(.061)


def test_shared_evaluation_resizes_and_reseeds_both_backends():
    from scripts.evaluate_pose_policy import evaluation_config
    from test_shared_scene import contract, manifest
    from cutting_world_model.shared_scene import configs
    from cutting_world_model.episode_conditions import EpisodeConditions
    import numpy as np
    c=contract();d,b=configs(c,manifest(c))
    dc=evaluation_config(d,5,91);bc=evaluation_config(b,5,91,True)
    assert all(g['worlds']==5 for g in dc['groups']) and bc['worlds']==5*len(c['objects'])
    assert bc['start_clearance_m']==[x for g in dc['groups'] for x in g['start_clearance_m']]
    assert dc['episode_conditions']['seed']==bc['episode_conditions']['seed']==91
    ds=EpisodeConditions(dc['episode_conditions'],dc['groups']).sample(np.ones(5*len(c["objects"]),bool))
    bs=EpisodeConditions(bc['episode_conditions'],bc['groups']).sample(np.ones(5*len(c["objects"]),bool))
    for key in ds:np.testing.assert_array_equal(ds[key],bs[key])
    changed=evaluation_config(d,5,92)
    cs=EpisodeConditions(changed['episode_conditions'],changed['groups']).sample(np.ones(5*len(c["objects"]),bool))
    assert not np.array_equal(ds['youngs_modulus'],cs['youngs_modulus'])
    assert dc['groups'][0]['start_clearance_m']!=changed['groups'][0]['start_clearance_m']
    assert d['groups'][0]['worlds']==c['common']['worlds_per_object']
    assert dc['task']==d['task'] and bc['task']==b['task']


def test_physical_mount_mismatch_rejected():
    meta=dict(free_cut_task={},environment={},controller={},
              state_shapes={"observation":[34]},board_pose_geometry={"vertices_in_ee":[[0,0,0]]})
    other=copy.deepcopy(meta);other['board_pose_geometry']['vertices_in_ee'][0][2]=.01
    with pytest.raises(ValueError,match='board_pose_geometry'):
        validate_interfaces(meta,other,{'task':{}})

def test_continuation_validates_original_cases_and_keeps_only_completed(tmp_path):
    import json
    from scripts.evaluate_pose_policy import completed_case_ids
    manifest=dict(checkpoint_sha256="same",config={"task":{}},seed=4,method="wm_mppi",planner={"samples":512})
    cases=[dict(case=0,group="food",condition={"speed":.03}),dict(case=1,group="food",condition={"speed":.04})]
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    (tmp_path/"cases.json").write_text(json.dumps(cases))
    row=dict(case=1,steps=20,ending="success")
    (tmp_path/"episodes.jsonl").write_text(json.dumps(row)+"\n")
    assert completed_case_ids(tmp_path,manifest,cases)==[1]
    changed=copy.deepcopy(cases);changed[1]["condition"]["speed"]=.05
    with pytest.raises(ValueError,match="conditions"):completed_case_ids(tmp_path,manifest,changed)
    wrong=copy.deepcopy(manifest);wrong["planner"]["samples"]=100
    with pytest.raises(ValueError,match="planner"):completed_case_ids(tmp_path,wrong,cases)
    (tmp_path/"episodes.jsonl").write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="Duplicate"):completed_case_ids(tmp_path,manifest,cases)
    row["ending"]="evaluation_incomplete"
    (tmp_path/"episodes.jsonl").write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="terminated"):completed_case_ids(tmp_path,manifest,cases)
