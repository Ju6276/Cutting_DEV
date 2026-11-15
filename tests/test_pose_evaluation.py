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

def test_interface_mismatch_rejected():
    meta=dict(free_cut_task={"version":"free_cut_pose_v1"},environment={"force_limits":[50]*6},
              controller={},policy_observation_revision="sensor_wrench6_v1",
              state_shapes={"observation":[34]})
    actual=copy.deepcopy(meta)
    validate_interfaces(meta,actual,{"task":meta["free_cut_task"]})
    actual["environment"]["force_limits"][0]=100
    with pytest.raises(ValueError,match="environment"):
        validate_interfaces(meta,actual,{"task":meta["free_cut_task"]})
