"""Paired cases must not depend on backend RNG consumption or group order."""
import json
from pathlib import Path
import pytest
import torch
from cutting_world_model.shared_scene import configs, fingerprint, load_configs
from cutting_world_model.scene_alignment import initial_clearances, validate_alignment

def contract():
    return json.loads(Path("configs/shared/scene_v1.json").read_text())

def manifest(c):
    return dict(contract_sha256=fingerprint(c), objects={
        o["name"]: dict(scene="fixture", elastic=o["elastic"],
                       blademaster_geometry_asset=o["blademaster_geometry_asset"])
        for o in c["objects"]})

def cases(c):
    d,b=configs(c,manifest(c))
    assert b["start_clearance_m"] == [v for g in d["groups"] for v in g["start_clearance_m"]]
    assert d["task"] == b["task"]
    assert d["force_limits"] == b["force_limits"]
    assert d["force_observation"] == b["force_observation"]
    return {g["name"]:g["start_clearance_m"] for g in d["groups"]}

def test_cases_are_paired_stable_and_bounded():
    c=contract(); first=cases(c)
    torch.manual_seed(711);torch.randn(123)
    assert cases(c)==first
    c["objects"].reverse()
    assert cases(c)==first
    c["common"]["worlds_per_object"]+=2
    assert all(cases(c)[k][:len(v)]==v for k,v in first.items())
    assert all(.005<=v<=.03 for row in first.values() for v in row)
    assert len(set(v for row in first.values() for v in row))>1

def test_seed_changes_case_values_not_task():
    c=contract(); first=cases(c);c["case_seed"]+=1
    assert cases(c)!=first

def test_manifest_mismatch_rejected():
    c=contract();m=manifest(c);c["case_seed"]+=1
    with pytest.raises(ValueError, match="another contract"): configs(c,m)

@pytest.mark.parametrize("bounds",[[0,.03],[.03,.005],[.005,float("nan")],[.005],[".005",.03]])
def test_invalid_clearance_range(bounds):
    with pytest.raises(ValueError):
        validate_alignment(dict(revision="settled_surface_gap_v2",clearance_range_m=bounds))

@pytest.mark.parametrize("values",[None,[.01],[.004,.02],[.01,.031],[.01,float("inf")]])
def test_runtime_clearance_rejects_wrong_count_or_range(values):
    task=contract()["common"]["task"]
    with pytest.raises(ValueError):initial_clearances(task,values,2,"cpu")

def test_explicit_values_are_copied_and_legacy_stays_optional():
    task=contract()["common"]["task"];values=[.005,.03]
    result=initial_clearances(task,values,2,"cpu")
    values[0]=.1
    torch.testing.assert_close(result,torch.tensor([.005,.03],dtype=torch.double))
    assert initial_clearances(None,None,2,"cpu") is None
    with pytest.raises(ValueError): initial_clearances(None,[.01,.01],2,"cpu")

def test_loader_checks_both_assets(tmp_path):
    import hashlib
    c=contract();m=manifest(c)
    for name,item in m["objects"].items():
        d=tmp_path/(name+".pt");d.write_bytes(b"scene")
        b=tmp_path/(name+".npz");b.write_bytes(b"geometry")
        item.update(scene=str(d),scene_sha256=hashlib.sha256(d.read_bytes()).hexdigest(),
                    blademaster_geometry_asset=str(b),
                    blademaster_geometry_sha256=hashlib.sha256(b.read_bytes()).hexdigest())
    for o in c["objects"]:o["blademaster_geometry_asset"]=m["objects"][o["name"]]["blademaster_geometry_asset"]
    m["contract_sha256"]=fingerprint(c)
    cp=tmp_path/"contract.json";mp=tmp_path/"manifest.json"
    cp.write_text(json.dumps(c));mp.write_text(json.dumps(m))
    load_configs(cp,mp)
    b.write_bytes(b"changed")
    with pytest.raises(ValueError,match="BladeMaster geometry changed"):load_configs(cp,mp)
