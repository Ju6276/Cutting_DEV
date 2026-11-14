"""Intact sampling and multiobject config invariants; no GPU needed."""
import importlib.util
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("bm_prepare",ROOT/"scripts/prepare_blademaster_mesh.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_tetra_sampling_uses_body_volume_not_a_cut_plane():
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=float)
    p=np.array([[.1,.1,.1],[.1,.1,.6],[.4,.4,.4],[-.1,.1,.1]])
    assert mod.contains_tetra(p,v,np.array([[0,1,2,3]])).tolist()==[True,True,False,False]

def test_three_objects_share_blade_task_and_have_only_explicit_parameter_transfers():
    c=json.loads((ROOT/"configs/blademaster/three_intact_nominal.json").read_text())
    assert c['worlds']==sum(g['worlds'] for g in c['groups'])==3
    assert len({g['name'] for g in c['groups']})==3
    assert all(g['geometry_asset'].endswith('.npz') for g in c['groups'])
    allowed={'young_modulus','poisson_ratio','density','blade_half_size'}
    assert all(set(g['material_overrides'])==allowed for g in c['groups'])
    assert len({tuple(g['material_overrides']['blade_half_size']) for g in c['groups']})==1
    assert c['task']['version']=='blademaster_pose_v1'


def test_batched_256_keeps_the_nominal_physics_and_reward():
    original=json.loads((ROOT/"configs/blademaster/three_intact_nominal.json").read_text())
    batch=json.loads((ROOT/"configs/blademaster/three_intact_batched256.json").read_text())
    assert batch["worlds"] == sum(g["worlds"] for g in batch["groups"]) == 256
    assert [g["worlds"] for g in batch["groups"]] == [86,85,85]
    assert batch["execution"] == "batched"
    assert batch["force_observation"] == "sensor_wrench6_v1"
    for key in ("task","force_limits","target_speed_m_s","clearance_m","settle_seconds"):
        assert batch[key] == original[key]
    for old,new in zip(original["groups"],batch["groups"]):
        assert {k:v for k,v in old.items() if k!="worlds"} == {k:v for k,v in new.items() if k!="worlds"}
