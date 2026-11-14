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
