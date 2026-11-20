import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from cutting_world_model.controller_profile import controller_parameters

def test_legacy_controller_profile_keeps_source_parameters(tmp_path):
    p=tmp_path/'runs/controller_source/controller_params.json'
    p.parent.mkdir(parents=True);p.write_text(json.dumps({'sensor':'legacy'}))
    assert controller_parameters({},tmp_path)=={'sensor':'legacy'}

def test_selected_robot_controller_overrides_legacy_file(tmp_path,monkeypatch):
    from cutting_world_model.robot_platform import RobotPlatform
    selected=[]
    def load(root):
        selected.append(root)
        return SimpleNamespace(controller_parameters={'sensor':'ati_sensing_frame'})
    monkeypatch.setattr(RobotPlatform,'load',load)
    assert controller_parameters({'robot_platform':{'asset_root':'new_robot'}},tmp_path)=={'sensor':'ati_sensing_frame'}
    assert selected==['new_robot']

def test_missing_robot_root_cannot_fall_back_to_ur5e(tmp_path):
    with pytest.raises(ValueError,match='asset_root'):
        controller_parameters({'robot_platform':{}},tmp_path)

def test_real_ur10e_analytic_controller_predicts_with_ati_frame():
    import torch
    from cutting_world_model.robot_platform import RobotPlatform
    from cutting_world_model.shared_baselines import AnalyticFDCCModel
    root=Path('/dataHDD/dong/Assets/Cutting_DEV/robots/ur10e_ati_ikea_estimated_v1')
    evidence=Path('/dataHDD/dong/Experiments/Cutting_DEV/runs/ur10e_platform_20261120/blademaster_winding_scene.json')
    if not evidence.exists() or not root.exists():pytest.skip('UR10e integration assets absent')
    meta=json.loads(evidence.read_text())['metadata'];platform=RobotPlatform.load(root)
    model=AnalyticFDCCModel(meta,controller_parameters(meta,root),substeps=1)
    spec=model.spec
    frame={k:torch.zeros((1,*shape),dtype=torch.float32) for k,shape in meta['state_shapes'].items()}
    q=torch.tensor(platform.q,dtype=torch.float32)[None]
    frame['joint_q']=q
    pose=spec.fk(q)
    for k,v in frame.items():
        if 'pose' in k and v.shape==(1,7):frame[k]=pose.clone()
    frame['depth_span'].fill_(.1)
    state=spec.pack_batch(frame).float()
    goal=torch.cat((pose,torch.full((1,1),.03)),dim=-1)
    predicted,reward,done=model.predict(state,torch.zeros(1,6),goal)
    assert torch.isfinite(predicted).all() and torch.isfinite(reward).all()
