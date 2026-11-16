import copy
import math
import pytest
import torch
from cutting_gpu_backend.controller.fdcc import SlicingActionMapper
from cutting_gpu_backend.controller.pitch import advance_pitch, validate_pitch_control
from cutting_world_model.free_cut import validate_task
from cutting_world_model.world_model import CuttingWorldModel
from test_pose_depth import model

CFG=dict(revision='bounded_pitch_v1',enabled=False,max_angle_deg=5.,max_rate_deg_s=5.)

def test_disabled_mapper_is_exact_legacy_and_reset_clears_offset():
    old=SlicingActionMapper();new=SlicingActionMapper(pitch_control=CFG)
    pose=torch.tensor([[0.,0.,.1,0.,0.,0.,1.]],dtype=torch.double)
    for mapper in (old,new):mapper.reset(1)
    for _ in range(20):
        action=torch.randn(1,6,dtype=torch.double)
        a=old(action,pose,pose);b=new(action,pose,pose)
        for key in a:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    assert new.added_motion_command[0,4]==0

def test_enabled_rate_angle_reverse_and_gradients():
    cfg={**CFG,'enabled':True}
    previous=torch.tensor([0.],dtype=torch.double)
    nxt=advance_pitch(previous,torch.ones_like(previous),cfg,.05)
    assert nxt.item()==pytest.approx(-math.radians(.25))
    for _ in range(100):previous=advance_pitch(previous,torch.ones_like(previous),cfg,.05)
    assert previous.item()==pytest.approx(-math.radians(5))
    reverse=advance_pitch(previous,-torch.ones_like(previous),cfg,.05)
    assert reverse>previous
    action=torch.tensor([.3],dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a:advance_pitch(torch.zeros_like(a),a,cfg,.05),(action,))

@pytest.mark.parametrize('enabled',[False,True])
def test_wm_and_physical_mapper_share_pitch_update(enabled):
    old,s,g=model();meta=copy.deepcopy(old.spec.metadata)
    cfg={**CFG,'enabled':enabled};meta['free_cut_task']['pitch_control']=cfg
    wm=CuttingWorldModel(meta,hidden=(12,12)).double();sp=wm.spec
    wm.load_state_dict(old.state_dict())
    mapper=SlicingActionMapper(pitch_control=cfg);mapper.reset(1)
    for sign in ([1.]*25+[-1.]*45):
        action=torch.tensor([[.3,.2,.1,.4,.2,sign]],dtype=s.dtype)
        oldpose=sp.fk(sp.field(s,'joint_q'));mapper(action,g[:,:7],oldpose)
        fields=sp.integrate_physical(s,torch.zeros(1,sp.delta_dim,dtype=s.dtype))
        nxt,_,_=sp.derive(s,fields,action,g)
        torch.testing.assert_close(sp.field(nxt,'motion_command'),mapper.added_motion_command,atol=1e-15,rtol=0)
        s=nxt

@pytest.mark.parametrize('field,value',[('enabled',1),('max_angle_deg',0),('max_rate_deg_s',float('nan')),('max_angle_deg',90)])
def test_bad_pitch_config_rejected(field,value):
    with pytest.raises(ValueError):validate_pitch_control({**CFG,field:value})

def test_old_tasks_cannot_silently_enable_new_mapping():
    wm,_,_=model();task=copy.deepcopy(wm.spec.metadata['free_cut_task'])
    task['pitch_control']=CFG
    assert validate_task(task)['pitch_control']['enabled'] is False
    task['version']='free_cut_pose_v1'
    with pytest.raises(ValueError):validate_task(task)
