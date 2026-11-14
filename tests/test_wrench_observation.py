"""Versioned wrench observations preserve pose/action/reward semantics."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from cutting_world_model.task_heads import SliceItHeads, configure_force_observation
from cutting_world_model.policy import SliceItPolicy
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.world_model import CuttingWorldModel
from cutting_world_model.free_cut import FreeCutTask
from cutting_world_model.checkpoint import check_checkpoint_pair
from test_task_heads import heads
from test_state_spec import fixture
from test_pose_cut import config


def pair():
    old=heads()
    meta=configure_force_observation(dict(environment=old.env,controller=old.controller), 'sensor_wrench6_v1')
    return old,SliceItHeads(meta['environment'],meta['controller'])


def test_wrench6_preserves_first_terms_and_reward_axes_with_gradients():
    old,new=pair()
    motion=torch.randn(2,6,dtype=torch.float64)
    wrench=torch.arange(36,dtype=torch.float64).reshape(2,3,6).requires_grad_()
    a=old.observation(motion,motion,motion,motion,wrench)
    b=new.observation(motion,motion,motion,motion,wrench)
    assert a.shape==(2,24) and b.shape==(2,33)
    assert old.dims==new.dims==[0,2,4]
    torch.testing.assert_close(a[:,:15],b[:,:15])
    expected=(wrench/wrench.new_tensor(new.force_limits)).transpose(-1,-2).flatten(-2)
    torch.testing.assert_close(b[:,15:],expected)
    no=torch.zeros(2,dtype=torch.bool)
    ra=old.reward(a,no,no,no);rb=new.reward(b,no,no,no)
    torch.testing.assert_close(ra,rb)
    torch.testing.assert_close(torch.autograd.grad(ra.sum(),wrench,retain_graph=True)[0],
                               torch.autograd.grad(rb.sum(),wrench)[0])
    for index in (1,3,5):
        changed=wrench.detach().clone();changed[...,index]+=1
        changed_obs=new.observation(motion,motion,motion,motion,changed)
        assert not torch.equal(changed_obs,b)
        torch.testing.assert_close(changed_obs[:,:15],b[:,:15])


def test_adding_time_does_not_shift_force_reward_slice_or_task_outcomes():
    old,new=pair();old.env['agent_control_dt']=new.env['agent_control_dt']=.05
    motion=torch.zeros(2,6);wrench=torch.arange(36.).reshape(2,3,6)
    ta,tb=FreeCutTask(config(),old),FreeCutTask(config(),new)
    a=ta.observation(old.observation(motion,motion,motion,motion,wrench),torch.tensor([10,20]))
    b=tb.observation(new.observation(motion,motion,motion,motion,wrench),torch.tensor([10,20]))
    assert a.shape[-1]==25 and b.shape[-1]==34
    torch.testing.assert_close(a[:,-1],b[:,-1])
    torch.testing.assert_close(old.force_norm(a),new.force_norm(b))
    peak=torch.ones(2,6);previous=torch.zeros(2);progress=torch.ones(2)*.1;speed=torch.ones(2)*.03
    torch.testing.assert_close(ta.step_reward(previous,progress,a,peak,speed),
                               tb.step_reward(previous,progress,b,peak,speed))
    no=torch.zeros(2,dtype=torch.bool);counter=torch.ones(2)*10
    for x,y in zip(old.termination(a,counter,no),new.termination(b,counter,no)):
        torch.testing.assert_close(x,y)


def six_model():
    meta,frame=fixture()
    frame.update(cut_progress=np.array([.2]),step_count=np.array([0.]),observation=np.zeros(34))
    meta.update(free_cut_task=config(),goal_dim=8,
                board_pose_geometry=dict(vertices_in_ee=[[0.,0.,0.]],board_origin_base=[0.,0.,-.1],
                                         board_normal_base=[0.,0.,1.],stop_gap_m=1e-6))
    meta=configure_force_observation(meta,'sensor_wrench6_v1')
    return ControlStateSpec.metadata_from_frame(meta,frame),frame


def test_wm_derived_observation_and_policy_use_all_six_axes():
    meta,frame=six_model()
    wm=CuttingWorldModel(meta,hidden=(12,12)).double();spec=wm.spec
    frame['wrench_history'][:]=np.arange(18).reshape(3,6)
    state=spec.pack(frame,dtype=torch.float64)[None]
    goal=torch.tensor([[.06,0.,0.,0.,0.,0.,1.,.03]],dtype=torch.float64)
    next_state,_,_=spec.derive(state,spec.integrate_physical(state,torch.zeros(1,spec.delta_dim,dtype=torch.float64)),
                              torch.zeros(1,6,dtype=torch.float64),goal)
    obs=spec.field(next_state,'observation')
    assert obs.shape==(1,34)
    expected=(spec.field(next_state,'wrench_history')/spec.force_limits).transpose(-1,-2).flatten(-2)
    torch.testing.assert_close(obs[:,15:33],expected)
    policy=SliceItPolicy(8,34).double()
    assert policy.observation_terms==(3,3,3,6,18,1)
    assert policy(obs[None].repeat(10,1,1),goal).shape==(1,6)
    with pytest.raises(ValueError,match='interface'):
        policy(torch.zeros(10,1,25),goal)


def test_old_metadata_unchanged_and_incompatible_checkpoint_rejected():
    meta,frame=fixture();original=copy.deepcopy(meta)
    assert configure_force_observation(meta) is meta
    assert meta==original
    new,_=six_model()
    with pytest.raises(ValueError,match='metadata differ'):
        check_checkpoint_pair({'metadata':meta},{'metadata':new})
    malformed=copy.deepcopy(new);malformed['state_shapes']['observation']=[25]
    with pytest.raises(ValueError,match='observation shape'):
        CuttingWorldModel(malformed,hidden=(4,4))
    malformed=copy.deepcopy(new);malformed.pop('policy_observation_revision')
    with pytest.raises(ValueError,match='versioned'):
        CuttingWorldModel(malformed,hidden=(4,4))
    with pytest.raises(ValueError,match='Unknown'):
        configure_force_observation(meta,'typo')


def test_multigroup_config_forwards_force_revision_before_physics():
    sys.path.append(str(Path(__file__).resolve().parents[1]/'backends/mujoco'))
    from cutting_world_model.multi_env import MultiConditionEnvironment
    class Capture(MultiConditionEnvironment):
        def __init__(self,*args,**kwargs):self.arguments=kwargs
    cfg={'groups':[{'name':'fixture','scene':'unused','worlds':1}],
         'force_limits':[50,50,50,5,5,5],'force_observation':'sensor_wrench6_v1'}
    result=Capture.from_config('unused',cfg)
    assert result.arguments['force_observation']=='sensor_wrench6_v1'
    cfg['force_observation']='typo'
    with pytest.raises(ValueError,match='Unknown force_observation'):
        MultiConditionEnvironment.from_config('unused',cfg)


def test_source_sac_bridge_rejects_new_observation_interface():
    from cutting_world_model.sac_gpu_bridge import observation_size,validate_task
    old,new=pair()
    env={**new.env,'n_actions':6}
    assert observation_size(env)==33
    with pytest.raises(ValueError,match='legacy target-axis'):
        validate_task({'ur3e_gym':{}},{'environment':env})
