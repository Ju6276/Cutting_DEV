import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
pytest.importorskip('stable_baselines3')
from cutting_world_model.sac_vector import CuttingVecEnv,TransitionSAC,observation


class TinyPhysical:
    worlds=2
    device=torch.device('cpu')
    def __init__(self):
        self.current={'observation':torch.zeros(2,34)}
        self.count=torch.zeros(2,dtype=torch.long)
        self.closed=0
    def goal(self,frame):return torch.zeros(2,8)
    def reset(self,mask):
        self.count[mask]=0
        self.current['observation'][mask]=0
        return self.current
    def step(self,action):
        self.count+=1
        self.current['observation'][:,0]=self.count
        done=self.count>=torch.tensor([2,3])
        return self.current,torch.ones(2),done,torch.zeros(2,dtype=torch.bool),{'timeout':done}
    def close(self):self.closed+=1


def test_vector_terminal_observation_before_masked_reset_and_no_timeout_bootstrap():
    env=TinyPhysical();vec=CuttingVecEnv(env,history=2)
    assert vec.reset().shape==(2,76)
    vec.step(np.zeros((2,6)))
    obs,reward,done,info=vec.step(np.zeros((2,6)))
    assert done.tolist()==[True,False]
    assert info[0]['terminal_observation'][34]==2
    assert obs[0,34]==0 and obs[1,34]==2
    assert not info[0]['TimeLimit.truncated']
    assert info[0]['episode']['l']==2
    obs,reward,done,info=vec.step(np.zeros((2,6)))
    assert done.tolist()==[False,True] and obs[0,34]==1
    vec.close();vec.close();assert env.closed==1


def test_sac_updates_and_resume_preserve_physical_transition_credit(tmp_path):
    torch.set_num_threads(1)
    env=CuttingVecEnv(TinyPhysical(),history=2)
    agent=TransitionSAC('MlpPolicy',env,device='cpu',learning_starts=2,
        buffer_size=100,batch_size=2,train_freq=1,transitions_per_gradient=3,
        policy_kwargs={'net_arch':[8,8]},seed=0)
    agent.learn(10)
    assert agent.num_timesteps==10 and agent._n_updates==2
    assert agent.transition_credit==2
    # Replay stores final observation, not reset observation, for terminal rows.
    rb=agent.replay_buffer
    assert rb.dones[:rb.pos].any()
    mask=rb.dones[:rb.pos].astype(bool)
    assert (rb.next_observations[:rb.pos][mask,34]>0).all()
    assert not rb.timeouts[:rb.pos].any()
    agent.save(tmp_path/'model');agent.save_replay_buffer(tmp_path/'replay')
    restored=TransitionSAC.load(tmp_path/'model',env=env,device='cpu')
    restored.load_replay_buffer(tmp_path/'replay')
    assert restored.transition_credit==2
    restored.learn(2,reset_num_timesteps=False)
    assert restored.num_timesteps==12 and restored._n_updates==3
    assert all(torch.isfinite(x).all() for x in restored.policy.parameters())
    env.close()


def test_current_mppi_independent_world_plans_and_gamma_guard(tmp_path,monkeypatch):
    pytest.importorskip('pytorch_mppi')
    from test_pose_depth import model
    from cutting_world_model.world_model import CuttingWorldModel
    from cutting_world_model.shared_baselines import SharedMPPIPolicy
    from cutting_world_model.mpc import MPPIConfig
    wm,state,goal=model();wm=wm.float();state=state.float()
    sp=wm.spec
    frame={name:sp.field(state,name).repeat(2,*([1]*(sp.field(state,name).ndim-1)))
           for name in sp.base_fields}
    frame['success_counter']=frame['success_counter'][:,None]
    frame['target_pose']=goal[:,:7].float().repeat(2,1)
    frame['target_speed']=goal[:,7:].float().repeat(2,1)
    env=SimpleNamespace(worlds=2,device=torch.device('cpu'),current=frame,metadata=sp.metadata)
    path=tmp_path/'checkpoint.pt'
    from cutting_world_model.policy import SliceItCritic
    critic=SliceItCritic(sp.state_dim,sp.goal_dim)
    torch.save({'metadata':sp.metadata,'learner':{'critic':critic.state_dict(),'world_model':wm.state_dict(),
        'policy':{},'updates':0,'wm_updates':1}},path)
    monkeypatch.setattr('cutting_world_model.shared_baselines.CuttingWorldModel',
                        lambda m:CuttingWorldModel(m,hidden=(12,12)))
    with pytest.raises(ValueError,match='discount'):
        SharedMPPIPolicy(env,MPPIConfig(discount=.99),path)
    planner=SharedMPPIPolicy(env,MPPIConfig(horizon=2,samples=8,discount=1.),path)
    act=planner.act(frame,torch.tensor([True,False]))
    assert act.shape==(2,6) and torch.isfinite(act).all()
    assert not act[1].any() and planner.steps==[1,0]
    planner.observe(frame)
    planner.reset(frame,torch.tensor([True,False]))
    assert planner.steps==[0,0] and planner.unchanged()
    with_critic=SharedMPPIPolicy(env,MPPIConfig(horizon=2,samples=8,discount=1.,terminal_value='critic'),path)
    assert torch.isfinite(with_critic.act(frame)).all() and with_critic.unchanged()
    with torch.no_grad():next(with_critic.critic.parameters()).add_(1.)
    assert not with_critic.unchanged()
    bad=copy.deepcopy(sp.metadata);bad['tool_revision']='wrong'
    from cutting_world_model.shared_baselines import validate_current_interface
    with pytest.raises(ValueError,match='tool_revision'):validate_current_interface(bad,sp.metadata)

def test_analytic_model_uses_measured_wrench_without_material_or_fitted_weights():
    from test_pose_depth import model
    from cutting_world_model.shared_baselines import AnalyticFDCCModel
    from cutting_world_model.pose_cut import gap_from_pose
    wm,state,goal=model()
    meta=copy.deepcopy(wm.spec.metadata)
    meta['robot_xml']=meta['robot_xml'].replace('</joint>','<limit lower="-3" upper="3" velocity="2"/></joint>')
    labels=['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
    selectors=['sel_x','sel_y','sel_z','sel_ax','sel_ay','sel_az']
    params={'compliance_ref_link':'l6','ft_sensor_ref_link':'l6',
        'stiffness':dict(zip(labels,[1000]*3+[40]*3)),
        'solver':{'iterations':1,'forward_dynamics':{'link_mass':.1},'error_scale':.015},
        'pd_gains':{k:{'p':.1,'d':0} for k in labels},
        'force':{'hand_frame_control':False}}
    params['stiffness'].update(dict(zip(selectors,[1,1,.5,1,.5,1])))
    params['stiffness']['use_parallel_force_position_control']=True
    model=AnalyticFDCCModel(meta,params,substeps=2)
    state=state.float()
    model.spec.field(state,'wrench_history')[:,-1,0]=-2.
    original=state.clone()
    predicted,reward,logits=model.predict(state,torch.zeros(1,6),goal.float())
    assert torch.isfinite(predicted).all() and torch.isfinite(reward).all()
    assert not torch.isnan(logits).any()
    torch.testing.assert_close(state,original)
    assert model.spec.field(predicted,'peak_wrench')[0,0]==2
    assert (model.spec.field(predicted,'wrench_history')[0,:,0]==-2).all()
    assert not any('particle' in name or 'spring' in name for name in model.spec.base_fields)
    assert (gap_from_pose(model.spec.fk(model.spec.field(predicted,'joint_q')),model.spec.board_geometry)>=0).all()


def test_sac_wall_budget_stops_only_after_inserted_transitions_and_updates(monkeypatch, tmp_path):
    from cutting_world_model.sac_vector import SACWallBudgetReached
    env=CuttingVecEnv(TinyPhysical(),history=2)
    agent=TransitionSAC('MlpPolicy',env,device='cpu',learning_starts=0,
        buffer_size=100,batch_size=2,train_freq=1,transitions_per_gradient=1,
        policy_kwargs={'net_arch':[8,8]},seed=0)
    with monkeypatch.context() as patch:
        patch.setattr(agent,'_wall_budget_reached',lambda:agent.num_timesteps>=4)
        with pytest.raises(SACWallBudgetReached):
            agent.learn(20)
    assert agent.num_timesteps==4 and agent._n_updates==4
    assert agent.replay_buffer.size()*env.num_envs==4 and agent.transition_credit==0
    agent.wall_deadline=0.
    agent.save(tmp_path/'budget_model')
    restored=TransitionSAC.load(tmp_path/'budget_model',env=env,device='cpu')
    assert restored.wall_deadline is None
    env.close()


def test_checkpoint_rejects_mixed_payload_generations(tmp_path):
    from cutting_world_model.baseline_checkpoint import save_sac_checkpoint,read_sac_metadata
    env=CuttingVecEnv(TinyPhysical(),history=2)
    model=TransitionSAC('MlpPolicy',env,device='cpu',learning_starts=0,
        buffer_size=100,batch_size=2,train_freq=1,policy_kwargs={'net_arch':[8,8]})
    model.learn(4)
    save_sac_checkpoint(model,tmp_path,{'num_timesteps':4})
    assert read_sac_metadata(tmp_path,True)['num_timesteps']==4
    restored=TransitionSAC.load(tmp_path/'model.zip',env=env,device='cpu')
    restored.load_replay_buffer(tmp_path/'replay.pkl')
    assert restored.num_timesteps==4 and restored.replay_buffer.size()==2
    with (tmp_path/'replay.pkl').open('ab') as stream:stream.write(b'wrong generation')
    with pytest.raises(ValueError,match='replay.pkl'):
        read_sac_metadata(tmp_path,True)
    env.close()
