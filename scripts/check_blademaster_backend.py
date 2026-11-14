#!/usr/bin/env python3
"""Bounded real UR5e/BM interaction, gradient-update and checkpoint check (no long run)."""
import argparse
import json
import sys
import time
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path('configs/blademaster/banana_ur5e.json'))
    p.add_argument('--backend-root',type=Path,default=Path('backends/mujoco'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=900)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.backend_root.resolve()))
    from cutting_world_model.blademaster_backend import BladeMasterEnvironment
    from cutting_world_model.control_state import ControlStateSpec
    from cutting_world_model.world_model import CuttingWorldModel
    from cutting_world_model.trainer import OnlineLearner
    from cutting_world_model.policy import SliceItCritic
    from cutting_world_model.online_replay import BatchedOnlineReplay
    from third_party.supermimic.dynamics import MLP
    from third_party.supermimic.trainer_config import WMTrainerCfg,PolicyTrainerCfg
    torch.set_num_threads(2);torch.manual_seed(19)
    started=time.monotonic()
    def budget():
        if time.monotonic()-started>a.seconds:raise TimeoutError('BladeMaster validation wall budget')
    cfg=json.loads(a.config.read_text())
    env=None
    report=dict(config=cfg,kind='integration validation; not policy performance',
                timing='shared GPU, not a throughput measurement')
    try:
        env=BladeMasterEnvironment(a.backend_root,cfg)
        origin={k:v.clone() for k,v in env.current.items()}
        rows=[]
        for step in range(100):
            budget()
            f,r,d,t,info=env.step(torch.zeros((env.worlds,6),device=env.device),reference=True)
            rows.append(dict(step=step+1,reward=r.tolist(),done=d.tolist(),
                  gap=f['knife_board_gap'][:,0].tolist(),force=info['peak_force'].tolist(),
                  sensor_wrench=f['peak_wrench'].tolist(),progress=info['depth_progress'].tolist(),
                  success=info['success'].tolist(),collision=info['collision'].tolist()))
            if step%10==0: print('reference',rows[-1],flush=True)
            if d.any():break
        report['reference']=rows
        assert max(max(x['force']) for x in rows)>1e-5,'No actual BM contact force measured'
        assert min(min(x['gap']) for x in rows)>=-2e-6,'Blade crossed board'
        frame=env.reset()
        torch.testing.assert_close(frame['joint_q'],origin['joint_q'],atol=0,rtol=0)
        report['reset_joint_q_exact']=True
        one={k:v[0].cpu().numpy() for k,v in frame.items()}
        metadata=ControlStateSpec.metadata_from_frame(env.metadata,one)
        class TinyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.net=MLP(25*10+8,(32,32),6,last_gain=.01)
            def forward(self,h,g):
                return self.net(torch.cat((h.permute(1,0,2).flatten(1),g),-1))
        def learner():
            wm=CuttingWorldModel(metadata,hidden=(32,32)).to(env.device)
            pi=TinyPolicy().to(env.device)
            critic=SliceItCritic(wm.spec.state_dim,8).to(env.device)
            return OnlineLearner(wm,pi,critic,microbatch=2,wm_only_updates=0,
                critic_target='real',actor_horizon=3,actor_bc_alpha=.25,
                wm_cfg=WMTrainerCfg(window=3,batch_size=4,batches_per_update=1,compile=None),
                policy_cfg=PolicyTrainerCfg(horizon=3,batch_size=4,gamma=1.,critic_epochs=1,
                                              critic_minibatches=2,compile=None))
        l=learner();spec=l.wm.spec
        replay=BatchedOnlineReplay(env.worlds,spec.state_dim,env.device,capacity=128,goal_dim=8)
        state=spec.pack_batch(frame).float();history=frame['observation'].float()[None].repeat(10,1,1)
        mapped=[]
        for step in range(48):
            budget();goal=env.goal(frame)
            with torch.no_grad():
                clean=l.policy(history,goal);executed=(clean+.05*torch.randn_like(clean)).clamp(-1,1)
                nxt,reward,done,truncated,info=env.step(executed)
                new_state=spec.pack_batch_transition(nxt,state).float()
                replay.add(state,clean,executed,goal,reward,new_state,done,truncated)
                mapped.append(dict(step=step+1,reward=reward.tolist(),done=done.tolist(),
                                   peak_force=info['peak_force'].tolist()))
                state=new_state;frame=nxt
                history=torch.cat((history[1:],frame['observation'].float()[None]),0)
                if done.any():
                    frame=env.reset(done);state=spec.reset_batch_history(frame,state,done)
                    history[:,done]=frame['observation'][done].float()[None]
        report['mapped_actions']=mapped
        before={key:[p.detach().clone() for p in module.parameters()]
                for key,module in (('wm',l.wm),('actor',l.policy),('critic',l.critic))}
        metrics=[]
        for _ in range(3):budget();metrics.append(l.fit_world_model(replay.buffer))
        metrics.append(l.improve_policy(replay.buffer))
        report['updates']=metrics
        for key,module in (('wm',l.wm),('actor',l.policy),('critic',l.critic)):
            report[key+'_weights_changed']=any(not torch.equal(x,y) for x,y in zip(before[key],module.parameters()))
            assert report[key+'_weights_changed']
            assert all(torch.isfinite(p).all() for p in module.parameters())
        saved=dict(metadata=metadata,learner=l.checkpoint(),replay=replay.state_dict())
        torch.save(saved,a.output/'checkpoint.pt')
        saved=torch.load(a.output/'checkpoint.pt',weights_only=False)
        restored=learner();restored.restore(saved['learner'])
        replay2=BatchedOnlineReplay(env.worlds,spec.state_dim,env.device,capacity=128,goal_dim=8)
        replay2.load_state_dict(saved['replay'])
        for x,y in zip(l.wm.parameters(),restored.wm.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
        report['restore_optimizer_steps']=restored.optimizer_steps
        report['restored_update']=restored.fit_world_model(replay2.buffer)
        report['state_dim']=spec.state_dim;report['delta_dim']=spec.delta_dim
        report['elapsed_seconds']=time.monotonic()-started
        report['passed']=True
        (a.output/'task.json').write_text(json.dumps(metadata,indent=2)+'\n')
    finally:
        if env is not None:env.close()
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('config','reference','mapped_actions')},indent=2))

if __name__=='__main__':main()
