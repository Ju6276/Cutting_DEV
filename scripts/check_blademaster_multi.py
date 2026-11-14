"""Bounded preflight for shared UR5e/BladeMaster intact-mesh training."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/blademaster/three_intact_nominal.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=1800)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path('backends/mujoco').resolve()))
    from cutting_world_model.blademaster_backend import BladeMasterEnvironment
    from cutting_world_model.control_state import ControlStateSpec
    from cutting_world_model.world_model import CuttingWorldModel
    from cutting_world_model.policy import SliceItCritic
    from cutting_world_model.trainer import OnlineLearner
    from cutting_world_model.online_replay import BatchedOnlineReplay
    from third_party.supermimic.dynamics import MLP
    from third_party.supermimic.trainer_config import WMTrainerCfg,PolicyTrainerCfg
    torch.set_num_threads(2);torch.manual_seed(17);started=time.monotonic()
    report=dict(config=json.loads(a.config.read_text()),status='running')
    env=None
    def budget():
        if time.monotonic()-started>a.seconds:raise TimeoutError('Bounded BM multiobject preflight expired')
    def physical(index):
        from blademaster_warp.topology import components
        solver=env.physics.solvers[index];s=solver.state()
        _,sizes=components(solver.node_numpy,solver.weight_numpy,s['sides'])
        return dict(name=env.object_names[index],particles=solver.n,mass_kg=float(solver.mass*solver.n),
            components=sizes.tolist(),damage_max=float(s['damage'].max()),
            damage_saturated_fraction=float((s['damage']>=1).mean()),min_det_F=float(np.linalg.det(s['deformation']).min()),
            min_particle_centre_z_m=float(s['position'][:,2].min()),
            particle_radius_m=float(solver.radius),
            speed_p99_m_s=float(np.percentile(np.linalg.norm(s['velocity'],axis=1),99)))
    try:
        env=BladeMasterEnvironment('backends/mujoco',report['config'])
        report['initial']=[physical(i) for i in range(env.worlds)]
        assert all(len(x['components'])==1 and x['damage_max']==0 for x in report['initial'])
        report['initial_ee_pose']=env.current['ee_pose'].tolist()
        report['target_ee_pose']=env.current['target_pose'].tolist()
        report['depth_span_m']=env.physics.depth_span
        report['worlds']=env.worlds
        n=env.worlds;outcomes={};rows=[];physics_times=[]
        for step in range(env.free_task.horizon):
            budget();tick=time.monotonic()
            frame,reward,done,trunc,info=env.step(torch.zeros((n,6),device=env.device),reference=True)
            physics_times.append(time.monotonic()-tick)
            rows.append(dict(step=step+1,force=info['peak_force'].tolist(),reward=reward.tolist(),
                board_gap=frame['knife_board_gap'][:,0].tolist(),
                force_limit_ratio=info['safety_margin'].tolist(),done=done.tolist()))
            assert torch.isfinite(frame['peak_wrench']).all()
            assert frame['knife_board_gap'].min()>=-2e-6
            for i in done.nonzero().flatten().tolist():
                if i not in outcomes:
                    x=physical(i)
                    x.update(step=step+1,success=bool(info['success'][i]),
                             collision=bool(info['collision'][i]),outside=bool(info['outside'][i]),
                             timeout=bool(info['timeout'][i]))
                    outcomes[i]=x
                    print('reference ending',x,flush=True)
            if len(outcomes)==n:break
            if bool(done.any()):env.reset(done)
            if step%10==0:print('reference step',step+1,'seconds',physics_times[-1],flush=True)
        report['reference_rows']=rows
        report['reference_outcomes']=[outcomes[i] for i in range(n)]
        assert all(max(r['force'][i] for r in rows)>1e-5 for i in range(n))
        assert all(x['min_particle_centre_z_m']>=-x['particle_radius_m'] for x in outcomes.values())
        report['mean_physics_seconds']=float(np.mean(physics_times))
        report['peak_memory_bytes']=torch.cuda.max_memory_allocated()
        # Reset only world 0; the other worlds' physical tensors and observations must not move.
        if n>1:
            old={k:v[1:].clone() for k,v in env.current.items()}
            material=[{k:getattr(s,k).numpy().copy() for k in ('x','v','F','Fp','codes')} for s in env.physics.solvers[1:]]
            mask=torch.zeros(n,dtype=torch.bool,device=env.device);mask[0]=True
            env.reset(mask)
            for k,v in old.items():torch.testing.assert_close(env.current[k][1:],v,atol=0,rtol=0)
            for s,old_s in zip(env.physics.solvers[1:],material):
                for k,v in old_s.items():np.testing.assert_array_equal(getattr(s,k).numpy(),v)
            report['masked_reset_unselected_exact']=True
        frame=env.reset()
        metadata=ControlStateSpec.metadata_from_frame(env.metadata,{k:v[0].cpu().numpy() for k,v in frame.items()})
        class TinyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.net=MLP(25*10+8,(32,32),6,last_gain=.01)
            def forward(self,h,g):return self.net(torch.cat((h.permute(1,0,2).flatten(1),g),-1))
        def learner():
            wm=CuttingWorldModel(metadata,hidden=(32,32)).to(env.device)
            return OnlineLearner(wm,TinyPolicy().to(env.device),SliceItCritic(wm.spec.state_dim,8).to(env.device),
                microbatch=2,wm_only_updates=0,critic_target='real',actor_horizon=8,actor_bc_alpha=.25,
                wm_cfg=WMTrainerCfg(window=3,batch_size=6,batches_per_update=1,compile=None),
                policy_cfg=PolicyTrainerCfg(horizon=8,batch_size=6,gamma=1.,critic_epochs=1,critic_minibatches=2,compile=None))
        l=learner();spec=l.wm.spec
        replay=BatchedOnlineReplay(n,spec.state_dim,env.device,capacity=128,goal_dim=8)
        state=spec.pack_batch(frame).float();history=frame['observation'].float()[None].repeat(10,1,1)
        for step in range(24):
            budget();goal=env.goal(frame)
            with torch.no_grad():
                clean=l.policy(history,goal);action=(clean+.1*torch.randn_like(clean)).clamp(-1,1)
                frame,reward,done,trunc,info=env.step(action)
                nxt=spec.pack_batch_transition(frame,state).float()
                replay.add(state,clean,action,goal,reward,nxt,done,trunc)
                state=nxt;history=torch.cat((history[1:],frame['observation'].float()[None]),0)
                if done.any():
                    frame=env.reset(done);state=spec.reset_batch_history(frame,state,done)
                    history[:,done]=frame['observation'][done].float()[None]
        before={k:[p.detach().clone() for p in m.parameters()] for k,m in [('wm',l.wm),('actor',l.policy),('critic',l.critic)]}
        metrics=[l.fit_world_model(replay.buffer) for _ in range(3)]
        metrics.append(l.improve_policy(replay.buffer))
        for k,m in [('wm',l.wm),('actor',l.policy),('critic',l.critic)]:
            assert any(not torch.equal(x,y) for x,y in zip(before[k],m.parameters()))
            assert all(torch.isfinite(p).all() for p in m.parameters())
        report['gradient_updates']=metrics
        saved=dict(metadata=metadata,learner=l.checkpoint(),replay=replay.state_dict())
        torch.save(saved,a.output/'checkpoint.pt')
        restored=learner();restored.restore(torch.load(a.output/'checkpoint.pt',weights_only=False)['learner'])
        replay2=BatchedOnlineReplay(n,spec.state_dim,env.device,capacity=128,goal_dim=8)
        replay2.load_state_dict(saved['replay'])
        report['restore_optimizer_steps']=restored.optimizer_steps
        report['resumed_fit']=restored.fit_world_model(replay2.buffer)
        report['resumed_policy']=restored.improve_policy(replay2.buffer)
        report['physical_transitions']=24*n
        report['status']='passed'
        (a.output/'task.json').write_text(json.dumps(metadata,indent=2)+'\n')
    except BaseException as exc:
        report['status']='failed';report['error']=repr(exc);raise
    finally:
        if env is not None:env.close()
        report['wall_seconds']=time.monotonic()-started
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PREFLIGHT PASSED',report['wall_seconds'],report['mean_physics_seconds'],flush=True)
if __name__=='__main__':main()
