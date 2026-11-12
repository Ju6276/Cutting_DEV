#!/usr/bin/env python3
"""Bounded real-control/update throughput check; not convergence or a production budget.

Uses a short 16-step deadline to exercise repeated material-specific resets. Full-size
networks, explicit small review minibatches. Also stress-tests every scratch cache size.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'backends/mujoco')]
from cutting_world_model.multi_env import LocalGroup
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.world_model import CuttingWorldModel
from cutting_world_model.policy import SliceItPolicy, SliceItCritic
from cutting_world_model.online_replay import BatchedOnlineReplay
from cutting_world_model.trainer import OnlineLearner
from third_party.supermimic.trainer_config import WMTrainerCfg, PolicyTrainerCfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worlds', type=int, default=256)
    p.add_argument('--iterations', type=int, default=32)
    p.add_argument('--staggered-start', action='store_true', help='Reset stress only: distribute initial remaining deadlines over 16 steps')
    p.add_argument('--full-update-profile', action='store_true', help='Also measure one source-sized WM and actor/real-critic update in eager mode')
    p.add_argument('--wall-seconds', type=float, default=1200)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(2); torch.manual_seed(0)
    cfg = json.loads((ROOT/'configs/free_cut/review_v3.json').read_text())
    cfg['task']['horizon_steps'] = 16  # reset-stress deadline, never a suggested task horizon
    rows, cache = [], []
    report = dict(purpose='Throughput/reset stress with real gradient updates; NOT efficacy',
                  config=cfg, worlds=args.worlds, iterations_requested=args.iterations, replay_capacity_per_world=1000,
                  rows=rows, cache=cache, git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (args.output/'working.patch').write_bytes(subprocess.check_output(['git','diff','HEAD']))
    def save():
        report['elapsed_seconds'] = time.monotonic()-start
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    def budget():
        if time.monotonic()-start >= args.wall_seconds:
            raise TimeoutError('Throughput review wall budget')
    def timed(fn):
        budget(); torch.cuda.synchronize(); t=time.monotonic(); result=fn(); torch.cuda.synchronize()
        return result, time.monotonic()-t
    def memory():
        return dict(allocated_mib=torch.cuda.memory_allocated()/2**20,
                    reserved_mib=torch.cuda.memory_reserved()/2**20,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    device_used_mib=(torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0])/2**20)
    n=args.worlds
    group=LocalGroup(str(ROOT/'backends/mujoco'),cfg['scene'],n,torch.zeros(n),None,cfg['force_limits'],
                     force_model='net',friction_velocity=cfg['friction_velocity'],
                     settle_seconds=cfg['settle_seconds'],task=cfg['task'])
    env=group.env
    def reset(mask):
        count=int(mask.sum())
        # Actual per-episode material randomization; no reused nominal-material settle.
        scales={'youngs_modulus': torch.exp(torch.empty(count).uniform_(-.693147,.693147)),
                'cut_spring_stiffness':torch.exp(torch.empty(count).uniform_(-.693147,.693147)),
                'density':torch.ones(count)}
        return group.reset(mask,scales,torch.full((count,),cfg['target_speed']))
    frame,report['initial_reset_seconds']=timed(lambda:reset(torch.ones(n,dtype=torch.bool,device='cuda')))
    meta=ControlStateSpec.metadata_from_frame(env.metadata,{k:v[0].cpu().numpy() for k,v in frame.items()})
    wm=CuttingWorldModel(meta).cuda(); spec=wm.spec
    policy=SliceItPolicy(8,25).cuda(); critic=SliceItCritic(spec.state_dim,8).cuda()
    wc=WMTrainerCfg(window=8,batches_per_update=1,batch_size=32,compile=None)
    pc=PolicyTrainerCfg(horizon=8,batch_size=32,gamma=1.,critic_epochs=1,critic_minibatches=1,compile=None)
    learner=OnlineLearner(wm,policy,critic,microbatch=16,wm_cfg=wc,policy_cfg=pc,
                          wm_only_updates=2,critic_target='real',actor_bc_alpha=.25)
    learner.progress = lambda **values: budget()
    replay=BatchedOnlineReplay(n,spec.state_dim,'cuda',capacity=1000,goal_dim=8)
    report.update(wm_config=asdict(wc),policy_config=asdict(pc),memory_before_loop=memory())
    if args.staggered_start:
        # Artificial distribution of remaining time for a reset-throughput stress case.
        # This is never exported as an ordinary training or efficacy experiment.
        env.step_count.copy_(torch.arange(n,device='cuda') % cfg['task']['horizon_steps'])
        env.current['step_count'][:,0] = env.step_count
        env.current['observation'][:,-1] = 1-env.step_count/cfg['task']['horizon_steps']
        frame=group._out(env.current)
    report['staggered_start_deadline_stress']=args.staggered_start
    state=spec.pack_batch(frame).float(); history=frame['observation'].float()[None].repeat(10,1,1)
    try:
        for i in range(args.iterations):
            tick=time.monotonic(); budget()
            goal=torch.cat((frame['target_pose'],env.target_speed[:,None]),-1).float()
            with torch.no_grad():
                clean=policy(history,goal)
                action=spec.execute(clean+.1*torch.randn_like(clean))
            result,physics_s=timed(lambda:group.step(action))
            frame,reward,done,trunc,info=result
            nxt=spec.pack_batch_transition(frame,state).float()
            replay.add(state,clean,action,goal,reward,nxt,done,trunc)
            state=nxt; history=torch.cat((history[1:],frame['observation'].float()[None]),0)
            ended=done|trunc; reset_s=0.
            if ended.any():
                frame,reset_s=timed(lambda:reset(ended))
                state=spec.reset_batch_history(frame,state,ended).float()
                history[:,ended]=frame['observation'][ended].float()[None]
            wm_s=policy_s=0.; metrics={}
            if i>=11 and (i+1)%4==0:
                metrics['wm'],wm_s=timed(lambda:learner.fit_world_model(replay.buffer))
                if learner.should_update_policy:
                    metrics['policy'],policy_s=timed(lambda:learner.improve_policy(replay.buffer))
            row=dict(iteration=i+1,physics_seconds=physics_s,reset_seconds=reset_s,
                     wm_seconds=wm_s,policy_seconds=policy_s,iteration_seconds=time.monotonic()-tick,
                     ended=int(ended.sum()),success=int(info['success'].sum()),
                     force=int(info['collision'].sum()),timeout=int(info['timeout'].sum()),
                     metrics=metrics,**memory())
            rows.append(row); save(); print(json.dumps({k:v for k,v in row.items() if k!='metrics'}),flush=True)
        if args.full_update_profile:
            learner.wcfg=WMTrainerCfg(compile=None)
            learner.pcfg=PolicyTrainerCfg(horizon=8,gamma=1.,compile=None)
            learner.microbatch=64;learner.critic_microbatch=64
            wm_full,wm_full_s=timed(lambda:learner.fit_world_model(replay.buffer))
            policy_full,policy_full_s=timed(lambda:learner.improve_policy(replay.buffer))
            report['source_sized_eager_update']=dict(wm_seconds=wm_full_s,policy_seconds=policy_full_s,
                wm_config=asdict(learner.wcfg),policy_config=asdict(learner.pcfg),microbatch=64,
                wm_metrics=wm_full,policy_metrics=policy_full,**memory())
            save()
        # Worst cache-size sequence on the SAME material, separated from training timings.
        env.reset()  # stationary knife above the object for cache-only checks
        m=env.physics.material
        for count in [2**k for k in range((n-1).bit_length()+1) if 2**k<=n]:
            mask=torch.arange(n,device='cuda')<count
            m.restore_export(mask)
            before={k:v.clone() for k,v in m.capture_rows(~mask)[0].items()}
            _,seconds=timed(lambda:m.settle_selected(.002,mask))
            after=m.capture_rows(~mask)[0]
            assert all(torch.equal(v,after[k]) for k,v in before.items())
            cache.append(dict(selected=count,seconds=seconds,cached_sizes=list(m._settle_batches),
                              cached_worlds=sum(m._settle_batches),**memory()))
            save()
        # Full-duration sparse reset after cache pressure: not just .002 s initialization.
        _,report['sparse_reset_seconds']=timed(lambda:reset(torch.arange(n,device='cuda')==0))
        report['final_memory']=memory()
        report['status']='completed'
    except BaseException as exc:
        report['status']='failed';report['error']=repr(exc);save();raise
    finally:
        save()
    print(json.dumps(dict(status=report['status'],elapsed=report['elapsed_seconds'],memory=report['final_memory'])),flush=True)

if __name__=='__main__': main()
