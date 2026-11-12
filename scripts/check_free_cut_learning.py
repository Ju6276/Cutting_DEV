#!/usr/bin/env python3
"""Bounded real-control collection and one WM/actor/critic update. Not policy training."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'backends/mujoco')]
from cutting_world_model.gpu_env import GPUEnvironment
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.world_model import CuttingWorldModel
from cutting_world_model.policy import SliceItPolicy, SliceItCritic
from cutting_world_model.online_replay import BatchedOnlineReplay
from cutting_world_model.trainer import OnlineLearner
from cutting_world_model.wm_diagnostics import completion_progress_report
from third_party.supermimic.trainer_config import WMTrainerCfg, PolicyTrainerCfg


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=float,default=600.)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    torch.set_num_threads(2);torch.manual_seed(0)
    cfg=json.loads((ROOT/'configs/free_cut/review_v2.json').read_text())
    # A shorter deadline only for this executable-update check; not a training config.
    cfg['task']['horizon_steps']=16
    env=GPUEnvironment(str(ROOT/'backends/mujoco'),2,scene=cfg['scene'],task=cfg['task'],
        force_limits=cfg['force_limits'],force_model='net',friction_velocity=cfg['friction_velocity'])
    env.target_speed=torch.full((2,),cfg['target_speed'],device='cuda')
    frame=env.settle(cfg['settle_seconds'])
    meta=ControlStateSpec.metadata_from_frame(env.metadata,{k:v[0].cpu().numpy() for k,v in frame.items()})
    wm=CuttingWorldModel(meta).cuda()
    spec=wm.spec
    policy=SliceItPolicy(8,25).cuda()
    critic=SliceItCritic(spec.state_dim,8).cuda()
    learner=OnlineLearner(wm,policy,critic,microbatch=4,
        wm_cfg=WMTrainerCfg(window=8,batches_per_update=1,batch_size=4,compile=None),
        policy_cfg=PolicyTrainerCfg(horizon=8,batch_size=4,gamma=1.,critic_epochs=1,
                                   critic_minibatches=1,compile=None),
        wm_only_updates=0,critic_target='real',actor_bc_alpha=.25)
    replay=BatchedOnlineReplay(2,spec.state_dim,'cuda',capacity=64,goal_dim=8)
    state=spec.pack_batch(frame).float()
    goal=torch.cat((frame['target_pose'],env.target_speed[:,None]),-1).float()
    samples=[];step_times=[]
    for i in range(16):
        if time.monotonic()-start > args.wall_seconds: raise TimeoutError('Learning check wall budget')
        # Execute through the actual six-action controller, never reference=True.
        action=torch.zeros(2,6,device='cuda')
        action[1,4]=.1
        torch.cuda.synchronize();t=time.monotonic()
        frame,reward,done,trunc,info=env.step(action)
        torch.cuda.synchronize();step_times.append(time.monotonic()-t)
        nxt=spec.pack_batch_transition(frame,state).float()
        replay.add(state,action,action,goal,reward,nxt,done,trunc)
        samples.append(dict(step=i+1,done=done.cpu().tolist(),success=info['success'].cpu().tolist(),
                            progress=frame['cut_progress'][:,0].cpu().tolist()))
        state=nxt
        if done.any():
            frame=env.reset(done)
            state=spec.reset_batch_history(frame,state,done).float()
    batch=replay.buffer.sample_windows(4,8)
    before=completion_progress_report(wm,batch.states,batch.a_exec,batch.xi,batch.valid_steps)
    wm_metrics=learner.fit_world_model(replay.buffer)
    policy_metrics=learner.improve_policy(replay.buffer)
    after=completion_progress_report(wm,batch.states,batch.a_exec,batch.xi,batch.valid_steps)
    report=dict(purpose='One executable gradient update on actual controller transitions; NOT accuracy evidence',
                config=cfg,transitions=32,collection=samples,step_seconds=step_times,
                world_model=wm_metrics,actor_critic=policy_metrics,
                completion_before=before,completion_after=after,
                elapsed_seconds=time.monotonic()-start)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    torch.save(dict(metadata=meta,states=batch.states.cpu(),actions=batch.a_exec.cpu(),
                    goal=batch.xi.cpu(),valid=batch.valid_steps.cpu()),args.output/'diagnostic_batch.pt')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
