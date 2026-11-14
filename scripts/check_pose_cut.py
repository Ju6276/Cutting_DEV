"""Physical pose-task check, WM/actor/critic updates and checkpoint round-trip."""
import argparse,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'backends/mujoco')]
from cutting_world_model.gpu_env import GPUEnvironment
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.world_model import CuttingWorldModel
from cutting_world_model.policy import SliceItPolicy,SliceItCritic
from cutting_world_model.online_replay import BatchedOnlineReplay
from cutting_world_model.trainer import OnlineLearner
from cutting_world_model.pose_cut import gap_from_pose
from third_party.supermimic.trainer_config import WMTrainerCfg,PolicyTrainerCfg

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--config',type=Path,default=ROOT/'configs/free_cut/pose_cucumber.json')
 p.add_argument('--scene',help='Optional scene override for a like-for-like reference check')
 p.add_argument('--collection-steps',type=int,default=24,help='Real-action transitions before the bounded gradient check')
 p.add_argument('--reference-only',action='store_true',help='Physical reference only; never fit WM on its dummy actions')
 p.add_argument('--wall-seconds',type=float,default=600)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(2);torch.manual_seed(0);started=time.monotonic()
 cfg=json.loads(a.config.read_text())
 group=cfg['groups'][0]
 a.scene=a.scene or group['scene']
 e=GPUEnvironment(str(ROOT/'backends/mujoco'),2,scene=a.scene,task=cfg['task'],
      force_limits=cfg['force_limits'],force_model=group['force_model'],
      friction_velocity=group['friction_velocity'],settle_seconds=group['settle_seconds'],sim_dt=group.get('sim_dt'))
 e.target_speed=torch.full((2,),.03,device='cuda')
 frame=e.current
 meta=ControlStateSpec.metadata_from_frame(e.metadata,{k:v[0].cpu().numpy() for k,v in frame.items()})
 wm=CuttingWorldModel(meta,hidden=(64,64)).cuda();spec=wm.spec
 policy=SliceItPolicy(8,25).cuda();critic=SliceItCritic(spec.state_dim,8).cuda()
 learner=OnlineLearner(wm,policy,critic,microbatch=4,
    wm_cfg=WMTrainerCfg(window=8,batches_per_update=1,batch_size=4,compile=None),
    policy_cfg=PolicyTrainerCfg(horizon=8,batch_size=4,gamma=1.,critic_epochs=1,critic_minibatches=1,compile=None),
    wm_only_updates=0,critic_target='real',actor_bc_alpha=.25)
 replay=BatchedOnlineReplay(2,spec.state_dim,'cuda',capacity=1024,goal_dim=8)
 state=spec.pack_batch(frame).float()
 goal=torch.cat((frame['target_pose'],e.target_speed[:,None]),-1).float()
 rows=[];finished=torch.zeros(2,dtype=torch.bool,device='cuda'); min_gap=1.;proxy_error=0.
 for i in range(cfg['task']['horizon_steps'] if a.reference_only else a.collection_steps):
  if time.monotonic()-started>a.wall_seconds: raise TimeoutError('Pose check budget')
  action=torch.zeros(2,6,device='cuda')
  frame,reward,done,trunc,info=e.step(action,reference=a.reference_only)
  nxt=spec.pack_batch_transition(frame,state).float()
  replay.add(state,action,action,goal,reward,nxt,done,trunc)
  min_gap=min(min_gap,float(frame['knife_min_board_gap'].min()))
  proxy=gap_from_pose(frame['ee_pose'],meta['board_pose_geometry'])
  proxy_error=max(proxy_error,float((proxy-frame['knife_board_gap'][:,0]).abs().max()))
  rows.append(dict(step=i+1,reward=reward.tolist(),done=done.tolist(),success=info['success'].tolist(),
    separated=info['separated'].tolist(),progress=frame['cut_progress'][:,0].tolist(),
    counter=frame['success_counter'][:,0].tolist(),board_gap_m=frame['knife_board_gap'][:,0].tolist(),
    error=(frame['observation'][:,:3]*frame['observation'].new_tensor(e.heads.max_distance)).tolist()))
  with (a.output/'episodes.jsonl').open('a') as stream: stream.write(json.dumps(rows[-1])+'\n')
  if i % 20 == 0: print(json.dumps(rows[-1]),flush=True)
  state=nxt
  if done.any():
   finished |= done
   if finished.all(): break
   frame=e.reset(done);state=spec.reset_batch_history(frame,state,done).float()
 if min_gap < -2e-6: raise AssertionError('Physical board violation')
 if a.reference_only:
  report=dict(scope='Prescribed physical reference; no optimizer updates',scene=a.scene,
      physical=rows,min_gap_m=min_gap,max_rigid_proxy_gap_error_m=proxy_error)
  (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
  return
 batch=replay.buffer.sample_windows(4,8)
 wm_metrics=learner.fit_world_model(replay.buffer)
 policy_metrics=learner.improve_policy(replay.buffer)
 # A newly versioned checkpoint can reload its model and replay exactly.
 saved=dict(metadata=meta,wm=wm.state_dict(),policy=policy.state_dict(),critic=critic.state_dict(),learner=learner.checkpoint())
 torch.save(saved,a.output/'checkpoint.pt')
 torch.save(replay.state_dict(),a.output/'replay.pt')
 loaded=torch.load(a.output/'checkpoint.pt',map_location='cuda',weights_only=False)
 wm2=CuttingWorldModel(loaded['metadata'],hidden=(64,64)).cuda();wm2.load_state_dict(loaded['wm'])
 x=batch.states[0];act=batch.a_exec[0];g=batch.xi
 with torch.no_grad():
  for v,w in zip(wm.step(x,act,g),wm2.step(x,act,g)): torch.testing.assert_close(v,w,rtol=0,atol=0)
 replay2=BatchedOnlineReplay(2,spec.state_dim,'cuda',capacity=1024,goal_dim=8)
 replay2.load_state_dict(torch.load(a.output/'replay.pt',map_location='cuda',weights_only=False))
 policy2=SliceItPolicy(8,25).cuda();critic2=SliceItCritic(spec.state_dim,8).cuda()
 resumed=OnlineLearner(wm2,policy2,critic2,microbatch=4,wm_cfg=learner.wcfg,policy_cfg=learner.pcfg,
     wm_only_updates=0,critic_target='real',actor_bc_alpha=.25)
 resumed.restore(loaded['learner'])
 assert resumed.optimizer_steps == learner.optimizer_steps
 resumed_wm=resumed.fit_world_model(replay2.buffer)
 resumed_policy=resumed.improve_policy(replay2.buffer)
 assert resumed.updates == learner.updates+1
 assert resumed.wm_updates == learner.wm_updates+1

 report=dict(scope='Bounded engineering check; reduced WM width64; no learning-performance claim',
   config=cfg,scene=a.scene,physical=rows,min_gap_m=min_gap,max_rigid_proxy_gap_error_m=proxy_error,
   wm=wm_metrics,actor_critic=policy_metrics,roundtrip=True,resume_wm=resumed_wm,resume_policy=resumed_policy,elapsed_seconds=time.monotonic()-started)
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('physical','config')},indent=2))
if __name__=='__main__':main()
