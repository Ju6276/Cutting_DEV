"""Bounded full-learner probe on actual multi-object transitions from either backend."""
import argparse,json,sys,time,signal,subprocess,math,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'backends/mujoco')]
from cutting_world_model.multi_env import MultiConditionEnvironment
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.world_model import CuttingWorldModel
from cutting_world_model.policy import SliceItPolicy,SliceItCritic
from cutting_world_model.trainer import OnlineLearner
from cutting_world_model.online_replay import BatchedOnlineReplay
from cutting_world_model.training_snapshot import capture_training_environment

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--config',type=Path,required=True);p.add_argument('--worlds',type=int,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--microbatch',type=int,default=512)
 p.add_argument('--seconds',type=int,default=2400);p.add_argument('--reserve-gib',type=float,default=3)
 p.add_argument('--snapshot',action='store_true')
 p.add_argument('--backend',choices=['disect','blademaster'],default='disect')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 cfg=json.loads(a.config.read_text());counts=[a.worlds//len(cfg['groups'])+(i<a.worlds%len(cfg['groups'])) for i in range(len(cfg['groups']))]
 if any(g.get('start_clearance_m') is not None and len(g['start_clearance_m'])!=n
        for g,n in zip(cfg['groups'],counts)):
  raise ValueError('Generate paired cases at the requested world count before this probe')
 if cfg.get('start_clearance_m') is not None and len(cfg['start_clearance_m'])!=a.worlds:
  raise ValueError('Generate paired cases at the requested world count before this probe')
 for g,n in zip(cfg['groups'],counts):g['worlds']=n
 if a.backend=='blademaster':cfg['worlds']=a.worlds
 (a.output/'config.json').write_text(json.dumps(cfg,indent=2))
 torch.set_num_threads(2);torch.manual_seed(0)
 report=dict(worlds=a.worlds,counts=counts,config=str(a.config),microbatch=a.microbatch,
  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
  source_status=subprocess.check_output(['git','status','--porcelain'],text=True),
  config_sha256=hashlib.sha256((a.output/'config.json').read_bytes()).hexdigest(),
  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 def check_finite(metrics):
  for key,value in metrics.items():
   if isinstance(value,(int,float)) and not math.isfinite(value):
    raise AssertionError('nonfinite metric: '+key)
 def check_parameters(learner):
  for module in (learner.wm,learner.policy,learner.critic):
   assert all(torch.isfinite(p).all() for p in module.parameters())

 start=time.monotonic();env=None
 def timeout(*_):raise TimeoutError('capacity probe budget')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(a.seconds)
 def save(phase):
  report.update(phase=phase,wall_seconds=time.monotonic()-start)
  (a.output/'progress.json').write_text(json.dumps(report,indent=2));print(phase,flush=True)
 def gpu():
  return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
 try:
  save('initializing')
  if a.backend=='blademaster':
   from cutting_world_model.blademaster_backend import BladeMasterEnvironment
   env=BladeMasterEnvironment(ROOT/'backends/mujoco',cfg)
  else:
   env=MultiConditionEnvironment.from_config(ROOT/'backends/mujoco',cfg)
  frame=env.current;n=env.worlds
  metadata=ControlStateSpec.metadata_from_frame(env.metadata,{k:v[0].cpu().numpy() for k,v in frame.items()})
  metadata['goal_dim']=8;metadata['task_conditions']=cfg
  wm=CuttingWorldModel(metadata).cuda();spec=wm.spec
  policy=SliceItPolicy(8,metadata['state_shapes']['observation'][0]).cuda()
  critic=SliceItCritic(spec.state_dim,8).cuda()
  learner=OnlineLearner(wm,policy,critic,a.microbatch,wm_only_updates=0,
                         critic_target='real',critic_microbatch=8192,actor_horizon=8,actor_bc_alpha=.25)
  collection_steps=max(40, learner.pcfg.horizon+2, learner.wcfg.window+2, policy.history+2)
  assert learner.pcfg.horizon==32 and learner.actor_horizon==8
  report.update(critic_window=learner.pcfg.horizon,actor_horizon=learner.actor_horizon,
                collection_steps=collection_steps,wm_batch=learner.wcfg.batch_size,
                policy_batch=learner.pcfg.batch_size)
  replay=BatchedOnlineReplay(n,spec.state_dim,'cuda',capacity=1000,goal_dim=8)
  report.update(observation_dim=policy.observation_dim,state_dim=spec.state_dim,
                gpu_after_allocation=gpu(),initial_seconds=time.monotonic()-start)
  free,total=torch.cuda.mem_get_info()
  report.update(free_after_allocation_gib=free/2**30,total_gib=total/2**30)
  if free < a.reserve_gib*2**30:
   report['status']='insufficient_training_headroom';return
  assert policy.observation_dim==34
  state=spec.pack_batch(frame).float()
  history=frame['observation'].float()[None].repeat(policy.history,1,1)
  durations=[];episodes=0
  save('collecting')
  for i in range(collection_steps):
   t=time.monotonic()
   with torch.no_grad():
    goal=env.goal(frame);clean=policy(history,goal);action=spec.execute(clean+.1*torch.randn_like(clean))
    frame,reward,done,trunc,info=env.step(action)
    next_state=spec.pack_batch_transition(frame,state)
    assert torch.isfinite(next_state).all() and torch.isfinite(reward).all()
    replay.add(state,clean,action,goal,reward,next_state,done,trunc)
    state=next_state;history=torch.cat((history[1:],frame['observation'].float()[None]),0)
    if (done|trunc).any():
     mask=done|trunc;episodes+=int(mask.sum());frame=env.reset(mask)
     state=spec.reset_batch_history(frame,state,mask);history[:,mask]=frame['observation'][mask].float()[None]
   durations.append(time.monotonic()-t)
   if i%5==0:print('collected',i+1,'seconds',durations[-1],flush=True)
  report.update(step_seconds=durations,transitions=collection_steps*n,episodes=episodes,gpu_after_collection=gpu())
  # Prove replay can supply terminal-aware real critic windows before updating.
  candidate=replay.buffer.sample_windows(learner.pcfg.batch_size,learner.pcfg.horizon)
  del candidate
  save('wm_update')
  t=time.monotonic();report['wm']=learner.fit_world_model(replay.buffer);report['wm_seconds']=time.monotonic()-t
  save('actor_critic_update')
  t=time.monotonic();report['policy']=learner.improve_policy(replay.buffer);report['policy_seconds']=time.monotonic()-t
  check_finite(report['wm']);check_finite(report['policy']);check_parameters(learner)
  assert report['policy']['critic_real_samples']==learner.pcfg.batch_size*learner.pcfg.horizon
  report.update(gpu_after_learning=gpu(),torch_peak_mib=torch.cuda.max_memory_allocated()/2**20,
                free_after_learning_gib=torch.cuda.mem_get_info()[0]/2**30)
  save('masked_reset')
  old={k:v.clone() for k,v in env.current.items() if isinstance(v,torch.Tensor)}
  mask=torch.zeros(n,dtype=torch.bool,device='cuda');mask[[sum(counts[:i]) for i in range(len(counts))]]=True
  t=time.monotonic();frame=env.reset(mask);report['reset_seconds']=time.monotonic()-t
  for key,value in old.items():assert torch.equal(value[~mask],frame[key][~mask]),key
  report['masked_reset_unselected_exact']=True
  if a.snapshot:
   save('snapshot')
   before_snapshot={k:v.clone() for k,v in env.current.items() if isinstance(v,torch.Tensor)}
   capture_training_environment(env,ROOT/'backends/mujoco',a.output/'snapshot',
      dict(iteration=collection_steps,training_stage='preflight',wm_updates=learner.wm_updates,policy_updates=learner.updates))
   for key,value in before_snapshot.items():assert torch.equal(value,env.current[key]),key
   report['snapshot_state_unchanged']=True
  save('save_restore')
  torch.save(learner.checkpoint(),a.output/'learner.pt')
  if getattr(env,'condition_sampler',None) is not None:
   torch.save(env.condition_sampler.state_dict(),a.output/'episode_conditions.pt')
   env.condition_sampler.load_state_dict(torch.load(a.output/'episode_conditions.pt',weights_only=False))
   report['episode_sampler_restored']=True
  replay.save(a.output/'replay.pt')
  learner.restore(torch.load(a.output/'learner.pt',map_location='cuda',weights_only=False))
  replay.load_state_dict(torch.load(a.output/'replay.pt',map_location='cpu',weights_only=False))
  report['restored_wm']=learner.fit_world_model(replay.buffer)
  report['restored_policy']=learner.improve_policy(replay.buffer)
  check_finite(report['restored_wm']);check_finite(report['restored_policy']);check_parameters(learner)
  free_after_restore=torch.cuda.mem_get_info()[0]/2**30
  report.update(status=('passed' if min(report['free_after_learning_gib'],free_after_restore)>=a.reserve_gib else 'insufficient_training_headroom'),
                learning_and_restore_finite=True,free_after_restore_gib=free_after_restore,
                gpu_final=gpu(),torch_peak_mib=torch.cuda.max_memory_allocated()/2**20,
                optimizer_steps=learner.optimizer_steps)
 except BaseException as e:
  report.update(status='failed',error=repr(e));raise
 finally:
  signal.alarm(0)
  try:
   if env is not None:env.close()
  except BaseException as cleanup_error:
   report.update(status='failed',cleanup_error=repr(cleanup_error))
  report['wall_seconds']=time.monotonic()-start
  (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
