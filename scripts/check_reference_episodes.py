"""Actual reference episodes through the shared controller, reward and reset paths."""
import argparse, json, signal, time, sys, subprocess, hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'backends/mujoco')]
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.world_model import CuttingWorldModel
from cutting_world_model.pose_depth import depth_progress
from cutting_world_model.training_snapshot import capture_training_environment

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--config',type=Path,required=True)
 p.add_argument('--backend',choices=['disect','blademaster'],required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--seconds',type=int,default=1800)
 p.add_argument('--cycles',type=int,default=1)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 cfg=json.loads(a.config.read_text());env=None;start=time.monotonic();rows=[]
 report=dict(status='running',backend=a.backend,config=cfg,episodes=[],max_reward_error=0.,
  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
  source_status=subprocess.check_output(['git','status','--porcelain'],text=True),
  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reset_calls=0)
 def expired(*_):raise TimeoutError('Reference episode verification budget')
 signal.signal(signal.SIGALRM,expired);signal.alarm(a.seconds)
 torch.set_num_threads(2);torch.manual_seed(0)
 try:
  if a.backend=='disect':
   from cutting_world_model.multi_env import MultiConditionEnvironment
   env=MultiConditionEnvironment.from_config(ROOT/'backends/mujoco',cfg)
  else:
   from cutting_world_model.blademaster_backend import BladeMasterEnvironment
   env=BladeMasterEnvironment(ROOT/'backends/mujoco',cfg)
  frame=env.current;n=env.worlds
  meta=ControlStateSpec.metadata_from_frame(env.metadata,{k:v[0].cpu().numpy() for k,v in frame.items()})
  (a.output/'metadata.json').write_text(json.dumps(meta,indent=2))
  wm=CuttingWorldModel(meta,hidden=(32,32)).to(env.device).double();sp=wm.spec
  state=sp.pack_batch(frame).double()
  capture_training_environment(env,ROOT/'backends/mujoco',a.output/'snapshot_initial',dict(stage='reference_preflight'))
  counts=torch.zeros(n,dtype=torch.int64,device=env.device)
  peak=torch.zeros((n,6),dtype=torch.double,device=env.device)
  returns=torch.zeros(n,dtype=torch.double,device=env.device)
  rows=[]
  for step in range(cfg['task']['horizon_steps']*a.cycles+1):
   action=torch.zeros((n,6),device=env.device,dtype=torch.double);old=frame
   frame,reward,done,trunc,info=env.step(action,reference=True)
   fields={k:frame[k].double() for k in sp.physical_fields}
   fields['depth_span']=sp.field(state,'depth_span')
   fields['cut_progress']=depth_progress(sp.fk(fields['joint_q']),sp.board_geometry,fields['depth_span'],sp.field(state,'cut_progress'))
   fields['board_contact']=frame['board_contact'].double()
   nxt,mr,md=sp.derive(state,fields,action,env.goal(old).double())
   error=float((mr-reward).abs().max());assert error<1e-3
   assert torch.equal(md,done)
   assert all(torch.isfinite(v).all() for v in (reward,frame['joint_q'],frame['peak_wrench']))
   report['max_reward_error']=max(report['max_reward_error'],error)
   returns+=reward;peak=torch.maximum(peak,frame['peak_wrench'].double())
   rows.append(dict(state=state.detach().cpu(),next_state=nxt.detach().cpu(),
     action=action.cpu(),reward=reward.cpu(),terminated=done.cpu(),truncated=trunc.cpu(),
     reference=True,frame={k:v.detach().cpu() for k,v in frame.items() if k in
      ('joint_q','joint_qd','wrench_history','last_wrench','peak_wrench','knife_min_board_gap',
       'knife_board_contact','cut_progress','success_counter','step_count')}))
   if (done|trunc).any():
    mask=done|trunc
    for i in mask.nonzero().flatten().tolist():
     report['episodes'].append(dict(world=i,step=step+1,return_value=float(returns[i]),peak=peak[i].tolist(),
       info={k:v[i].detach().cpu().tolist() for k,v in info.items() if isinstance(v,torch.Tensor) and v.ndim and len(v)==n}))
    counts[mask]+=1;returns[mask]=0;peak[mask]=0
    keep={k:v[~mask].clone() for k,v in frame.items()}
    frame=env.reset(mask);report['reset_calls']+=1
    for k,v in keep.items():assert torch.equal(frame[k][~mask],v),k
    nxt=sp.reset_batch_history(frame,nxt,mask)
    assert bool((frame['step_count'][mask]==0).all())
    assert bool((frame['success_counter'][mask]==0).all())
    if bool((counts>=a.cycles).all()):
     # Exercise a new physical step after the last natural terminal/reset.
     check_frame,check_reward,_,_,_=env.step(action,reference=True)
     assert torch.isfinite(check_reward).all()
     assert bool((check_frame['step_count'][mask]==1).all())
     report['step_after_final_reset']=True
     break
   state=nxt
   if step%10==0:print('step',step+1,'episodes',counts.tolist(),'depth',frame['cut_progress'].flatten().tolist(),flush=True)
  assert bool((counts>=a.cycles).all()),counts
  capture_training_environment(env,ROOT/'backends/mujoco',a.output/'snapshot_final',dict(stage='reference_preflight_end'))
  report.update(status='passed',counts=counts.tolist(),steps=step+1,
    source_commit_at_finish=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
 except BaseException as error:
  report.update(status='failed',error=repr(error));raise
 finally:
  signal.alarm(0)
  torch.save(rows,a.output/'transitions.pt')
  try:
   if env is not None:env.close()
  except BaseException as cleanup_error:
   report.update(status='failed',cleanup_error=repr(cleanup_error))
  report['wall_seconds']=time.monotonic()-start
  (a.output/'report.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
