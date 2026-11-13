#!/usr/bin/env python3
"""Frozen v3 policy transfer to board-stop v4, plus same-condition legacy audit.
No networks are updated; old critic/WM scores are deliberately not compared.
"""
import argparse, copy, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'backends/mujoco')]
from cutting_world_model.multi_env import MultiConditionEnvironment
from cutting_world_model.policy import SliceItPolicy
from cutting_world_model.provenance import git_revision
from cutting_gpu_backend.knife_board import KnifeBoard

BOARD=dict(height_m=0.,stop_gap_m=1e-6,coverage_tolerance_m=.0005)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--conditions',type=Path,default=ROOT/'configs/free_cut/eval_cucumber.json')
 p.add_argument('--mode',choices=['legacy','board'],required=True)
 p.add_argument('--worlds',type=int)
 p.add_argument('--max-seconds',type=float,default=1800)
 p.add_argument('--output',type=Path,required=True)
 args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
 start=time.monotonic(); torch.set_num_threads(2)
 saved=torch.load(args.checkpoint,map_location='cuda',weights_only=False); meta=saved['metadata']
 cfg=json.loads(args.conditions.read_text())
 if cfg['task']!=meta['free_cut_task']: raise ValueError('Original evaluation task must match original checkpoint')
 if len(cfg['groups'])!=1: raise ValueError('One scene per paired run')
 cfg['parallel']=False
 if args.worlds: cfg['groups'][0]['worlds']=args.worlds
 old_task=copy.deepcopy(cfg['task'])
 if args.mode=='board':
  cfg['task'].update(version='free_cut_v4',board=BOARD)
 policy=SliceItPolicy(int(meta['goal_dim']),int(meta['state_shapes']['observation'][0])).cuda()
 policy.load_state_dict(saved['learner']['policy']); policy.eval()
 manifest={'git':git_revision(ROOT),'command':sys.argv,'checkpoint':str(args.checkpoint),
           'checkpoint_sha256':sha(args.checkpoint),'original_task':old_task,'evaluated_config':cfg,
           'mode':args.mode,'policy':'Frozen deterministic v3 policy; no WM/critic update or reuse for scoring',
           'budget_seconds':args.max_seconds,
           'sources':{str(p.relative_to(ROOT)):sha(p) for p in [
             ROOT/'scripts/evaluate_board_cut.py', ROOT/'backends/mujoco/cutting_gpu_backend/knife_board.py',
             ROOT/'backends/mujoco/cutting_gpu_backend/environment.py',ROOT/'cutting_world_model/gpu_env.py',
             ROOT/'cutting_world_model/free_cut.py']}}
 (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
 env=None
 try:
  env=MultiConditionEnvironment.from_config(ROOT/'backends/mujoco',cfg)
  physical=env.members[0].env; physics=physical.physics; n=env.worlds
  board=physics.board if args.mode=='board' else KnifeBoard(physics.material,BOARD,enforce=False)
  if args.mode=='legacy': board.reset(torch.ones(n,device='cuda',dtype=torch.bool))
  frame=env.current; conditions=env.conditions(range(n))
  q0=physics.material.state_field('particle_q'); v0=physics.material.state_field('particle_qd')
  inv_mass=physics.material.model.particle_inv_mass.reshape(n,-1)
  mass=inv_mass.reciprocal(); com_velocity=(mass[...,None]*v0).sum(1)/mass.sum(1)[:,None]
  initial_rest=dict(particle_speed_max_mps=v0.norm(dim=-1).amax(-1).cpu().tolist(),
                    particle_speed_rms_mps=v0.square().sum(-1).mean(-1).sqrt().cpu().tolist(),
                    com_speed_mps=com_velocity.norm(dim=-1).cpu().tolist(),
                    minimum_particle_height_m=q0[:,:,1].amin(-1).cpu().tolist())
  (args.output/'initial_rest.json').write_text(json.dumps(initial_rest,indent=2))
  (args.output/'conditions.json').write_text(json.dumps(conditions,indent=2))
  history=frame['observation'].float()[None].repeat(policy.history,1,1)
  alive=torch.ones(n,device='cuda',dtype=torch.bool)
  acc={k:torch.zeros(n,device='cuda',dtype=torch.float64) for k in ['return','peak_ratio','force_peak']}
  old_pose=[physics.material.position.clone(),physics.material.rotation.clone()]
  first_board=torch.full((n,),-1,device='cuda',dtype=torch.long)
  first_covered=torch.full_like(first_board,-1)
  records={k:[] for k in ['alive','action','knife_pose','min_gap','path_progress','release_progress','reward','board_contact','joint_q','peak_wrench','cut_midpoint','coverage_mask','particle_min_y']}
  rows=[]; t=0
  def callback(ph,active):
   if args.mode=='legacy':
    board.observe(old_pose[0],old_pose[1],active & alive)
    board.contacted |= (board.gap()<=BOARD['stop_gap_m']) & active & alive
   old_pose[0]=ph.material.position.clone();old_pose[1]=ph.material.rotation.clone()
   touched=alive & board.contacted & (first_board<0); first_board[touched]=t
   covered=alive & board.covered.all(-1) & (first_covered<0); first_covered[covered]=t
  with torch.no_grad():
   for t in range(1,old_task['horizon_steps']+1):
    if time.monotonic()-start>args.max_seconds: raise TimeoutError('Evaluation wall budget')
    goal=env.goal(frame); action=policy(history,goal)
    bounds=action.new_tensor(meta['action_bounds'])
    action=action.clamp(bounds[0],bounds[1])
    # Preserve finished worlds without resampling and without charging reset cost.
    physics.evaluation_paused=~alive
    physical.needs_reset.zero_()
    f,reward,term,trunc,info=physical.step(action,sample_callback=callback)
    # Direct single-group step for diagnostic callback; preserve task conditions.
    frame=dict(f,target_speed=env.target_speed[:,None].to(f['observation']))
    rho=(f['cut_spring_ke']==0).double().mean(-1)
    for k,v in dict(alive=alive,action=action,knife_pose=f['knife_pose'],
                    min_gap=board.min_gap,path_progress=board.covered.double().mean(-1),
                    release_progress=rho,reward=reward,board_contact=board.contacted,
                    joint_q=f['joint_q'],peak_wrench=f['peak_wrench'],cut_midpoint=board.midpoints(),
                    coverage_mask=board.covered,particle_min_y=physics.material.state_field('particle_q')[:,:,1].amin(-1)).items():
     records[k].append(v.detach().cpu().clone().numpy())
    acc['return']+=torch.where(alive,reward,0)
    acc['peak_ratio']=torch.maximum(acc['peak_ratio'],torch.where(alive,info['safety_margin'],0))
    acc['force_peak']=torch.maximum(acc['force_peak'],torch.where(alive,info['peak_force'],0))
    ended=(term|trunc)&alive
    for w in ended.nonzero().flatten().tolist():
     ending=('success' if info['success'][w] else 'force' if info['collision'][w] else
             'workspace' if info['outside'][w] else 'board_incomplete' if args.mode=='board' and info['board_incomplete'][w] else 'timeout')
     row=dict(world=w,**conditions[w],ending=ending,steps=t,success=bool(info['success'][w]),
              released_fraction=float(rho[w]),path_fraction=float(board.covered[w].double().mean()),
              min_knife_height_m=float(board.min_gap[w]),first_board_step=int(first_board[w]),
              first_coverage_step=int(first_covered[w]),return_value=float(acc['return'][w]),
              peak_force_n=float(acc['force_peak'][w]),peak_limit_ratio=float(acc['peak_ratio'][w]))
     rows.append(row)
     with open(args.output/'episodes.jsonl','a') as stream: stream.write(json.dumps(row)+'\n')
    alive &= ~ended
    history=torch.cat((history[1:],frame['observation'].float()[None]),0)
    if t%20==0 or not bool(alive.any()):
     print(json.dumps({'step':t,'finished':len(rows),'seconds':time.monotonic()-start}),flush=True)
    if not bool(alive.any()): break
  if bool(alive.any()): raise RuntimeError('Incomplete evaluation')
  np.savez_compressed(args.output/'transitions.npz',**{k:np.stack(v) for k,v in records.items()})
  summary=dict(mode=args.mode,episodes=n,success=sum(r['success'] for r in rows),
               endings={k:sum(r['ending']==k for r in rows) for k in ['success','force','workspace','board_incomplete','timeout']},
               below_board=sum(r['min_knife_height_m'] < -2e-6 for r in rows),
               minimum_gap_m=min(r['min_knife_height_m'] for r in rows),
               mean_return=float(np.mean([r['return_value'] for r in rows])),
               seconds=time.monotonic()-start)
  (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
  print(json.dumps(summary),flush=True)
 finally:
  if env is not None: env.close()
if __name__=='__main__': main()
