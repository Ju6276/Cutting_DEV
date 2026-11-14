"""Capture every world after a randomized settled reset; no policy or optimizer.

The snapshot's leading dimension is WORLD, not time. Tile offsets are applied only
later by the renderer. Limit this preview to one object group.
"""
import argparse,json,signal,sys,time
from pathlib import Path
import numpy as np
import mujoco
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'backends/mujoco')]
from cutting_world_model.multi_env import MultiConditionEnvironment
from scripts.record_episode import GeometryRecorder
from cutting_gpu_backend.controller.fdcc import quat_apply
from cutting_world_model.provenance import git_revision

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--config',type=Path,default=ROOT/'configs/free_cut/pose_cucumber.json')
 p.add_argument('--worlds',type=int,default=256)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--wall-seconds',type=int,default=300)
 a=p.parse_args()
 if a.worlds<1:raise ValueError('worlds must be positive')
 if a.wall_seconds<1:raise ValueError('wall budget must be positive')
 a.output.mkdir(parents=True,exist_ok=False)
 def timeout(*_):raise TimeoutError('Batch preview wall budget')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(a.wall_seconds)
 torch.set_num_threads(2);torch.manual_seed(0);started=time.monotonic()
 cfg=json.loads(a.config.read_text())
 if len(cfg['groups'])!=1:raise ValueError('Capture one object group at a time')
 cfg['groups'][0]['worlds']=a.worlds;cfg['parallel']=False
 (a.output/'config.json').write_text(json.dumps(cfg,indent=2))
 print('Initializing actual worlds and per-material settling; no training',flush=True)
 multi=None
 try:
  multi=MultiConditionEnvironment.from_config(str(ROOT/'backends/mujoco'),cfg)
  env=multi.members[0].env;frame=env.current
  recorder=GeometryRecorder(env,ROOT/'backends/mujoco')
  recorder(frame,None);recorder.save(a.output,1)
  (a.output/'task.json').write_text(json.dumps(env.metadata))
  model=env.physics.material.model
  pose=env.physics.material.state_field('body_X_sc')[:,model.knife_link_index].double()
  verts=model.knife_tri_vertices.reshape(-1,3).to(env.device,torch.float64)
  knife=quat_apply(pose[:,None,3:].expand(-1,len(verts),-1),verts[None].expand(a.worlds,-1,-1))+pose[:,None,:3]
  particles=frame['particle_q']
  arrays=dict(joint_q=frame['joint_q'].cpu().numpy(),
    particles=recorder.to_world(particles.reshape(-1,3)).reshape(a.worlds,-1,3),
    knife=recorder.to_world(knife.reshape(-1,3)).reshape(a.worlds,-1,3),
    lift=multi.lift[0].cpu().numpy(),target_speed=multi.target_speed.cpu().numpy())
  for k,v in multi.material_scales.items():arrays[k]=v.cpu().numpy()
  # GeometryRecorder captures world 0 positions. Its packed material model
  # still has all-world topology, so retain precisely the first local block.
  template=dict(np.load(a.output/'episode.npz'))
  for key in ('tri','cut_tri','cut_edge_indices','cut_edge_coords'):
   if len(template[key]) % a.worlds:
    raise ValueError('Packed topology must have equal per-world blocks')
   template[key]=template[key][:len(template[key])//a.worlds]
  if template['cut_edge_indices'].max() >= arrays['particles'].shape[1]:
   raise ValueError('World-0 contact topology is not local')
  np.savez_compressed(a.output/'episode.npz',**template)
  geoms=template['geoms'];bodies=recorder.mj.geom_bodyid[geoms]
  positions=[];rotations=[]
  for joint_q in arrays['joint_q']:
   recorder.data.qpos[recorder.qpos_indices]=joint_q
   mujoco.mj_forward(recorder.mj,recorder.data)
   positions.append(recorder.data.xpos[bodies].copy())
   rotations.append(recorder.data.xmat[bodies].copy())
  arrays['body_pos']=np.stack(positions)
  arrays['body_mat']=np.stack(rotations)
  arrays['world_id']=np.arange(a.worlds)
  if not all(np.isfinite(v).all() for v in arrays.values()):
   raise ValueError('Nonfinite captured state')
  np.savez_compressed(a.output/'world_states.npz',**arrays)
  report=dict(scope='Actual randomized settled INITIAL states; no policy loaded, no training',
    worlds=a.worlds,object=cfg['groups'][0]['name'],leading_dimension='world',
    elapsed_seconds=time.monotonic()-started,revision=git_revision(ROOT),
    task=cfg['task']['version'],physical_actions=0,optimizer_updates=0,
    render_layout='Offsets for viewing only; physical worlds are isolated')
  (a.output/'report.json').write_text(json.dumps(report,indent=2))
  print(json.dumps(report),flush=True)
 finally:
  if multi is not None:multi.close()
  signal.alarm(0)
if __name__=='__main__':main()
