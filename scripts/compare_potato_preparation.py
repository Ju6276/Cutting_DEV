"""Bounded whole-vs-half potato preparation check. No training or parameter fitting.
Six settling conditions, then one nominal-material pose-task reference per shape.
Operational stability uses a trailing 0.25 s window and the same tolerances for both
assets; raw traces are retained, so these thresholds are not a new task reward.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'backends/mujoco')]
from cutting_world_model.gpu_env import GPUEnvironment
from cutting_world_model.multi_env import knife_clearance
from cutting_gpu_backend.reference import KnifeTrajectory
from scripts.record_episode import GeometryRecorder
from cutting_world_model.provenance import git_revision

def metrics(material):
    q=material.state_field('particle_q').double()
    v=material.state_field('particle_qd').double()
    mass=material.model.particle_mass.reshape(material.worlds,-1).double()
    weights=mass/mass.sum(1,keepdim=True)
    return dict(com_m=(q*weights[:,:,None]).sum(1),
                com_velocity_mm_s=(v*weights[:,:,None]).sum(1)*1000,
                particle_rms_mm_s=torch.sqrt((v.square().sum(-1)*weights).sum(1))*1000,
                minimum_y_mm=q[:,:,1].min(1).values*1000,
                released=(material.state_field('cut_spring_ke')==0).sum(1))

def stable_windows(rows):
    """At 50 ms sampling, require five intervals / six endpoints, not a velocity zero crossing."""
    accepted=[]
    for i in range(len(rows)):
        if i<5:
            accepted.append([False]*len(rows[0]['com_m']));continue
        w=rows[i-5:i+1]
        com=np.array([r['com_m'] for r in w])
        speed=np.linalg.norm(np.array([r['com_velocity_mm_s'] for r in w]),axis=-1)
        rms=np.array([r['particle_rms_mm_s'] for r in w])
        travel=np.linalg.norm(com-com[0],axis=-1).max(0)*1000
        accepted.append(((speed.max(0)<=.2)&(rms.max(0)<=1.)&(travel<=.01)).tolist())
    return accepted

def cpu_row(values):
    return {k:v.detach().cpu().tolist() for k,v in values.items()}

def check(shape,scene,args,cfg):
    output=args.output/shape;output.mkdir(parents=True,exist_ok=False)
    begin=time.monotonic()
    clearance=.01;dt=8e-6
    lift=clearance-knife_clearance(ROOT/'backends/mujoco',scene)
    def construct(n):
        e=GPUEnvironment(str(ROOT/'backends/mujoco'),n,scene=scene,sim_dt=dt,initial_lift=torch.full((n,),lift),
                         force_model='net',friction_velocity=.01,settle_seconds=0,task=cfg['task'],force_limits=cfg['force_limits'])
        e.target_speed=torch.full((n,),.03,device=e.device)
        return e
    env=construct(3);m=env.physics.material
    mask=torch.ones(3,dtype=torch.bool,device='cuda')
    if bool((m.model.particle_inv_mass==0).any()):raise ValueError('Fixed particles not allowed')
    # Record effective kernel parameters, not only JSON aliases.
    (output/'effective_contact.json').write_text(json.dumps(dict(
        contact_ke=float(m.model.contact_ke), contact_kd=float(m.model.contact_kd),
        contact_distance_m=float(m.model.contact_distance),
        config_contact_closeness_m=float(m.model.contact_closeness),
        scope='Kernel reads contact_distance; exported contact_closeness is a different field'),indent=2)+'\n')
    scales={k:v.clone() for k,v in m.material_scales.items()}
    scales['youngs_modulus']=torch.tensor([1.,2.,2.],device='cuda')
    scales['cut_spring_stiffness']=torch.tensor([1.,1.,2.],device='cuda')
    m.set_material_scales(mask,**scales);m.restore_export(mask)
    torch.cuda.synchronize();start=time.monotonic()
    rows=[dict(time_s=0.,wall_seconds=0.,**cpu_row(metrics(m)))]
    contact_peak=torch.zeros(3,device='cuda')
    with (output/'settle.jsonl').open('w') as log:
        log.write(json.dumps(rows[0])+'\n')
        for i in range(1,161):
            if time.monotonic()>args.deadline:raise TimeoutError('Preparation comparison wall budget')
            peak=m.advance_stationary(.05)
            contact_peak=torch.maximum(contact_peak,peak)
            torch.cuda.synchronize()
            row=dict(time_s=i*.05,wall_seconds=time.monotonic()-start,**cpu_row(metrics(m)))
            rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
            if i%20==0:print(shape,'settle',i*.05,'s',row['wall_seconds'],'wall',flush=True)
    if bool((contact_peak>0).any()):raise ValueError('Knife contacted the object during settling')
    if any(any(r['released']) for r in rows):raise ValueError('Initial material damaged during settling')
    # Retain complete physical state for the nominal reference; no velocity zeroing.
    nominal=torch.tensor([True,False,False],device='cuda')
    state=m.capture_rows(nominal)
    torch.save(state,output/'settled_nominal.pt')
    accepted=np.asarray(stable_windows(rows))
    first=[];persistent=[]
    for w in range(3):
        ids=np.flatnonzero(accepted[:,w]);first.append(None if not len(ids) else rows[ids[0]]['time_s'])
        persistent.append(next((rows[j]['time_s'] for j in ids if accepted[j:,w].all()),None))
    settle_report=dict(shape=shape,scene=scene,first_stable_window_s=first,
      stable_through_remaining_trace_from_s=persistent,final=rows[-1],wall_seconds=rows[-1]['wall_seconds'],
      criteria=dict(window_seconds=.25,com_speed_mm_s=.2,particle_mass_weighted_rms_mm_s=1.,com_window_displacement_mm=.01),
      sample_seconds=.05,material_pairs=[[1,1],[2,1],[2,2]],density_scale=1,
      scope='Discrete samples do not bound unsampled instantaneous velocities; cold graph construction included in wall time')
    (output/'settle_report.json').write_text(json.dumps(settle_report,indent=2)+'\n')
    del m,env
    # New controller instance, identical nominal material and knife start, restoring
    # the computed material state only. Robot, filter and task counters reset normally.
    e=construct(1);mat=e.physics.material;one=torch.ones(1,dtype=torch.bool,device='cuda')
    mat.restore_rows(one,state);mat.save_reset_state(one);frame=e.reset(one)
    rec=GeometryRecorder(e,ROOT/'backends/mujoco');rec(frame,None)
    (output/'task.json').write_text(json.dumps(e.metadata)+'\n')
    gap=float(frame['knife_board_gap'][0,0])
    distance=max(0.,gap-cfg['task']['board']['stop_gap_m'])
    e.physics.set_reference_trajectory(KnifeTrajectory.constant([0.,-.03,0.],max(distance/.03,1e-6),kind='board_ending_press'))
    initial=cpu_row(metrics(mat));physical=[];peak=0.;ratio=0.;min_gap=gap;ret=0.
    with (output/'reference.jsonl').open('w') as log:
        for step in range(cfg['task']['horizon_steps']):
            if time.monotonic()>args.deadline:raise TimeoutError('Preparation comparison wall budget')
            frame,reward,done,truncated,info=e.step(torch.zeros(1,6,device='cuda'),reference=True)
            rec(frame,info)
            peak=max(peak,float(info['peak_force'][0]));ratio=max(ratio,float(info['safety_margin'][0]))
            min_gap=min(min_gap,float(frame['knife_min_board_gap'][0,0]) if frame['knife_min_board_gap'].ndim>1 else float(frame['knife_min_board_gap'][0]))
            ret+=float(reward[0])
            row=dict(step=step+1,reward=float(reward[0]),done=bool(done[0]),truncated=bool(truncated[0]),
              success=bool(info['success'][0]),outside=bool(info['outside'][0]),collision=bool(info['collision'][0]),
              separated=bool(info['separated'][0]),released=int(info['released_springs'][0]),
              total=int(frame['cut_spring_ke'].shape[-1]),gap_m=float(frame['knife_board_gap'][0,0]),
              peak_force_n=float(info['peak_force'][0]),force_limit_utilisation=float(info['safety_margin'][0]),
              goal_error=(frame['observation'][0,:3]*frame['observation'].new_tensor(e.heads.max_distance)).tolist(),
              success_counter=float(frame['success_counter'][0,0]))
            physical.append(row);log.write(json.dumps(row)+'\n');log.flush()
            if bool(done[0] or truncated[0]):break
    rec.save(output,1)
    report=dict(shape=shape,scene=scene,task=cfg['task'],force_limits=cfg['force_limits'],
      initial_clearance_requested_m=clearance,nominal_material=True,settling_seconds=8.,
      reference_speed_m_s=.03,reference_duration_s=distance/.03,initial=initial,final=cpu_row(metrics(mat)),
      endpoint=physical[-1],return_sum=ret,peak_material_force_n=peak,max_force_limit_utilisation=ratio,
      min_board_gap_m=min_gap,wall_seconds=time.monotonic()-begin,
      scope='One reference episode; pose task completion and material separation reported separately')
    (output/'reference_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return dict(settle=settle_report,reference=report)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wall-seconds',type=float,default=1800)
    p.add_argument('--shape',choices=['whole','half','both'],default='both')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(0);args.deadline=time.monotonic()+args.wall_seconds
    cfg=json.loads((ROOT/'configs/free_cut/pose_potato.json').read_text())
    scenes=dict(whole='runs/sweep/free_potato_stable_20261114_v3/scene.pt',half='runs/sweep/free_potato_half_20261114/scene.pt')
    selected=scenes if args.shape=='both' else {args.shape:scenes[args.shape]}
    (args.output/'manifest.json').write_text(json.dumps(dict(git=git_revision(ROOT),command=sys.argv,
      scene_sha256={k:hashlib.sha256((ROOT/'backends/mujoco'/v).read_bytes()).hexdigest() for k,v in selected.items()},
      config=cfg,wall_seconds=args.wall_seconds),indent=2)+'\n')
    result={k:check(k,v,args,cfg) for k,v in selected.items()}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
