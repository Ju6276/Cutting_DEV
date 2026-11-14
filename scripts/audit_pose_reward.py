"""Batched reference reward audit using the actual pose environment and WM equations.

References bypass the learned six-action mapping. No dummy reference actions are
inserted into replay. Finished worlds pause and are scored only until their first
terminal transition. This is an engineering comparison, not policy training.
"""
import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'backends/mujoco')]
from cutting_world_model.gpu_env import GPUEnvironment
from cutting_world_model.multi_env import knife_clearance
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.pose_cut import PoseCutStateSpec
from cutting_world_model.provenance import git_revision
from cutting_gpu_backend.reference import KnifeTrajectory

CASES = ('press','saw','fast','extreme_fast','slow','stall','retract',
         'slow_press','hold_in_material','force_attempt')


def trajectory(case, distance, speed, dt, horizon):
    factor = 3 if case == 'fast' else 20 if case in ('extreme_fast','force_attempt') else .25 if case == 'slow' else 1
    rate = speed*factor
    depth = 0.
    times = [0.]
    velocities = []
    for step in range(horizon):
        t = step*dt
        down = min(rate, max(0.,distance-depth)/dt)
        if case == 'stall':
            down = 0.
        elif case == 'retract' and t >= .5*distance/speed:
            down = -speed if depth > 0 else 0.
        elif case == 'hold_in_material' and depth >= distance*.5:
            down = 0.
        elif case == 'slow_press' and depth >= distance*.5:
            down = min(speed*.25, max(0.,distance-depth)/dt)
        depth += down*dt
        amplitude = .08 if case == 'force_attempt' else .02
        frequency = 3. if case == 'force_attempt' else 1.
        lateral = (amplitude*(math.sin(2*math.pi*frequency*(t+dt))-math.sin(2*math.pi*frequency*t))/dt
                   if case in ('saw','force_attempt') and down > 0 else 0.)
        velocities.append([0.,-down,lateral])
        times.append((step+1)*dt)
    return KnifeTrajectory(times, velocities, {'case':case})


class IndependentReferences:
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def displacement(self, time, rotation=None):
        if len(time) != len(self.trajectories):
            raise ValueError('One reference per physical world')
        return torch.cat([traj.displacement(time[i:i+1], rotation)
                          for i,traj in enumerate(self.trajectories)])


def components(task, old_progress, frame, target_speed, info):
    w=task.weights
    increment=frame['cut_progress'][:,0]-old_progress
    speed=task.heads.cutting_speed(frame['observation'])
    mismatch=((speed-target_speed)/(speed.abs()+target_speed)).square()
    use=(frame['peak_wrench'].abs()/frame['peak_wrench'].new_tensor(task.heads.force_limits)).amax(-1).clamp(max=1)
    return dict(progress=w['progress']*increment,
                speed_progress=-w['speed']*mismatch*increment.clamp_min(0),
                speed_time=-w['speed']*mismatch/task.horizon,
                load=-task.dt*w['load']*use,
                jerk=-task.dt*w['jerk']*task.heads.jerkiness(frame['observation']).clamp(max=1),
                settlement=w['success']*info['success'].double()
                           -w['failure']*(info['collision']|info['outside']).double())


def run(config_path, output, deadline):
    cfg=json.loads(config_path.read_text())
    group=cfg['groups'][0]
    if cfg['task']['version'] != 'free_cut_pose_v1':
        raise ValueError('This audit is for the current pose task')
    output.mkdir(parents=True,exist_ok=False)
    n=len(CASES);speed=.03;clearance=.01
    lift=clearance-knife_clearance(ROOT/'backends/mujoco',group['scene'])
    env=GPUEnvironment(ROOT/'backends/mujoco',n,scene=group['scene'],
                       initial_lift=torch.full((n,),lift), sim_dt=group.get('sim_dt'),
                       force_model=group['force_model'],friction_velocity=group['friction_velocity'],
                       task=cfg['task'],force_limits=cfg['force_limits'])
    env.target_speed=torch.full((n,),speed,device=env.device,dtype=torch.float64)
    frame=env.settle(group['settle_seconds'])
    meta=ControlStateSpec.metadata_from_frame(env.metadata,{k:v[0].cpu().numpy() for k,v in frame.items()})
    spec=PoseCutStateSpec(meta).to(env.device).double()
    state=spec.pack_batch(frame).double()
    goal=torch.cat((frame['target_pose'],env.target_speed[:,None]),-1).double()
    distance=frame['knife_board_gap'][:,0]-cfg['task']['board']['stop_gap_m']
    refs=[trajectory(case,float(distance[i]),speed,env.free_task.dt,env.free_task.horizon)
          for i,case in enumerate(CASES)]
    env.physics.set_reference_trajectory(IndependentReferences(refs))
    manifest=dict(git=git_revision(ROOT),source_config=cfg,worlds=n,cases=CASES,nominal_material=True,
                  clearance_m=clearance,target_speed_mps=speed,
                  trajectories=[r.to_dict() for r in refs],action_interface='prescribed reference; not policy actions')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    alive=torch.ones(n,device=env.device,dtype=torch.bool)
    totals=torch.zeros(n,device=env.device,dtype=torch.float64)
    sums={}
    endpoints=[]
    records=[]
    formula_error=0.;wm_error=0.
    started=time.monotonic()
    with (output/'steps.jsonl').open('w') as stream, torch.no_grad():
        for step in range(env.free_task.horizon):
            if time.monotonic()>deadline:
                raise TimeoutError('Reward audit wall budget')
            active=alive.clone()
            env.physics.evaluation_paused=~alive
            env.needs_reset.zero_()  # Finished worlds stay physically paused, never resampled.
            previous=frame['cut_progress'][:,0].clone()
            action=torch.zeros(n,6,device=env.device,dtype=torch.float64)
            frame,reward,done,truncated,info=env.step(action,reference=True)
            parts=components(env.free_task,previous,frame,env.target_speed,info)
            reconstructed=sum(parts.values())
            err=float((reconstructed[active]-reward[active]).abs().max())
            formula_error=max(formula_error,err)
            if err>2e-5:raise AssertionError('Reward decomposition disagrees with training environment')
            fields={k:frame[k].double() for k in spec.physical_fields}
            _,model_reward,_=spec.derive(state,fields,action,goal)
            err=float((model_reward[active]-reward[active]).abs().max())
            wm_error=max(wm_error,err)
            if err>1e-3:raise AssertionError(f'WM reward equations disagree on real transition: {err}')
            next_state=spec.pack_batch_transition(frame,state).double()
            totals+=torch.where(active,reward,0)
            for k,v in parts.items():
                sums[k]=sums.get(k,torch.zeros_like(totals))+torch.where(active,v,0)
            row=dict(step=step+1,alive=active.tolist(),reward=reward.tolist(),
                     components={k:v.tolist() for k,v in parts.items()},
                     success=info['success'].tolist(),force=info['collision'].tolist(),
                     outside=info['outside'].tolist(),timeout=info['timeout'].tolist(),
                     progress=frame['cut_progress'][:,0].tolist(),
                     speed_mps=env.heads.cutting_speed(frame['observation']).tolist(),
                     peak_force_n=info['peak_force'].tolist(),limit_utilization=info['safety_margin'].tolist(),
                     board_gap_m=frame['knife_min_board_gap'].reshape(-1).tolist())
            stream.write(json.dumps(row)+'\n');stream.flush()
            records.append(dict(state=state.cpu().numpy(),next_state=next_state.cpu().numpy(),
                                reward=reward.cpu().numpy(),alive=active.cpu().numpy(),done=done.cpu().numpy(),
                                target_pose=env.physics.reference_target(env.physics.sim_time).cpu().numpy()))
            ended=(done|truncated)&alive
            for i in ended.nonzero().flatten().tolist():
                outcome=('force' if info['collision'][i] else 'workspace' if info['outside'][i]
                         else 'success' if info['success'][i] else 'timeout')
                endpoints.append(dict(case=CASES[i],world=i,outcome=outcome,steps=step+1,
                                      return_value=float(totals[i]),components={k:float(v[i]) for k,v in sums.items()},
                                      released=int(info['released_springs'][i]),total_springs=frame['cut_spring_ke'].shape[-1],
                                      separated=bool(info['separated'][i])))
            alive &= ~ended
            state=next_state
            if step%20==0 or not alive.any():
                print(config_path.stem,step+1,'finished',len(endpoints),'wall',round(time.monotonic()-started,1),flush=True)
            if not alive.any():break
    if alive.any():raise RuntimeError('Incomplete case batch')
    np.savez_compressed(output/'transitions.npz',**{k:np.stack([r[k] for r in records]) for k in records[0]})
    report=dict(source_config=cfg,effective_overrides=dict(worlds=n,nominal_material=True,clearance_m=clearance,target_speed_mps=speed),endpoints=sorted(endpoints,key=lambda x:x['world']),
                reward_decomposition_max_abs_error=formula_error,wm_reward_on_real_transition_max_abs_error=wm_error,
                wall_seconds=time.monotonic()-started,
                physical_force_event_observed=any(r['outcome']=='force' for r in endpoints),
                scope='One nominal case per reference; no learned WM prediction-accuracy or policy-performance claim')
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs',nargs='+',type=Path,default=[ROOT/'configs/free_cut'/f'pose_{s}.json' for s in ('cucumber','potato','prism')])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wall-seconds',type=int,default=2700)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    def stop(*_):raise TimeoutError('Reward audit wall budget')
    signal.signal(signal.SIGALRM,stop);signal.alarm(a.wall_seconds)
    torch.set_num_threads(2);torch.manual_seed(0)
    deadline=time.monotonic()+a.wall_seconds
    try:
        reports={cfg.stem:run(cfg,a.output/cfg.stem,deadline) for cfg in a.configs}
        (a.output/'summary.json').write_text(json.dumps(reports,indent=2))
    finally:
        signal.alarm(0)


if __name__=='__main__':
    main()
