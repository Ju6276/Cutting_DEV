#!/usr/bin/env python3
"""Parallel current-task SAC via pinned Stable-Baselines3; not SliceIt TF weights."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'backends/mujoco')]
import torch
from stable_baselines3.common.callbacks import BaseCallback
from cutting_world_model.sac_vector import CuttingVecEnv,TransitionSAC,SACWallBudgetReached
from cutting_world_model.shared_baselines import frame_metadata,validate_current_interface
from cutting_world_model.benchmark import write_json,file_digest
from cutting_world_model.provenance import git_revision
from cutting_world_model.baseline_checkpoint import save_sac_checkpoint,read_sac_metadata
from evaluate_pose_policy import evaluation_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    c=p.add_mutually_exclusive_group(required=True)
    c.add_argument('--conditions',type=Path)
    c.add_argument('--blademaster-config',type=Path)
    p.add_argument('--worlds-per-object',type=int,default=256)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--history',type=int,default=10)
    p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--learning-starts',type=int,default=30720)
    p.add_argument('--transitions-per-gradient',type=int,default=1)
    p.add_argument('--buffer-size',type=int,default=1000000)
    p.add_argument('--max-transitions',type=int,default=1280000)
    p.add_argument('--max-hours',type=float,default=24)
    p.add_argument('--checkpoint-every',type=int,default=100000)
    p.add_argument('--resume',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true',help='Bounded interface/update test only; no cloud requirement')
    args=p.parse_args()
    if min(args.max_transitions,args.checkpoint_every,args.buffer_size,args.batch_size)<1 or args.max_hours<=0:
        p.error('Positive budgets/buffer/batch required')
    if args.smoke and (args.max_transitions>4096 or args.max_hours>1):
        p.error('Smoke test is limited to4096 transitions and1 hour')
    from cutting_world_model.free_cut import refuse_retired_training
    refuse_retired_training(json.loads((args.conditions or args.blademaster_config).read_text()).get('task'))
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    raw=json.loads((args.conditions or args.blademaster_config).read_text())
    config=evaluation_config(raw,args.worlds_per_object,args.seed,bool(args.blademaster_config))
    env=vec=model=tracker=None
    status=dict(state='initializing',pid=os.getpid())
    def record():
        write_json(args.output/'progress.json',dict(status,wall_seconds=time.monotonic()-started))
    def save():
        save_sac_checkpoint(model,args.output,dict(metadata=metadata,config=config,
             history=args.history,method='stable_baselines3_sac',sb3_version='2.7.0',
             num_timesteps=model.num_timesteps,updates=model._n_updates,
             transition_credit=model.transition_credit,
             transitions_per_gradient=model.transitions_per_gradient,
             source_commit=git_revision(ROOT),
             resume_scope='network/optimizers/replay/counters; physical episodes restart'))
    try:
        record()
        if args.blademaster_config:
            from cutting_world_model.blademaster_backend import BladeMasterEnvironment
            env=BladeMasterEnvironment(ROOT/'backends/mujoco',config)
        else:
            from cutting_world_model.multi_env import MultiConditionEnvironment
            env=MultiConditionEnvironment.from_config(ROOT/'backends/mujoco',config)
        metadata=frame_metadata(env)
        from cutting_world_model.free_cut import POSE_DEPTH_VERSIONS
        if metadata.get('free_cut_task',{}).get('version') not in POSE_DEPTH_VERSIONS:
            raise ValueError('SAC multi requires current pose_depth_v2/v3 task')
        vec=CuttingVecEnv(env,args.history)
        if args.resume:
            previous=read_sac_metadata(args.resume,require_replay=True)
            validate_current_interface(previous['metadata'],metadata)
            if previous['history']!=args.history or previous['transitions_per_gradient']!=args.transitions_per_gradient:
                raise ValueError('Resume changes observation or update cadence')
            model=TransitionSAC.load(args.resume/'model.zip',env=vec,device='cuda')
            model.load_replay_buffer(args.resume/'replay.pkl')
            if model.replay_buffer.n_envs != env.worlds:
                raise ValueError('Resume requires unchanged replay world count')
        else:
            model=TransitionSAC('MlpPolicy',vec,learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,learning_starts=args.learning_starts,
                batch_size=args.batch_size,tau=.005,gamma=config['task']['gamma'],
                train_freq=(1,'step'),gradient_steps=1,ent_coef='auto',seed=args.seed,
                policy_kwargs=dict(net_arch=[256,256]),device='cuda',verbose=0,
                transitions_per_gradient=args.transitions_per_gradient)
        start_transitions=model.num_timesteps
        manifest=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            config=config,metadata=metadata,source=git_revision(ROOT),method='SB3 SAC 2.7.0',
            input='observable history + commanded goal, no particle/material labels',
            cadence='physical transitions per SAC gradient; not WM optimizer-round equivalence',
            reset_resume='physical episodes restart; terminal replay preserved',
            source_sha256={str(path.relative_to(ROOT)): file_digest(path) for path in (
                ROOT/'scripts/train_sac_multi.py', ROOT/'cutting_world_model/sac_vector.py',
                ROOT/'cutting_world_model/shared_baselines.py', ROOT/'cutting_world_model/baseline_checkpoint.py')},
            config_sha256=file_digest(args.conditions or args.blademaster_config),
            effective_optimizer=dict(learning_rate=model.learning_rate, batch_size=model.batch_size,
                learning_starts=model.learning_starts, transitions_per_gradient=model.transitions_per_gradient))
        write_json(args.output/'manifest.json',manifest)
        if not args.smoke:
            import wandb
            tracker=wandb.init(project='sliceit-supermimic',mode='online',dir=str(args.output),
                               name=args.output.name,config=manifest)
            write_json(args.output/'wandb.json',dict(id=tracker.id,url=tracker.url))
        status.update(state='running',worlds=env.worlds)
        record()
        class Monitor(BaseCallback):
            last_save=start_transitions
            def _on_rollout_start(self):
                # Previous rollout and all its updates are complete. Do not save
                # from _on_step: SB3 has not inserted that transition yet.
                if model.num_timesteps-self.last_save >= args.checkpoint_every:
                    save()
                    self.last_save=model.num_timesteps
            def _on_step(self):
                status.update(transitions=self.num_timesteps,new_transitions=self.num_timesteps-start_transitions,
                              updates=model._n_updates,replay_rows=model.replay_buffer.size()*env.worlds)
                rows=[x['episode'] for x in self.locals['infos'] if 'episode' in x]
                if rows:status['last_episode_return']=sum(x['r'] for x in rows)/len(rows)
                if tracker is not None:tracker.log(dict(status),step=self.num_timesteps)
                record()
                return True
        model.wall_deadline=started+args.max_hours*3600
        try:
            model.learn(total_timesteps=args.max_transitions,callback=Monitor(),
                        reset_num_timesteps=not bool(args.resume))
        except SACWallBudgetReached:
            pass
        tensors=list(model.policy.state_dict().values())
        if model.log_ent_coef is not None:tensors.append(model.log_ent_coef)
        if any(not torch.isfinite(t).all() for t in tensors):
            raise FloatingPointError('Nonfinite SAC parameters after learning')
        status.update(state='completed',transitions=model.num_timesteps,updates=model._n_updates,
                      new_transitions=model.num_timesteps-start_transitions,
                      replay_rows=model.replay_buffer.size()*env.worlds,
                      stop_reason='transition_budget' if model.num_timesteps-start_transitions >= args.max_transitions else 'wall_budget',
                      transition_credit=model.transition_credit)
        save()
    except BaseException as error:
        status.update(state='failed',error=repr(error))
        raise
    finally:
        try:
            if vec is not None:vec.close()
            elif env is not None:env.close()
        finally:
            record()
            if tracker is not None:tracker.finish(exit_code=int(status['state']=='failed'))


if __name__=='__main__':
    main()
