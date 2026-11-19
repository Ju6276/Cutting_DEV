"""Historical V3 latch loss check (requires metadata without board.motion).

Uses fixed valid six-channel actions, not reference=True with invented replay actions.
A missing physical latch is a failed coverage check, never a pass. No policy updates.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import torch
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.world_model import CuttingWorldModel
from cutting_world_model.online_replay import BatchedOnlineReplay
from cutting_world_model.policy import SliceItPolicy, SliceItCritic
from cutting_world_model.trainer import OnlineLearner
from cutting_world_model.provenance import git_revision
from third_party.supermimic.trainer_config import WMTrainerCfg, PolicyTrainerCfg

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--backend',choices=['disect','blademaster'],default='disect')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=int,default=1800)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--saw',action='store_true',help='Probe bounded oscillatory lateral actions too')
    args=parser.parse_args()
    if args.steps < 1 or args.wall_seconds < 1:
        parser.error('steps and wall-seconds must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    def timeout(*_): raise TimeoutError('Physical latch-learning check wall budget')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(args.wall_seconds)
    torch.set_num_threads(2);torch.manual_seed(17)
    cfg=json.loads(args.config.read_text())
    assert cfg['task']['version']=='pose_depth_v3'
    if cfg['task']['board'].get('motion') == 'project_continue_v1':
        raise ValueError('Historical latch-only probe; current V3 uses check_board_continue.py')
    # Bounded single-object coverage probe, not a multi-object training result.
    cfg['groups']=cfg['groups'][:1]
    cfg['groups'][0]['worlds']=3
    cfg['groups'][0]['start_clearance_m']=[.01]*3
    cfg['parallel']=False
    if args.backend == 'blademaster':
        cfg['worlds']=3
        cfg['start_clearance_m']=[.01]*3
    report={'status':'running','git':git_revision(ROOT),'config':cfg,
            'backend':args.backend,'action_interface':'actual six-channel FDCC; reference=False',
            'policy_updates':0,'saw_probe':args.saw,'rows':[]}
    (args.output/'probe_source.py').write_text(Path(__file__).read_text())
    started=time.monotonic();env=None
    try:
        if args.backend=='disect':
            from cutting_world_model.multi_env import MultiConditionEnvironment
            env=MultiConditionEnvironment.from_config(ROOT/'backends/mujoco',cfg)
        else:
            from cutting_world_model.blademaster_backend import BladeMasterEnvironment
            env=BladeMasterEnvironment(ROOT/'backends/mujoco',cfg)
        f=env.current;n=env.worlds;device=env.device
        meta=ControlStateSpec.metadata_from_frame(env.metadata,{k:v[0].cpu().numpy() for k,v in f.items()})
        def make_learner():
            wm=CuttingWorldModel(meta,hidden=(32,32)).to(device)
            return OnlineLearner(wm,SliceItPolicy(8,34).to(device),
                SliceItCritic(wm.spec.state_dim,8).to(device),microbatch=3,
                wm_cfg=WMTrainerCfg(batch_size=12,window=8,batches_per_update=1,compile=None),
                policy_cfg=PolicyTrainerCfg(gamma=1.,compile=None),
                critic_target='real',wm_only_updates=500)
        learner=make_learner();sp=learner.wm.spec
        replay=BatchedOnlineReplay(n,sp.state_dim,device,capacity=512,goal_dim=8)
        state=sp.pack_batch(f).float()
        first=held=after_reset=0
        episodes=torch.zeros(n,dtype=torch.long,device=device)
        action=torch.zeros(n,6,device=device)
        action[:,:4]=torch.tensor([.5,1.,1.] if args.saw else [0.,.5,1.],device=device)[:,None]
        with torch.no_grad(), (args.output/'steps.jsonl').open('w') as log:
            for step in range(args.steps):
                if args.saw:
                    action[:,4]=torch.tensor([0.,.5,-.5],device=device)*math.sin(step*.3)
                goal=env.goal(f).float()
                old=state
                f,reward,done,trunc,info=env.step(action)
                nxt=sp.pack_batch_transition(f,old).float()
                before=sp.contact(old)[:,0];now=sp.contact(nxt)[:,0]
                first+=int((~before & now).sum());held+=int(before.sum())
                after_reset+=int(((episodes>0)&~before).sum())
                if before.any():
                    assert torch.equal(sp.field(nxt,'joint_q')[before],sp.field(old,'joint_q')[before])
                    assert not sp.field(nxt,'joint_qd')[before].any()
                replay.add(old,action,sp.execute(action),goal,reward.float(),nxt,done,trunc)
                row={'step':step+1,'latched_start':before.tolist(),'latched_end':now.tolist(),
                     'done':done.tolist(),'gap_m':f['knife_board_gap'].flatten().tolist()}
                log.write(json.dumps(row)+'\n');log.flush()
                state=nxt
                ended=done|trunc
                if ended.any():
                    previous_q=f['joint_q'].clone()
                    f=env.reset(ended)
                    assert torch.equal(f['joint_q'][~ended],previous_q[~ended])
                    state=sp.reset_batch_history(f,state,ended).float()
                    assert not sp.contact(state)[ended].any()
                    episodes+=ended.long()
                if step%10==0:
                    print(json.dumps({'step':step+1,'first':first,'held':held,
                                      'episodes':episodes.tolist()}),flush=True)
                if first and held>=4 and after_reset>=12 and (episodes>=1).all():
                    break
        report.update(first_contact_transitions=first,latched_start_transitions=held,
                      after_reset_unlatched_transitions=after_reset,episodes=episodes.tolist(),
                      collected_steps=step+1)
        assert first>0 and held>0 and after_reset>0, 'Missing physical latch/reset coverage'
        # Exact dataset counts exclude terminal rows, which are states rather than transitions.
        b=replay.buffer
        fit_rows=b.valid & ~b.terminal
        observed_mask=sp.contact(b.s)[...,0] & fit_rows
        assert int(observed_mask.sum())==held
        report['replay_latched_rows']=int(observed_mask.sum())
        report['updates']=[]
        for _ in range(3):
            metrics=learner.fit_world_model(b)
            assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
            report['updates'].append(metrics)
        # Require that physical held-start windows are actually sampleable.
        lengths=b.window_lengths(8)
        indices=torch.nonzero((lengths>0)&observed_mask)
        assert len(indices), 'No sampleable physical held-start window'
        # Ordinary replay sampling must exercise masking too.
        assert sum(m['dynamics_masked_count'] for m in report['updates'])>0
        e,t=indices[0]
        length=lengths[e,t]
        offsets=torch.arange(9,device=device)
        times=(t+torch.minimum(offsets,length))%b.T
        def held_batch(*_):
            return SimpleNamespace(states=b.s[e,times][:,None].repeat(1,12,1),
                a_exec=b.a_exec[e,times[:-1]][:,None].repeat(1,12,1),
                xi=b.xi[e,t][None].repeat(12,1),
                done=b.done[e,times[1:]][:,None].repeat(1,12),
                valid_steps=(offsets[:-1]<length)[:,None].repeat(1,12))
        dyn=[p.detach().clone() for p in learner.wm.dynamics.parameters()]
        adam=[copy.deepcopy(learner.wopt.state[p]) for p in learner.wm.dynamics.parameters()]
        held_metrics=learner.fit_world_model(SimpleNamespace(sample_windows=held_batch))
        assert held_metrics['dynamics_fit_count']==0
        assert held_metrics['dynamics_skipped_batches']==1
        for old,param,old_adam in zip(dyn,learner.wm.dynamics.parameters(),adam):
            assert torch.equal(old,param)
            for name,value in old_adam.items():
                assert torch.equal(value,learner.wopt.state[param][name])
        report['all_held_update']=held_metrics
        saved={'learner':learner.checkpoint(),'replay':replay.state_dict(),'metadata':meta}
        torch.save(saved,args.output/'checkpoint.pt')
        loaded=torch.load(args.output/'checkpoint.pt',weights_only=False,map_location=device)
        restored=make_learner();restored.restore(loaded['learner'])
        restored_replay=BatchedOnlineReplay(n,sp.state_dim,device,capacity=512,goal_dim=8)
        restored_replay.load_state_dict(loaded['replay'])
        for name in b.tensors:
            assert torch.equal(getattr(b,name),getattr(restored_replay.buffer,name)),name
        rng=torch.cuda.get_rng_state(device)
        expected=learner.fit_world_model(b)
        torch.cuda.set_rng_state(rng,device)
        actual=restored.fit_world_model(restored_replay.buffer)
        for name,value in learner.wm.state_dict().items():
            torch.testing.assert_close(value,restored.wm.state_dict()[name],atol=0,rtol=0)
        assert expected==actual
        report.update(status='passed',resume_metrics=actual,replay_bit_identical=True,
                      resumed_update_bit_identical=True,
                      loss_normalization=restored.checkpoint()['wm_loss_normalization'])
        torch.save(restored.checkpoint(),args.output/'resumed.pt')
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds']=time.monotonic()-started
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if env is not None:
            close=getattr(env,'close',None)
            if close:close()
    print(json.dumps({k:v for k,v in report.items() if k not in ('config','rows','updates')},indent=2))

if __name__=='__main__':main()
