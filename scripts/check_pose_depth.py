"""Bounded physical/WM reward consistency check, not a policy evaluation."""
import argparse, json, signal, time
from pathlib import Path
import torch
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.world_model import CuttingWorldModel
from cutting_world_model.pose_depth import depth_progress

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['disect','blademaster'],required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=600)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def expire(*_):raise TimeoutError('Depth-task verification wall budget')
    signal.signal(signal.SIGALRM,expire);signal.alarm(a.seconds)
    torch.set_num_threads(2);torch.manual_seed(0)
    config=json.loads(a.config.read_text());env=None
    report=dict(backend=a.backend,config=config,status='running')
    started=time.monotonic()
    try:
        if a.backend=='disect':
            from cutting_world_model.multi_env import MultiConditionEnvironment
            env=MultiConditionEnvironment.from_config('backends/mujoco',config)
        else:
            from cutting_world_model.blademaster_backend import BladeMasterEnvironment
            env=BladeMasterEnvironment('backends/mujoco',config)
        frame=env.current;n=env.worlds
        metadata=ControlStateSpec.metadata_from_frame(env.metadata,{k:v[0].cpu().numpy() for k,v in frame.items()})
        model=CuttingWorldModel(metadata,hidden=(32,32)).to(env.device).double()
        sp=model.spec;state=sp.pack_batch(frame).double()
        assert frame['cut_progress'].abs().max()<1e-8
        report['initial_depth_span_m']=frame['depth_span'].flatten().tolist()
        rows=[];maximum_error=0.
        for step in range(16):
            old=frame
            action=torch.zeros((n,6),device=env.device,dtype=torch.double)
            frame,reward,done,trunc,info=env.step(action,reference=True)
            goal=env.goal(old).double()
            fields={k:frame[k].double() for k in sp.physical_fields}
            fields['depth_span']=sp.field(state,'depth_span')
            fields['cut_progress']=depth_progress(sp.fk(fields['joint_q']),sp.board_geometry,
                fields['depth_span'],sp.field(state,'cut_progress'))
            nxt,model_reward,model_done=sp.derive(state,fields,action,goal)
            error=float((model_reward-reward).abs().max());maximum_error=max(maximum_error,error)
            assert error<1e-3, error
            torch.testing.assert_close(fields['cut_progress'],frame['cut_progress'].double(),atol=1e-7,rtol=0)
            assert torch.equal(model_done,done)
            assert torch.isfinite(frame['peak_wrench']).all()
            assert bool((frame['cut_progress']>=old['cut_progress']-1e-10).all())
            rows.append(dict(step=step+1,depth=frame['cut_progress'].flatten().tolist(),
                reward=reward.tolist(),done=done.tolist(),material_separation=info.get('separated',torch.tensor([])).tolist()))
            state=sp.pack_batch_transition(frame,state).double()
            reset=done.clone()
            if step==7:reset[0]=True
            if reset.any():
                retained={k:v[~reset].clone() for k,v in frame.items()}
                frame=env.reset(reset)
                for k,v in retained.items():torch.testing.assert_close(frame[k][~reset],v,atol=0,rtol=0)
                assert frame['cut_progress'][reset].abs().max()<1e-8
                state=sp.reset_batch_history(frame,state,reset)
            print(a.backend,'step',step+1,'depth',rows[-1]['depth'],flush=True)
        report.update(status='passed',steps=rows,wm_reward_error_max=maximum_error,
            masked_reset_unselected_exact=True,metadata=metadata)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        if env is not None:env.close()
        report['wall_seconds']=time.monotonic()-started
        a.output.write_text(json.dumps(report,indent=2))
        signal.alarm(0)

if __name__=='__main__':main()
