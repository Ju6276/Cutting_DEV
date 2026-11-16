"""Bounded physical/WM reward consistency check, not a policy evaluation."""
import argparse, json, signal, time
from pathlib import Path
import torch
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.world_model import CuttingWorldModel
from cutting_world_model.pose_depth import depth_progress

def audit_loaded_scenes(env, backend):
    if env.metadata.get('free_cut_task',{}).get('scene_alignment'):
        from cutting_world_model.scene_alignment import material_top
        from cutting_world_model.pose_cut import gap_from_pose
        from cutting_gpu_backend.controller.fdcc import quat_apply
        loaded=[]
        groups=([m.env for m in env.members] if hasattr(env,'members') else [env])
        for group_env in groups:
            ph=group_env.physics;f=group_env.current
            top=material_top(group_env,f).double()
            ee=group_env.fk(f['joint_q'].double())
            gap=gap_from_pose(ee,group_env.metadata['board_pose_geometry'])
            if backend=='disect':
                knife=ph.board.transformed(ph.material.position,ph.material.rotation,ph.board.vertices.to(ph.material.position))
                physical_gap=knife[:,:,1].amin(-1).double()
                mu=dict(ground=float(ph.material.model.contact_mu),
                        knife=sorted(ph.material.model.sdf_mu.cpu().unique().tolist()))
            else:
                pose=ph.tool_pose()
                knife=quat_apply(pose[:,None,3:].expand(-1,len(ph.vertices),-1),
                    ph.vertices[None].expand(group_env.worlds,-1,-1))+pose[:,None,:3]
                physical_gap=knife[:,:,2].amin(-1)
                mu=[dict(ground=c.friction_dynamic,knife=c.friction_dynamic) for c in ph.material_configs]
            loaded.append(dict(
                top_m=top.tolist(),clearance_m=(physical_gap-top).tolist(),
                requested_clearance_m=group_env.start_clearance_m.tolist(),
                robot_proxy_clearance_m=(gap-top).tolist(),
                ee_pose_base=ee.tolist(),target_pose_base=f['target_pose'].tolist(),
                observation_width=f['observation'].shape[-1],
                motion_command=f['motion_command'].tolist(),friction=mu,
                pitch_control=ph.mapper.pitch_control))
            assert bool(((physical_gap-top)-group_env.start_clearance_m).abs().lt(3e-6).all())
        return loaded
    return []

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['disect','blademaster'],required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--shared-contract',type=Path)
    p.add_argument('--prepared',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=600)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def expire(*_):raise TimeoutError('Depth-task verification wall budget')
    signal.signal(signal.SIGALRM,expire);signal.alarm(a.seconds)
    torch.set_num_threads(2);torch.manual_seed(0)
    if a.shared_contract:
        if a.config or not a.prepared:p.error('Use --shared-contract with --prepared, without --config')
        from cutting_world_model.shared_scene import load_configs
        dc,bc=load_configs(a.shared_contract,a.prepared)
        config=dc if a.backend=='disect' else bc
    else:
        if not a.config or a.prepared:p.error('Specify --config or --shared-contract/--prepared')
        config=json.loads(a.config.read_text())
    env=None
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
        report['loaded_scenes']=audit_loaded_scenes(env,a.backend)
        report['reset_audits']=[]

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
                reward=reward.tolist(),done=done.tolist(),material_separation=info.get('separated',torch.tensor([])).tolist(),
                object_slip_m=info.get('object_slip_m',torch.tensor([])).tolist(),
                object_slip_failure=info.get('object_slip_failure',torch.tensor([])).tolist()))
            state=sp.pack_batch_transition(frame,state).double()
            reset=done.clone()
            if step==7:reset[0]=True
            if reset.any():
                retained={k:v[~reset].clone() for k,v in frame.items()}
                frame=env.reset(reset)
                for k,v in retained.items():torch.testing.assert_close(frame[k][~reset],v,atol=0,rtol=0)
                assert frame['cut_progress'][reset].abs().max()<1e-8
                if 'object_slip' in frame:
                    assert frame['object_slip'][reset].abs().max()<1e-8
                # Audit only reset worlds; other worlds are already cutting.
                report['reset_audits'].append(dict(step=step+1,worlds=reset.nonzero().flatten().tolist()))
                # Actual reset clearance is checked inside align_reset; requested cases stay identical.
                if 'start_clearance_m' in frame:
                    torch.testing.assert_close(frame['start_clearance_m'], old['start_clearance_m'],
                                               atol=0, rtol=0)
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
