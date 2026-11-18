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

def measure_virtual_translation(env, distance):
    """Probe the actual observation adapter by offsetting only its reference cloud.
    Physical particles, velocities and forces remain untouched and are not advanced.
    """
    from cutting_world_model.object_slip import observe
    groups = [member.env for member in env.members] if hasattr(env, 'members') else [env]
    measured = []
    for group in groups:
        frame = dict(group.current)
        if group.metadata['physical_backend'] == 'mujoco_warp_disect':
            old = group._object_reference
            reference = frame['particle_q'].reshape(group.worlds, -1, 3).clone()
            reference[..., 0] -= distance
            group._object_reference = reference
            try:
                observe(group, frame)
                measured.append(frame['object_slip'].clone())
            finally:
                group._object_reference = old
        else:
            import warp as wp
            old = group._object_references
            references = {}
            for indices, solver in group.physics.batch_groups:
                reference = wp.to_torch(solver.x.data).clone()
                reference[..., 0] -= distance
                references[indices.start] = reference
            group._object_references = references
            try:
                observe(group, frame)
                measured.append(frame['object_slip'].clone())
            finally:
                group._object_references = old
    return torch.cat(measured).flatten()


def audit_motion_reward(env, spec, old_state, fields, action, goal, frame):
    """Counterfactual reward sweep at one recorded physical transition, not sliding dynamics."""
    from cutting_world_model.object_slip import penalty
    cfg = spec.free_task.config['object_slip']
    baseline = None
    rows = []
    for distance in (0., .025, .035, .0425, .05, .06, .061):
        measured = measure_virtual_translation(env, distance)
        torch.testing.assert_close(measured, torch.full_like(measured, distance), atol=2e-7, rtol=0)
        # Exact values isolate the strict reward boundary from float32 geometry roundoff.
        variant = dict(fields, object_slip=torch.full_like(fields['object_slip'], distance))
        nxt, model_reward, model_done = spec.derive(old_state, variant, action, goal)
        observation = spec.field(nxt, 'observation')
        events = spec.free_task.events(observation, variant['peak_wrench'], None,
            spec.field(nxt, 'step_count')[..., 0], pose=spec.fk(variant['joint_q']),
            board={k: frame[k] for k in ('knife_min_board_gap', 'knife_board_contact')},
            success_counter=spec.field(old_state, 'success_counter'),
            object_slip=variant['object_slip'][..., 0])
        reward = spec.free_task.reward(spec.field(old_state, 'cut_progress')[..., 0],
            variant['cut_progress'][..., 0], observation, variant['peak_wrench'],
            goal[:, -1], events)
        torch.testing.assert_close(model_reward, reward, atol=1e-7, rtol=0)
        assert torch.equal(model_done, events['terminated'])
        failed = distance > cfg['failure_m']
        assert bool((events['object_slip_failure'] == failed).all())
        if failed:
            assert bool(events['terminated'].all()) and not bool(events['success'].any())
        if baseline is None:
            baseline = reward.clone()
            baseline_events = events
        cost = penalty(variant['object_slip'][..., 0], cfg, spec.free_task.horizon)
        old_failure = baseline_events['collision'] | baseline_events['outside'] | baseline_events['object_slip_failure']
        new_failure = events['collision'] | events['outside'] | events['object_slip_failure']
        expected_delta = (-cost + spec.free_task.weights['success'] *
            (events['success'].double() - baseline_events['success'].double()) -
            spec.free_task.weights['failure'] * (new_failure.double() - old_failure.double()))
        torch.testing.assert_close(reward-baseline, expected_delta, atol=1e-7, rtol=0)
        rows.append(dict(distance_m=distance, measured_distance_m=measured.tolist(),
            running_penalty=cost.tolist(), reward=reward.tolist(),
            reward_change_from_zero=(reward-baseline).tolist(),
            object_motion_failure=events['object_slip_failure'].tolist(),
            terminated=events['terminated'].tolist()))
    return dict(kind='counterfactual_reference_translation_and_task_equations',
                natural_physical_sliding=False, rows=rows)


def audit_tool_modes(spec, old_state, fields, action, goal, frame):
    """Evaluate both configured modes on fixed measured kinematics, varying only goal XY."""
    import copy
    from cutting_world_model.free_cut import FreeCutTask
    original = spec.free_task
    result = dict(kind='counterfactual_goal_xy_on_measured_transition', natural_large_motion=False, modes={})
    try:
        for enabled in (False,True):
            config = copy.deepcopy(original.config)
            config['tool_motion']['enabled'] = enabled
            spec.free_task = FreeCutTask(config,spec.heads)
            rows=[]; baseline=None
            pose=spec.fk(fields['joint_q'])
            for distance in (0.,.025,.035,.04,.05,.06,.061,.08):
                target=goal.clone();target[:,:2]=pose[:,:2]
                target[:,1]+=distance
                nxt,wm_reward,wm_done=spec.derive(old_state,dict(fields),action,target)
                obs=spec.field(nxt,'observation')
                ev=spec.free_task.events(obs,fields['peak_wrench'],None,
                    spec.field(nxt,'step_count')[...,0],pose=pose,target_pose=target[:,:7],
                    board={k:frame[k] for k in ('knife_min_board_gap','knife_board_contact')},
                    success_counter=spec.field(old_state,'success_counter'))
                reward=spec.free_task.reward(spec.field(old_state,'cut_progress')[...,0],
                    fields['cut_progress'][...,0],obs,fields['peak_wrench'],target[:,-1],ev)
                torch.testing.assert_close(reward,wm_reward,atol=1e-7,rtol=0)
                assert torch.equal(wm_done,ev['terminated'])
                expected_failure = enabled and distance > config['tool_motion']['failure_m']
                assert bool((ev['tool_motion_failure'] == expected_failure).all())
                if baseline is None:baseline=reward.clone()
                rows.append(dict(requested_distance_m=distance,
                    distance_m=ev['completion_xy_error_m'].tolist(),
                    reward_change_from_zero=(reward-baseline).tolist(),
                    success=ev['success'].tolist(),tool_motion_failure=ev['tool_motion_failure'].tolist()))
            result['modes']['path_and_final' if enabled else 'final_only']=rows
    finally:
        spec.free_task=original
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['disect','blademaster'],required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--shared-contract',type=Path)
    p.add_argument('--prepared',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--motion-reward-audit',action='store_true')
    p.add_argument('--tool-reward-audit',action='store_true')
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
    if a.motion_reward_audit and 'object_slip' not in config['task']:
        p.error('--motion-reward-audit requires a historical object_slip task')
    if a.tool_reward_audit and 'tool_motion' not in config['task']:
        p.error('--tool-reward-audit requires a tool_motion task')
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
        if getattr(env,'condition_sampler',None) is not None:
            report['initial_conditions']=env.conditions(range(n))
            report['episode_sampler_initial']=env.condition_sampler.state_dict()
            if a.backend=='blademaster':
                report['native_material_parameters']=[
                    {name:array.numpy().tolist() for name,array in solver.material_parameters.items()}
                    for _,solver in env.physics.batch_groups]
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
            if 'board_contact' in sp.history_fields:
                # pose_depth_v3: the latch flag is observed from the physical step, as in
                # recorded training transitions (pose_depth.observe_board_latch).
                fields['board_contact']=frame['board_contact'].double()
            nxt,model_reward,model_done=sp.derive(state,fields,action,goal)
            error=float((model_reward-reward).abs().max());maximum_error=max(maximum_error,error)
            assert error<1e-3, error
            torch.testing.assert_close(fields['cut_progress'],frame['cut_progress'].double(),atol=1e-7,rtol=0)
            assert torch.equal(model_done,done)
            assert torch.isfinite(frame['peak_wrench']).all()
            assert bool((frame['cut_progress']>=old['cut_progress']-1e-10).all())
            rows.append(dict(step=step+1,depth=frame['cut_progress'].flatten().tolist(),
                reward=reward.tolist(),done=done.tolist(),material_separation=info.get('separated',torch.tensor([])).tolist(),
                completion_xy_error_m=info.get('completion_xy_error_m',torch.tensor([])).tolist(),
                tool_motion_failure=info.get('tool_motion_failure',torch.tensor([])).tolist(),
                object_slip_m=info.get('object_slip_m',torch.tensor([])).tolist(),
                object_slip_failure=info.get('object_slip_failure',torch.tensor([])).tolist()))
            if a.tool_reward_audit and step == 15:
                torch.save(dict(metadata=metadata,old_state=state,fields=fields,action=action,goal=goal,
                    frame={k:frame[k] for k in ('knife_min_board_gap','knife_board_contact')}),
                    a.output.with_suffix('.transition.pt'))
                report['tool_reward_audit']=audit_tool_modes(sp,state,fields,action,goal,frame)
            if a.motion_reward_audit and step == 15:
                report['motion_reward_audit']=audit_motion_reward(env,sp,state,fields,action,goal,frame)
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
                report['reset_audits'].append(dict(step=step+1,worlds=reset.nonzero().flatten().tolist(),
                    conditions=(env.conditions(range(n)) if hasattr(env,'conditions') else None)))
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
