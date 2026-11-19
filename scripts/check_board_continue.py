"""Bounded real-physics probe of V3 projection without a permanent latch.

Controller target override is diagnostic only, never labelled as policy data.
Runs physics directly so arrival success cannot conceal the post-board behavior.
Phases: press to the board, hold, pure inward FDCC target motion for 1 s at
0.5/2/5 cm/s (tangential drift and rotation of the EE), lateral slide, lift,
partial reset. --repo-root runs this probe against another (clean) checkout.
"""
import argparse,json,signal,time
from pathlib import Path
import torch
from cutting_world_model.provenance import git_revision

ROOT=Path(__file__).resolve().parents[1]
DRIFT_SPEEDS=(.005,.02,.05)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--backend',choices=['disect','blademaster'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wall-seconds',type=int,default=1800)
    p.add_argument('--repo-root',type=Path,default=ROOT,
                   help='Checkout whose code and assets are probed (default: this one)')
    a=p.parse_args()
    root=a.repo_root.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    def timeout(*_):raise TimeoutError('Bounded board continuation probe')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(a.wall_seconds)
    torch.set_num_threads(2)
    cfg=json.loads(a.config.read_text())
    import cutting_gpu_backend.tool_mount as mount
    assert Path(mount.__file__).resolve().is_relative_to(root), ('Probe imports code outside --repo-root',mount.__file__)
    assert cfg['task']['board']['motion']==mount.CONTINUE_BOARD
    cfg['parallel']=False
    cfg['groups']=cfg['groups'][:1]
    cfg['groups'][0]['worlds']=2
    cfg['groups'][0]['start_clearance_m']=[.01]*2
    if a.backend=='blademaster':
        cfg['worlds']=2;cfg['start_clearance_m']=[.01]*2
    env=None;report=dict(status='running',git=git_revision(root),repo_root=str(root),config=cfg,phases=[])
    started=time.monotonic()
    try:
        if a.backend=='disect':
            from cutting_world_model.multi_env import MultiConditionEnvironment
            env=MultiConditionEnvironment.from_config(root/'backends/mujoco',cfg)
            group=env.members[0].env
        else:
            from cutting_world_model.blademaster_backend import BladeMasterEnvironment
            env=BladeMasterEnvironment(root/'backends/mujoco',cfg);group=env
        ph=group.physics
        action=torch.zeros(ph.worlds,6,device=ph.device)
        # Use the original FDCC, sensors, limits and material. Only its commanded
        # target is controlled explicitly to exercise contact, slide and release.
        original=ph.controller_command
        target=ph.frames[ph.chain.tip].clone();start_target=target.clone()
        ph.controller_command=lambda arguments,active:original({'target_pose':target},active)
        def particles():
            if a.backend=='disect':return ph.material.state_field('particle_q').clone()
            import warp as wp
            return torch.cat([wp.to_torch(solver.x.data).clone().reshape(ph.worlds,-1,3)
                              for _,solver in ph.batch_groups],1)
        records=[]
        def step(stage,allow_collision=False):
            seen=[]
            def sample(physics,active):
                seen.append(active.clone())
            tick=time.monotonic()
            frame,collision,force=ph.step(action,sample_callback=sample)
            assert torch.isfinite(frame['joint_q']).all()
            assert (frame['knife_board_gap']>=-2e-6).all()
            record=dict(stage=stage,gap=frame['knife_board_gap'].flatten().tolist(),
                        contact=frame['knife_board_contact'].flatten().tolist(),
                        pose=frame['ee_pose'].tolist(),peak=frame['peak_wrench'].tolist(),
                        active_samples=sum(int(x.sum()) for x in seen),
                        collision=collision.tolist(),step_seconds=time.monotonic()-tick)
            records.append(record)
            (a.output/'transitions.json').write_text(json.dumps(records,indent=2))
            if not allow_collision:
                assert not collision.any(), ('Force limit hit: probe is incomplete',record)
            return frame
        with torch.no_grad():
            for i in range(90):
                target[:,2]=torch.maximum(start_target[:,2]-(i+1)*.0015,ph.target[:,2]-.003)
                f=step('press')
                if bool(f['knife_board_contact'].all()):break
            else:raise AssertionError('No board contact; did not test continuation')
            contact_pose=f['ee_pose'].clone()
            q=f['joint_q'].clone();x=particles()
            sensor=ph.wrench_queue.clone();clock=ph.sim_time.clone()
            for _ in range(3):f=step('hold')
            assert torch.all(ph.sim_time-clock>.149), 'World clock stopped at board'
            assert not torch.equal(sensor,ph.wrench_queue), 'Sensor queue froze'
            assert not torch.equal(x,particles()), 'Material froze'
            report['hold']=dict(material_changed=True,sensor_changed=True,
                elapsed=(ph.sim_time-clock).tolist())
            # Pure inward target motion on the board: any tangential EE motion or
            # rotation here is produced by projection/controller, not commanded.
            from cutting_gpu_backend.controller.fdcc import quat_inverse,quat_mul
            geometry=getattr(ph,'rigid_tool_geometry',None) or ph.board_geometry
            normal=f['ee_pose'].new_tensor(geometry['board_normal_base'])
            normal=normal/normal.norm()
            report['inward_press']={}
            for speed in DRIFT_SPEEDS:
                start=f['ee_pose'].clone();target=start.clone();ticks=time.monotonic()
                collided=torch.zeros(ph.worlds,dtype=torch.bool,device=start.device)
                for k in range(20):
                    target=start.clone();target[:,:3]-=(k+1)*speed*.05*normal
                    f=step('inward_%g'%speed,allow_collision=True)
                    collided|=torch.as_tensor(records[-1]['collision'],device=start.device)
                move=f['ee_pose'][:,:3]-start[:,:3]
                along=(move*normal).sum(-1)
                relative=quat_mul(quat_inverse(start[:,3:]),f['ee_pose'][:,3:])
                rotation=2*torch.atan2(relative[:,:3].norm(dim=-1),relative[:,3].abs())
                report['inward_press']['%g'%speed]=dict(
                    tangential_drift_m=(move-along[:,None]*normal).norm(dim=-1).tolist(),
                    normal_motion_m=along.tolist(),rotation_rad=rotation.tolist(),
                    final_gap_m=f['knife_board_gap'].flatten().tolist(),
                    contact=f['knife_board_contact'].flatten().tolist(),
                    force_limit_event=collided.tolist(),seconds=time.monotonic()-ticks)
                (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
                if bool(collided.any()):
                    raise AssertionError(('Force limit during inward press; drift phase incomplete',speed))
            # Keep a small inward target component to ensure lateral movement
            # is possible even while the controller continues asking to descend.
            target=f['ee_pose'].clone();target[:,0]+=.01;target[:,2]-=.001
            for _ in range(20):f=step('slide')
            slide=f['ee_pose'][:,0]-contact_pose[:,0]
            assert (slide>.003).all(), ('Lateral motion suppressed',slide.tolist())
            target=f['ee_pose'].clone();target[:,2]+=.005
            for _ in range(20):f=step('lift')
            assert (f['knife_board_gap']>.001).all(), 'Could not lift from board'
            assert not f['knife_board_contact'].any(), 'Contact flag stayed latched'
            # Refresh wrapper state after direct-physics diagnostic, then partial reset.
            untouched=f['joint_q'][1:].clone()
            mask=torch.tensor([True,False],device=ph.device)
            ph.reset(mask)
            assert torch.equal(ph.robot.q[1:],untouched.to(ph.robot.q)), 'Partial reset moved other world'
            contact=ph.board.contacted if a.backend=='disect' else ph.contacted
            assert not contact[0]
            report.update(status='passed',contact_step=i+1,slide_m=slide.tolist(),
                          final_gap_m=f['knife_board_gap'].flatten().tolist(),partial_reset=True)
        (a.output/'transitions.json').write_text(json.dumps(records,indent=2))
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        signal.alarm(0);report['seconds']=time.monotonic()-started
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if env is not None and hasattr(env,'close'):env.close()
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
