"""Explicit reset clearance relative to settled material; no material repositioning."""
import math
import torch
from .pose_cut import gap_from_pose

REVISION = 'settled_surface_gap_v2'

def validate_alignment(config):
    if (not isinstance(config, dict)
            or set(config) != {'revision', 'clearance_range_m'}
            or config['revision'] != REVISION):
        raise ValueError('Specify settled_surface_gap_v2 and clearance_range_m')
    bounds = config['clearance_range_m']
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) for x in bounds)
            or not 0 < bounds[0] <= bounds[1]):
        raise ValueError('Positive finite ordered start-clearance bounds required')
    return dict(config, clearance_range_m=list(bounds))

def initial_clearances(task, values, worlds, device):
    config = (task or {}).get('scene_alignment')
    if config is None:
        if values is not None:
            raise ValueError('Start clearances require scene_alignment')
        return None
    config = validate_alignment(config)
    if values is None:
        raise ValueError('Explicit paired per-world start_clearance_m cases required')
    result = torch.as_tensor(values, dtype=torch.float64, device=device)
    low, high = config['clearance_range_m']
    if (result.shape != (worlds,) or not bool(torch.isfinite(result).all())
            or not bool(((result >= low) & (result <= high)).all())):
        raise ValueError('Start clearances must match world count and configured range')
    return result.clone()

@torch.no_grad()
def material_top(env, frame):
    if env.metadata['physical_backend']=='mujoco_warp_disect':
        return frame['particle_q'].reshape(env.worlds,-1,3)[:,:,1].amax(-1)
    import warp as wp
    result=frame['joint_q'].new_zeros(env.worlds)
    if not env.physics.batch_groups:
        raise ValueError('Aligned BladeMaster scene requires batched execution')
    for indices,solver in env.physics.batch_groups:
        result[indices]=wp.to_torch(solver.x.data)[:,:,2].amax(-1).to(result)
    return result

@torch.no_grad()
def align_reset(env, frame, mask):
    config=env.metadata.get('free_cut_task',{}).get('scene_alignment')
    if config is None or not bool(mask.any()):
        return frame
    physics=env.physics
    geo=env.metadata['board_pose_geometry']
    pose=env.fk(frame['joint_q'].double())
    top=material_top(env,frame).to(pose)
    target=pose.clone()
    normal=pose.new_tensor(geo['board_normal_base'])
    shift=top+env.start_clearance_m-gap_from_pose(pose,geo)
    target[mask,:3]+=shift[mask,None]*normal
    q=physics.robot.q.double().clone()
    from cutting_gpu_backend.controller.fdcc import pose_error
    for _ in range(80):
        frames,jac,_=physics.chain(q,mass_matrix=False)
        error=pose_error(frames[physics.chain.tip],target)
        if float(error[mask].abs().max())<1e-8:
            break
        system=jac@jac.transpose(-1,-2)+1e-9*torch.eye(6,device=q.device,dtype=q.dtype)
        correction=(jac.transpose(-1,-2)@torch.linalg.solve(system,error[...,None])).squeeze(-1)
        q[mask]+=correction[mask]
    else:
        raise RuntimeError('Settled-surface start IK failed')
    physics.robot.set_joint_positions(q,mask)
    retained={k:v[~mask].clone() for k,v in vars(physics.controller).items() if isinstance(v,torch.Tensor)}
    physics.controller.reset(physics.robot.q.double(),physics.robot.qd.double())
    for k,v in retained.items():
        getattr(physics.controller,k)[~mask]=v
    physics.frames=physics.chain(physics.robot.q.double(),mass_matrix=False)[0]
    pose=physics.frames[physics.chain.tip]
    box=env.metadata['free_cut_task']['workspace']
    lo,hi=pose.new_tensor(box['lower']),pose.new_tensor(box['upper'])
    if bool(((pose[mask,:3]<lo)|(pose[mask,:3]>hi)).any()):
        raise ValueError('Object height/clearance puts the start outside the fixed robot workspace')
    actual=gap_from_pose(pose,geo)-top
    if not bool((actual[mask]-env.start_clearance_m[mask]).abs().le(2e-6).all()):
        raise AssertionError('Loaded start clearance differs from scene contract')
    if env.metadata['physical_backend']=='mujoco_warp_disect':
        from cutting_gpu_backend.coupling import compose_pose
        desired=compose_pose(physics.coupling.disect_from_base,
            compose_pose(pose,physics.rigid_tool_mount.expand(env.worlds,-1)))
        physics.material.set_motion(torch.zeros(env.worlds,6,device=q.device),desired.float(),active=mask)
        knife=physics.knife_chain(physics.robot.q.double(),mass_matrix=False)[0][getattr(physics,'tool_frame','b_bot_knife_sim')]
        physics.coupling.previous_pose_base[mask]=knife[mask]
        physics.motion_initialized[mask]=False
        physics.reference_start[mask]=pose[mask]
        physics.board.reset(mask, gap=gap_from_pose(pose, geo))
    else:
        import warp as wp
        desired=physics.tool_pose()
        for indices,solver in physics.batch_groups:
            local=mask[indices]
            if not bool(local.any()):
                continue
            wp.to_torch(solver.tool_pos.data)[:,0][local]=desired[indices,:3][local].float()
            wp.to_torch(solver.tool_rot.data)[:,0][local]=desired[indices,3:][local].float()
            wp.to_torch(solver.tool_vel.data)[local]=0
            wp.to_torch(solver.tool_omega.data)[local]=0
        physics.initial_pose[mask]=pose[mask]
        physics.reward_depth_span[mask]=(gap_from_pose(pose,geo)[mask]-geo['stop_gap_m'])[:,None]
        if getattr(physics, 'board_current_continuation', False):
            from cutting_gpu_backend.tool_mount import board_contact
            physics.contacted[mask]=board_contact(gap_from_pose(pose,geo))[mask]
    return physics.frame()
