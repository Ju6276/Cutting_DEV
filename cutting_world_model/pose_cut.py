"""Target-pose task with a finite-blade board boundary; legacy v3 running costs retained."""
import torch
from .kinematics import quat_apply, quat_inverse
from .free_cut_state import FreeCutStateSpec

def gap_from_pose(pose, geometry):
    vertices = pose.new_tensor(geometry['vertices_in_ee'])
    q = pose[..., None, 3:].expand(*pose.shape[:-1], len(vertices), 4)
    v = vertices.expand(*pose.shape[:-1], len(vertices), 3)
    points = pose[..., None, :3] + quat_apply(q, v)
    origin = pose.new_tensor(geometry['board_origin_base'])
    normal = pose.new_tensor(geometry['board_normal_base'])
    return ((points-origin)*normal).sum(-1).amin(-1)

@torch.no_grad()
def align_board_goal(physics, backend_root):
    from cutting_gpu_backend.controller.fdcc import quat_apply as apply
    from pathlib import Path
    import numpy as np
    # Canonical source reset avoids batch/clearance-dependent metadata roundoff.
    q = torch.as_tensor(np.load(Path(backend_root)/'runs/source/episode_000.npz')['joint_q'][0],
                        device=physics.device,dtype=physics.dtype)[None]
    initial_pose = physics.chain(q,mass_matrix=False)[0][physics.chain.tip]
    tool_pose = physics.knife_chain(q,mass_matrix=False)[0]['b_bot_knife_sim']
    material_pose = physics.coupling.reset_pose(tool_pose)
    board = physics.board
    transform = physics.coupling.base_from_disect
    vertices = board.transformed(material_pose[:,:3], material_pose[:,3:], board.vertices.to(material_pose)).to(initial_pose)
    base = apply(transform[3:].expand(*vertices.shape[:-1],4), vertices) + transform[:3]
    local = quat_apply(quat_inverse(initial_pose[:,None,3:]).expand(-1, vertices.shape[1], -1),
                       base-initial_pose[:,None,:3])
    # All initial-lift worlds differ only by a translation.
    if not torch.allclose(local, local[:1].expand_as(local), atol=2e-6, rtol=0):
        raise ValueError('Board geometry varies between worlds; cannot share goal geometry')
    normal = apply(transform[3:], transform.new_tensor([0.,1.,0.]))
    origin = transform[:3] + normal*board.y
    geometry = dict(vertices_in_ee=local[0].cpu().tolist(),
                    board_origin_base=origin.cpu().tolist(), board_normal_base=normal.cpu().tolist(),
                    stop_gap_m=board.stop_gap,
                    model='rigid finite-blade proxy; physical execution uses integrated DiSECt pose')
    old_gap = gap_from_pose(physics.target, geometry)
    physics.target[:,:3] += (board.stop_gap-old_gap)[:,None]*normal
    if not torch.allclose(gap_from_pose(physics.target,geometry),
                          old_gap.new_full(old_gap.shape,board.stop_gap),atol=1e-9,rtol=0):
        raise AssertionError('Board goal alignment failed')
    geometry['original_target_gap_m'] = old_gap[0].item()
    return geometry

class PoseCutStateSpec(FreeCutStateSpec):
    terminal_feature_dim = 14  # force6, workspace3, pose3, arrival count, remaining time

    def __init__(self, metadata):
        super().__init__(metadata)
        if 'board_pose_geometry' not in metadata:
            raise ValueError('Pose task needs recorded blade geometry and aligned goal')
        self.board_geometry = metadata['board_pose_geometry']

    @torch.compiler.disable
    def integrate_physical(self, state, delta):
        fields = super().integrate_physical(state, delta)
        return project_board(self, state, fields)

    def derive(self, old, fields, action, goal):
        fields=dict(fields,control_history=self.shifted_history(old))
        fields,speed=self.derive_motion(old,fields,action,goal)
        count=self.field(old,'step_count')+1
        obs=self.free_task.observation(fields['observation'],count[...,0])
        peak=fields['peak_wrench']
        force=(peak.abs()>self.force_limits).any(-1)
        _,reached,_,counter=self.heads.termination(obs,self.field(old,'success_counter'),force)
        outside=self.free_task.workspace(obs,self.fk(fields['joint_q']))
        success=reached & ~outside & ~force
        fields.update(observation=obs,success_counter=counter,step_count=count)
        next_state=self.assemble(fields)
        reward=self.free_task.step_reward(self.field(old,'cut_progress')[...,0],
                    fields['cut_progress'][...,0],obs,peak,speed)
        reward=reward+self.free_task.weights['success']*success-self.free_task.weights['failure']*(force|outside)
        return next_state,reward,self.hard_termination(next_state)

    def pose_success(self,state):
        obs=self.field(state,'observation')
        error=(obs[...,:len(self.heads.dims)]*obs.new_tensor(self.heads.max_distance)).abs()
        arrived=(error < error.new_tensor(self.heads.env['goal_threshold'])).all(-1)
        failure,_=self.hard_events(state)
        return arrived & (self.field(state,'success_counter') > self.heads.env['successes_threshold']) & ~failure

    def hard_termination(self,state):
        failure,timeout=self.hard_events(state)
        return failure | timeout | self.pose_success(state)

    def completion_reward(self,state,logits):
        # +100 is paid analytically in derive, never by the learned terminal head.
        return torch.zeros_like(logits)

    def terminal_features(self,state,goal=None):
        obs=self.field(state,'observation')
        force=self.field(state,'peak_wrench').abs()/self.limits(goal)-1
        workspace=self.free_task.workspace_features(obs,self.fk(self.field(state,'joint_q')))
        target=(obs[...,:3]*obs.new_tensor(self.heads.max_distance)).abs()/obs.new_tensor(self.heads.env['goal_threshold'])-1
        counter=self.field(state,'success_counter')[...,None]/(self.heads.env['successes_threshold']+1)-1
        remaining=(1-self.field(state,'step_count')/self.free_task.horizon).clamp(0,1)
        return torch.cat((force,workspace,target,counter,remaining),-1)

def project_board(self, state, fields):
    old_q = self.field(state,'joint_q')
    proposed = fields['joint_q']
    start_gap = gap_from_pose(self.fk(old_q), self.board_geometry)
    end_gap = gap_from_pose(self.fk(proposed), self.board_geometry)
    stop = self.board_geometry['stop_gap_m']
    limited = end_gap < stop
    # A learned transition is a discrete endpoint prediction. Projection is a
    # kinematic boundary proxy, not a claim of predicted board contact force.
    fraction=torch.ones_like(end_gap)
    if bool(limited.any()):
        lo, hi = torch.zeros_like(end_gap), torch.ones_like(end_gap)
        with torch.no_grad():
            for _ in range(32):
                mid=(lo+hi)/2
                safe=gap_from_pose(self.fk(old_q+mid[...,None]*(proposed-old_q)),self.board_geometry)>=stop
                lo=torch.where(safe,mid,lo);hi=torch.where(safe,hi,mid)
        root_q=old_q+lo[...,None]*(proposed-old_q)
        root_gap=gap_from_pose(self.fk(root_q),self.board_geometry)
        if torch.is_grad_enabled() and root_q.requires_grad:
            # Implicit derivative of gap(q_old + alpha*dq)=stop. Treating
            # bisection alpha as a constant gives an incorrect BPTT gradient.
            normal=torch.autograd.grad(root_gap.sum(),root_q,retain_graph=True)[0]
            slope=(normal*(proposed-old_q)).sum(-1).detach()
            valid=slope.abs()>1e-10
            denom=torch.where(valid,slope,torch.ones_like(slope))
            correction=(root_gap-root_gap.detach())/denom
            lo=lo-torch.where(valid,correction,torch.zeros_like(correction))
        fraction=torch.where(limited,lo,fraction)
    fraction=torch.where(start_gap < stop-2e-6,torch.zeros_like(fraction),fraction)
    fields['joint_q']=old_q+fraction[...,None]*(proposed-old_q)
    fields['joint_qd']=torch.where(limited[...,None],torch.zeros_like(fields['joint_qd']),fields['joint_qd'])
    return fields
