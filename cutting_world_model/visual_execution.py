"""Pre-episode placement for a bounded visual goal experiment (BladeMaster only)."""
import numpy as np
import torch
from .pose_cut import gap_from_pose

@torch.no_grad()
def place_before_episode(env, pose, goal=None):
    """Simulation reset/IK placement, not a collision-checked approach trajectory.

    Material states are untouched. Forbidden after any policy step.
    """
    if env.metadata["physical_backend"] != "mujoco_warp_blademaster" or env.worlds != 1:
        raise ValueError("Visual reset prototype requires one native BladeMaster world")
    if int(env.step_count.max()) != 0 or bool(env.needs_reset.any()):
        raise ValueError("Cannot reposition during an episode")
    p=env.physics
    target=torch.as_tensor(pose,device=env.device,dtype=torch.float64).reshape(1,7)
    box=env.free_task.config["workspace"]
    if not bool(torch.isfinite(target).all()) or bool(((target[:,:3]<target.new_tensor(box["lower"])) |
              (target[:,:3]>target.new_tensor(box["upper"]))).any()):
        raise ValueError("Visual starting pose outside workspace")
    if not bool(torch.isclose(target[:,3:].norm(dim=-1),target.new_ones(1),atol=1e-5).all()):
        raise ValueError("Invalid tool quaternion")
    goal_tensor=None
    if goal is not None:
        goal_tensor=torch.as_tensor(goal,device=env.device,dtype=torch.float64).reshape(1,8)
        if not bool(torch.isfinite(goal_tensor).all()):
            raise ValueError("Nonfinite visual goal")
        if bool(((goal_tensor[:,:3]<target.new_tensor(box["lower"])) |
                 (goal_tensor[:,:3]>target.new_tensor(box["upper"]))).any()):
            raise ValueError("Visual goal outside workspace")
        if not bool(torch.isclose(goal_tensor[:,3:7].norm(dim=-1),target.new_ones(1),atol=1e-5).all()):
            raise ValueError("Invalid goal quaternion")
        if not .01 <= float(goal_tensor[0,7]) <= .05:
            raise ValueError("Visual goal speed outside current training range")
    import warp as wp
    before=[wp.to_torch(s.x.data).clone() for _,s in p.batch_groups]
    q=p.solve_pose(p.robot.q.double().clone(),target)
    p.robot.set_joint_positions(q)
    p.frames=p.chain(p.robot.q.double(),mass_matrix=False)[0]
    actual=p.frames[p.chain.tip]
    from cutting_gpu_backend.controller.fdcc import pose_error
    error=pose_error(actual,target)
    if float(error.abs().max())>2e-6:
        raise AssertionError("Visual starting pose IK mismatch: "+str(error.cpu().tolist()))
    if float(gap_from_pose(actual,env.metadata["board_pose_geometry"])[0])<=0:
        raise ValueError("Visual start must remain above board")
    p.controller.reset(p.robot.q.double(),p.robot.qd.double())
    p.mapper.reset(env.worlds,env.device,torch.float64)
    desired=p.tool_pose()
    for indices,s in p.batch_groups:
        wp.to_torch(s.tool_pos.data)[:,0]=desired[indices,:3].float()
        wp.to_torch(s.tool_rot.data)[:,0]=desired[indices,3:].float()
        wp.to_torch(s.tool_vel.data).zero_();wp.to_torch(s.tool_omega.data).zero_()
    if goal_tensor is not None:
        p.target.copy_(goal_tensor[:,:7]);env.target_speed.copy_(goal_tensor[:,7])
    p.initial_pose.copy_(actual)
    p.reward_depth_span=(gap_from_pose(actual,p.board_geometry)-p.stop_gap)[:,None]
    env.depth_span=p.reward_depth_span.clone()
    p.progress.zero_();p.sim_time.zero_()
    p.contacted.zero_();p.peak.zero_();p.net.zero_();p.wrench_queue.zero_()
    p.feedback["sensor_wrench"].zero_()
    env.previous_pose=actual.clone()
    env.previous_velocity.zero_();env.previous_acceleration.zero_();env.success_counter.zero_()
    env.current=env._observe(p.frame(),torch.zeros((1,6),device=env.device))
    if any(not torch.equal(old,wp.to_torch(s.x.data)) for old,(_,s) in zip(before,p.batch_groups)):
        raise AssertionError("Pre-cut pose placement changed material")
    return env.current


def validate_paired_visual_checkpoint(policy_data, model_data):
    """Require the full WM checkpoint to contain the exact frozen source actor."""
    if policy_data["metadata"] != model_data["metadata"]:
        raise ValueError("Visual policy/WM checkpoint metadata differ")
    actor=model_data["learner"]["policy"]
    if actor.keys()!=policy_data["policy"].keys() or any(
        not torch.equal(actor[k].cpu(),v.cpu()) for k,v in policy_data["policy"].items()):
        raise ValueError("Visual policy/WM checkpoint actor weights differ")
