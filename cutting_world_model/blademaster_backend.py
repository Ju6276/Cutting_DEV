"""UR5e/FDCC coupled to the external BladeMaster forward solver.

The material frame is right handed, z up, metres. A single fixed tool mounting
maps the material knife to the robot EE. Material reactions are interval-average
wrenches at 500 Hz, transformed to the sensor origin before the source filter.
Each world owns an independent material solver; these solvers currently advance
sequentially. No DiSECt scene, pre-cut interface or spring array is constructed.
"""
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from cutting_gpu_backend.controller.fdcc import (
    ControllerConfig, FDCCController, SerialChain, SlicingActionMapper,
    pose_error as controller_pose_error,
)
from cutting_gpu_backend.coupling import (
    SourceWrenchFilter, compose_pose, inverse_pose, rigid_wrench_transform,
    fixed_pose_from_urdf,
)
from cutting_gpu_backend.environment import CuttingBatch
from cutting_gpu_backend.metrics import StepMetrics
from cutting_gpu_backend.robot import Robot
from .kinematics import quat_apply, quat_log, quat_mul, quat_inverse, SerialKinematics
from .task_heads import SliceItHeads, validate_force_limits
from .free_cut import FreeCutTask, validate_task
from .gpu_env import GPUEnvironment
from .pose_cut import gap_from_pose


def blade_vertices(half, bevel):
    x, y, z = half
    section = [(0., -z), (x, -z+bevel), (x, z), (-x, z), (-x, -z+bevel)]
    return np.array([[a, side, b] for side in (-y, y) for a, b in section])


def geometric_progress(pose, geometry, depth_span_m, previous):
    """Maximum board-normal depth reached; not physical separation."""
    depth = (1-gap_from_pose(pose, geometry)/depth_span_m).clamp(0, 1)
    return torch.maximum(previous, depth[..., None])


@torch.no_grad()
def swept_joint_fraction(old_q, new_q, fk, geometry, reach_bound, active):
    """Conservative finite-blade clearance along a revolute joint interpolation.

    Every link-offset vector rotates with angular speed bounded by sum(abs(dq)).
    Thus ||p''(alpha)|| <= reach_bound * sum(abs(dq))**2. A curve deviates
    from its endpoint chord by at most max||p''||/8. The serial-chain reach
    includes every fixed translation and the most distant blade vertex.
    """
    start = gap_from_pose(fk(old_q), geometry)
    delta = new_q-old_q
    curvature = reach_bound*delta.abs().sum(-1).square()/8
    stop = geometry['stop_gap_m']
    def bound(f):
        end = gap_from_pose(fk(old_q+f[:,None]*delta), geometry)
        return torch.minimum(start,end)-curvature*f.square()
    full = torch.ones_like(start)
    hit = active & (bound(full) <= stop)
    lo,hi = torch.zeros_like(start),full.clone()
    if bool(hit.any()):
        for _ in range(30):
            mid=(lo+hi)/2
            safe=bound(mid)>stop
            lo=torch.where(safe,mid,lo);hi=torch.where(safe,hi,mid)
    return torch.where(hit,lo,full),hit


def reaction_at_sensor(force, moment_at_material_origin, base_from_material, sensor_pose):
    wrench = torch.cat((force, moment_at_material_origin), -1)
    base = rigid_wrench_transform(wrench, base_from_material)
    return rigid_wrench_transform(base, inverse_pose(sensor_pose))


def load_material(config):
    root = Path(config['source_root']).resolve()
    if not (root/'blademaster_warp/coupling.py').is_file():
        raise FileNotFoundError('BladeMaster source_root lacks ExternalToolPort')
    sys.path.insert(0, str(root))
    from blademaster_warp.solver import Solver
    from blademaster_warp.coupling import ExternalToolPort, ToolState
    from blademaster_warp.presets import banana_config, banana_dt
    from blademaster_warp.scenarios import scene_config
    name = config['material']
    if name == 'paper-banana':
        cfg = banana_config()
        cfg = replace(cfg, dt=banana_dt(cfg, cfg.grid_spacing, .002))
    else:
        names = {'fig4-cake':'fig4', 'fig7-tofu':'fig7-shallow', 'fig8-luncheon':'fig8'}
        if name not in names:
            raise ValueError('Unknown BladeMaster preset')
        cfg = scene_config(names[name])
    cfg = replace(cfg, external_tool=True, tool_substep='hold', grid_storage='compact')
    if abs(round(.002/cfg.dt)*cfg.dt-.002) > 1e-12:
        raise ValueError('BladeMaster material dt must divide the 2 ms coupling interval')
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip())
    return cfg, Solver, ExternalToolPort, ToolState, {'commit':revision,'dirty':dirty}


class BladeMasterBatch(CuttingBatch):
    @torch.no_grad()
    def __init__(self, root, config):
        root = Path(root)
        self.device, self.dtype = torch.device('cuda:0'), torch.float64
        self.worlds = config['worlds']
        if type(self.worlds) is not int or self.worlds < 1:
            raise ValueError('worlds must be positive')
        self.task = json.loads((root/'runs/source/task.json').read_text())
        self.task['controller']['max_force_torque'] = validate_force_limits(config['force_limits'], required=True)
        self.material_config, self.Solver, self.Port, self.ToolState, revision = load_material(config)
        self.material_metadata = dict(preset=config['material'], config=asdict(self.material_config),
                                      source_root=config['source_root'], source_revision=revision)
        self.clearance = float(config['clearance_m'])
        self.settle_seconds = float(config['settle_seconds'])
        if not all(math.isfinite(x) for x in (self.clearance,self.settle_seconds)) or self.clearance <= 0 or self.settle_seconds < 0:
            raise ValueError('Clearance must be positive and settling nonnegative')
        episode = np.load(root/'runs/source/episode_000.npz')
        self.robot = Robot(root/'runs/model_coupled/robot.mjb',episode['joint_q'][0],
                           episode['joint_qd'][0],self.worlds)
        self.config = ControllerConfig.from_parameters(json.loads((root/'runs/controller_source/controller_params.json').read_text()))
        t = self.task
        self.chain = SerialChain(t['robot_xml'],t['robot_base'],t['robot_tip'],t['joint_names']).to(self.device,self.dtype)
        self.knife_chain = SerialChain(t['robot_xml'],t['robot_base'],'b_bot_knife_sim',t['joint_names']).to(self.device,self.dtype)
        frames = self.chain(self.robot.q.double(),mass_matrix=False)[0]
        source_tool = self.knife_chain(self.robot.q.double(),mass_matrix=False)[0]['b_bot_knife_sim']
        # Keep the source knife centre as the mounting origin. BladeMaster's local
        # z is vertical and local y is along the edge at the initial pose.
        mounted = source_tool.clone()
        mounted[:,3:] = mounted.new_tensor([0.,0.,0.,1.])
        self.ee_from_blade = compose_pose(inverse_pose(frames[self.chain.tip]),mounted)[0].clone()
        board = fixed_pose_from_urdf(t['robot_xml'],t['robot_base'],'cutting_board_disect',
                                    device=self.device,dtype=self.dtype)
        self.base_from_material = mounted[0].clone()
        self.base_from_material[2] = board[2]
        self.base_from_material[3:] = mounted.new_tensor([0.,0.,0.,1.])
        self.material_from_base = inverse_pose(self.base_from_material)
        self.vertices = torch.tensor(blade_vertices(self.material_config.blade_half_size,
                                  self.material_config.blade_bevel_height),device=self.device,dtype=self.dtype)
        self.stop_gap = float(config['task']['board']['stop_gap_m'])
        self.depth_span = float(self.material_config.size[2]+self.clearance)
        wanted = frames[self.chain.tip].clone()
        wanted[:,2] += board[2]+self.depth_span+self.material_config.blade_half_size[2]-mounted[:,2]
        self.robot.set_joint_positions(self.solve_pose(self.robot.q.double(),wanted))
        self.controller = FDCCController(self.chain,self.config)
        self.mapper = SlicingActionMapper(max_twist=t['controller']['max_twist'])
        self.mapper.reset(self.worlds,self.device,self.dtype)
        self.filter = SourceWrenchFilter()
        raw = -self.robot.sensor_wrench(self.config.sensor_ref_link).double()
        self.filter.reset(raw)
        hold = self.robot.q.clone()
        for _ in range(self.filter.window):
            self.robot.step(hold); self.robot.step(hold)
            self.filter.update(-self.robot.sensor_wrench(self.config.sensor_ref_link).double())
        self.filter.tare()
        self.controller.reset(self.robot.q.double(),self.robot.qd.double())
        self.frames = self.chain(self.robot.q.double(),mass_matrix=False)[0]
        local_vertices = quat_apply(self.ee_from_blade[3:].expand(len(self.vertices),-1),self.vertices)+self.ee_from_blade[:3]
        if any(kind == 'prismatic' for kind in self.chain.kinds):
            raise ValueError('Board swept bound currently supports the UR5e revolute chain only')
        self.reach_bound = float(self.chain.origin_position.norm(dim=-1).sum()+local_vertices.norm(dim=-1).max())
        self.board_geometry = dict(vertices_in_ee=local_vertices.cpu().tolist(),
            board_origin_base=self.base_from_material[:3].cpu().tolist(),board_normal_base=[0.,0.,1.],
            stop_gap_m=self.stop_gap,model='Rigid fixed mounted BladeMaster wedge; held 2 ms exchange',
            execution_boundary='1ms revolute joint swept curvature bound',reach_bound_m=self.reach_bound)
        self.initial_pose = self.frames[self.chain.tip].clone()
        self.target = self.initial_pose.clone()
        self.target[:,2] -= gap_from_pose(self.target,self.board_geometry)-self.stop_gap
        self.force_limits = torch.tensor(config['force_limits'],device=self.device,dtype=self.dtype)[None].repeat(self.worlds,1)
        self.desired_wrench = torch.tensor(t['controller']['desired_force_torque'],device=self.device,dtype=self.dtype)[None].expand(self.worlds,-1)
        self.solvers = [None]*self.worlds
        self.ports = [None]*self.worlds
        self.sim_time = torch.zeros(self.worlds,device=self.device,dtype=self.dtype)
        self.peak = torch.zeros((self.worlds,6),device=self.device,dtype=self.dtype)
        self.net = torch.zeros_like(self.peak)
        self.feedback = {'sensor_wrench':torch.zeros_like(self.peak)}
        self.wrench_queue = torch.zeros((self.worlds,25,6),device=self.device,dtype=self.dtype)
        self.contacted = torch.zeros(self.worlds,device=self.device,dtype=torch.bool)
        self.progress = torch.zeros((self.worlds,1),device=self.device,dtype=self.dtype)
        self.metrics = StepMetrics(self.worlds,self.device,self.dtype)
        self.step_metrics = self.metrics.as_dict()
        speed=float(config['target_speed_m_s'])
        if not math.isfinite(speed) or speed<=0:raise ValueError('Target speed must be finite and positive')
        self.target_speed = torch.full((self.worlds,),speed,device=self.device,dtype=self.dtype)
        self.robot.save_reset_state()
        self._controller_initial = {k:v.clone() for k,v in vars(self.controller).items() if isinstance(v,torch.Tensor)}
        self._filter_initial = {k:getattr(self.filter,k).clone() for k in ('history','zi','offset')}
        try:
            self.reset()
        except BaseException:
            self.close()
            raise

    def solve_pose(self, q, target):
        for _ in range(80):
            frames,jacobian,_ = self.chain(q,mass_matrix=False)
            error = controller_pose_error(frames[self.chain.tip],target)
            if float(error.abs().max()) < 1e-7:
                return q
            system = jacobian@jacobian.transpose(-1,-2)+1e-9*torch.eye(6,device=q.device,dtype=q.dtype)
            q = q+(jacobian.transpose(-1,-2)@torch.linalg.solve(system,error[...,None])).squeeze(-1)
        raise RuntimeError('BladeMaster initial pose IK failed')

    def tool_pose(self):
        return compose_pose(self.material_from_base,compose_pose(
            self.frames[self.chain.tip],self.ee_from_blade.expand(self.worlds,-1)))

    def _new_material(self, index):
        solver = self.Solver(self.material_config,device='cuda:0')
        port = self.Port(solver)
        pose = self.tool_pose()[index].cpu().numpy()
        state = self.ToolState(pose[:3],pose[3:],np.zeros(3),np.zeros(3))
        steps = round(self.settle_seconds/.002)
        if abs(steps*.002-self.settle_seconds)>1e-9:
            raise ValueError('settle_seconds must divide the exchange interval')
        for _ in range(steps):
            reaction = port.advance(state,.002,cutting=False)
            if np.linalg.norm(reaction.force_n)>1e-6:
                raise ValueError('Blade touches material during settling')
        self.solvers[index],self.ports[index] = solver,port

    @torch.no_grad()
    def reset(self, mask=None):
        if mask is None: mask=torch.ones(self.worlds,device=self.device,dtype=torch.bool)
        self.robot.reset(mask)
        for k,v in self._controller_initial.items():getattr(self.controller,k)[mask]=v[mask]
        for k,v in self._filter_initial.items():getattr(self.filter,k)[mask]=v[mask]
        self.mapper.added_motion_command[mask]=0
        self.sim_time[mask]=0;self.peak[mask]=0;self.net[mask]=0
        self.feedback['sensor_wrench'][mask]=0;self.wrench_queue[mask]=0
        self.contacted[mask]=False;self.progress[mask]=0
        self.frames=self.chain(self.robot.q.double(),mass_matrix=False)[0]
        for index in mask.nonzero().flatten().tolist():self._new_material(index)
        return self.frame()

    def frame(self):
        pose=self.frames[self.chain.tip]
        gap=gap_from_pose(pose,self.board_geometry)
        return dict(joint_q=self.robot.q.clone(),joint_qd=self.robot.qd.clone(),
            target_pose=self.target.clone(),ee_pose=pose.clone(),
            wrench_history=self.wrench_queue[:,:self.task['environment']['wrench_hist_size']].clone(),
            peak_wrench=self.peak.clone(),motion_command=self.mapper.added_motion_command.clone(),
            cut_progress=self.progress.clone(),knife_board_gap=gap[:,None],
            knife_min_board_gap=gap[:,None],knife_board_contact=self.contacted[:,None].clone(),
            sim_time=self.sim_time[:,None].clone())

    @torch.no_grad()
    def step(self, action, reference=False, sample_callback=None):
        action=action.to(self.device,self.dtype)
        if not bool(torch.isfinite(action).all()):raise ValueError('Actions must be finite')
        if action.shape!=(self.worlds,6):raise ValueError('Expected [worlds,6] actions')
        self.peak.zero_();self.metrics.zero_()
        if reference:
            target=self.initial_pose.clone()
            target[:,2]-=torch.minimum((self.sim_time+.05)*self.target_speed,
                                     self.initial_pose[:,2]-self.target[:,2])
            arguments={'target_pose':target}
        else:
            arguments=self.mapper(action,self.target,self.frames[self.chain.tip])
        collision=torch.zeros(self.worlds,device=self.device,dtype=torch.bool)
        peak_force=torch.zeros(self.worlds,device=self.device,dtype=self.dtype)
        for _ in range(round(self.task['environment']['agent_control_dt']/.002)):
            active=~(collision|self.contacted)
            if not bool(active.any()):break
            old_tool=self.tool_pose()
            for _ in range(2):
                moving=active & ~self.contacted
                if not bool(moving.any()):break
                old_q=self.robot.q.double().clone()
                command=self.controller_command(arguments,moving)
                self.robot.step(command,active=moving)
                new_q=self.robot.q.double().clone()
                fk=lambda q:self.chain(q,mass_matrix=False)[0][self.chain.tip]
                fraction,hit=swept_joint_fraction(old_q,new_q,fk,self.board_geometry,
                                                   self.reach_bound,moving)
                if bool(hit.any()):
                    # Robot uses float32; retain positive numerical clearance.
                    alpha=(fraction-1e-5).clamp_min(0)
                    corrected=torch.where(hit[:,None],old_q+alpha[:,None]*(new_q-old_q),new_q)
                    saved_qd=self.robot.qd.clone()
                    self.robot.set_joint_positions(corrected)
                    self.robot.qd[~hit]=saved_qd[~hit]
                self.frames=self.chain(self.robot.q.double(),mass_matrix=False)[0]
                gap=gap_from_pose(self.frames[self.chain.tip],self.board_geometry)
                self.contacted |= hit & (gap<=1e-5)
            tool_pose=self.tool_pose()
            linear=(tool_pose[:,:3]-old_tool[:,:3])/.002
            angular=quat_log(quat_mul(tool_pose[:,3:],quat_inverse(old_tool[:,3:])))/.002
            for index in active.nonzero().flatten().tolist():
                p=tool_pose[index].cpu().numpy()
                command=self.ToolState(p[:3],p[3:],linear[index].cpu().numpy(),angular[index].cpu().numpy())
                reaction=self.ports[index].advance(command,.002).about(np.zeros(3))
                self.net[index]=torch.as_tensor(np.r_[reaction.force_n,reaction.torque_nm],device=self.device)
            raw=-self.robot.sensor_wrench(self.config.sensor_ref_link).double()
            sensor=reaction_at_sensor(self.net[:,:3],self.net[:,3:],self.base_from_material,
                                      self.frames[self.config.sensor_ref_link])
            saved_filter={k:getattr(self.filter,k)[~active].clone() for k in ('history','zi','offset')}
            filtered=self.filter.update(raw+sensor)
            for k,v in saved_filter.items():getattr(self.filter,k)[~active]=v
            filtered=torch.where(active[:,None],filtered,self.feedback['sensor_wrench'])
            self.feedback['sensor_wrench']=filtered
            self.wrench_queue=torch.where(active[:,None,None],
                torch.cat((self.wrench_queue[:,1:],filtered[:,None]),1),self.wrench_queue)
            observed=filtered.abs()
            self.peak=torch.maximum(self.peak,torch.where(active[:,None],observed,0))
            self.metrics.check_wrench(observed,self.force_limits,active)
            collision|=active & (observed>self.force_limits).any(-1)
            force=self.net[:,:3].norm(dim=-1)
            peak_force=torch.maximum(peak_force,torch.where(active,force,0))
            self.metrics.accumulate(force,linear,active,.002,net_wrench=self.net,
                                    knife_position=tool_pose[:,:3],angular_velocity_disect=angular)
            self.sim_time+=active*.002
            self.progress=geometric_progress(self.frames[self.chain.tip],self.board_geometry,
                                             self.depth_span,self.progress)
            if sample_callback is not None:sample_callback(self,active)
        if bool((gap_from_pose(self.frames[self.chain.tip],self.board_geometry)<-2e-6).any()):
            raise FloatingPointError('BladeMaster knife penetrated the board')
        self.step_metrics=self.metrics.as_dict()
        return self.frame(),collision,peak_force

    def close(self):
        self.ports.clear();self.solvers.clear()


class BladeMasterEnvironment(GPUEnvironment):
    @torch.no_grad()
    def __init__(self, backend_root, config):
        if config['task']['version']!='blademaster_pose_v1':
            raise ValueError('BladeMaster needs its own task identity')
        validate_task(config['task'])
        self.physics=BladeMasterBatch(backend_root,config)
        self.worlds,self.device=self.physics.worlds,self.physics.device
        self.metadata=copy.deepcopy(self.physics.task)
        self.metadata.update(physical_backend='mujoco_warp_blademaster',
            task_interface_revision='blademaster_pose_v1',state_layout='coupled_recurrent_v1',
            material_feedback='blademaster_net_sensor_v1',reset_observation='post_reset',
            free_cut_task=validate_task(config['task']),goal_dim=8,
            board_pose_geometry=self.physics.board_geometry,
            progress_semantics='maximum_board_normal_blade_depth_v1',
            depth_span_m=self.physics.depth_span,
            blademaster=self.physics.material_metadata,
            backend_config=copy.deepcopy(config))
        self.environment=self.metadata['environment'];self.controller=self.metadata['controller']
        self.heads=SliceItHeads(self.environment,self.controller)
        self.free_task=FreeCutTask(config['task'],self.heads)
        self.target_speed=self.physics.target_speed
        self.fk=SerialKinematics(self.metadata['robot_xml'],self.metadata['robot_base'],
                                 self.metadata['robot_tip'],self.metadata['joint_names']).to(self.device)
        self.previous_pose=self.fk(self.physics.robot.q.double())
        self.previous_velocity=torch.zeros((self.worlds,6),device=self.device,dtype=torch.float64)
        self.previous_acceleration=torch.zeros_like(self.previous_velocity)
        self.success_counter=torch.zeros(self.worlds,device=self.device,dtype=torch.float64)
        self.step_count=torch.zeros(self.worlds,device=self.device,dtype=torch.long)
        self.needs_reset=torch.zeros(self.worlds,device=self.device,dtype=torch.bool)
        self.current=None
        # Physics initialization has already produced settled reset states.
        self.current=self._observe(self.physics.frame(),torch.zeros((self.worlds,6),device=self.device))

    def _observe(self, frame, action):
        saved=self.free_task
        self.free_task=None
        try:frame=GPUEnvironment._observe(self,frame,action)
        finally:self.free_task=saved
        frame['observation']=self.free_task.observation(frame['observation'],self.step_count)
        frame['target_speed']=self.target_speed.clone()
        return frame

    @torch.no_grad()
    def step(self, action, reference=False, sample_callback=None):
        if bool(self.needs_reset.any()):raise RuntimeError('Reset ended worlds before stepping')
        action=action.to(self.device).clamp(-1,1)
        frame,collision,force=self.physics.step(action,reference,sample_callback)
        self.step_count+=1
        frame=self._observe(frame,action)
        events=self.free_task.events(frame['observation'],frame['peak_wrench'],frame['cut_progress'][:,0],
            self.step_count,collision,pose=frame['ee_pose'],board=frame,success_counter=self.success_counter)
        reward=self.free_task.reward(self.current['cut_progress'][:,0],frame['cut_progress'][:,0],
                                    frame['observation'],frame['peak_wrench'],self.target_speed,events)
        self.success_counter=events['counter']
        frame['success_counter']=self.success_counter[:,None].clone()
        self.previous_pose=frame['previous_ee_pose'].clone()
        self.previous_velocity=frame['previous_velocity'].clone()
        self.previous_acceleration=frame['previous_acceleration'].clone()
        self.needs_reset=events['terminated']
        self.current=frame
        info={k:events[k] for k in ('success','collision','outside','timeout','board_contact')}
        info.update(peak_force=force,depth_progress=frame['cut_progress'][:,0],**self.physics.step_metrics)
        return frame,reward.float(),events['terminated'],torch.zeros_like(events['terminated']),info

    @staticmethod
    def goal(frame):
        return torch.cat((frame['target_pose'],frame['target_speed'][:,None]),-1).float()

    def restore_frame(self, frame):
        raise NotImplementedError('BM frame lacks deformation/topology state; reset physical episodes on resume')

    def settle(self, seconds, mask=None):
        raise NotImplementedError('Set BladeMaster settle_seconds explicitly in the backend config')

    def close(self):
        self.physics.close()
