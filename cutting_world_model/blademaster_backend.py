"""UR5e/FDCC coupled to the external BladeMaster forward solver.

The material frame is right handed, z up, metres. A single fixed tool mounting
maps the material knife to the robot EE. Material reactions are interval-average
wrenches at 500 Hz, transformed to the sensor origin before the source filter.
The legacy path gives each world an independent material solver; these advance
sequentially. No DiSECt scene, pre-cut interface or spring array is constructed.
"""
import copy
from dataclasses import asdict, replace
import json
import hashlib
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
from .task_heads import SliceItHeads, validate_force_limits, configure_force_observation
from .free_cut import FreeCutTask, validate_task, POSE_DEPTH_VERSIONS
from .gpu_env import GPUEnvironment
from .pose_cut import gap_from_pose


def blade_vertices(half, bevel, neck=None):
    x, y, z = half
    neck = x if neck is None else neck
    section = [(0., -z), (neck, -z+bevel), (x, z), (-x, z), (-neck, -z+bevel)]
    return np.array([[a, side, b] for side in (-y, y) for a, b in section])


def geometric_progress(pose, geometry, depth_span_m, previous):
    """Maximum board-normal depth reached; not physical separation."""
    depth = (1-gap_from_pose(pose, geometry)/depth_span_m).clamp(0, 1)
    return torch.maximum(previous, depth[..., None])


from cutting_gpu_backend.tool_mount import swept_joint_fraction


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
    import inspect
    if Path(inspect.getfile(Solver)).resolve().parents[1] != root:
        raise RuntimeError('A different BladeMaster source is already imported; start a fresh process')
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
    if config.get('geometry_asset'):
        asset=Path(config['geometry_asset'])
        with np.load(asset) as data:
            cfg=replace(cfg,size=tuple(data['size'].tolist()),grid_spacing=float(data['grid_spacing']),shape='intact_mesh')
        allowed={'young_modulus','poisson_ratio','density','blade_half_size'}
        overrides=dict(config.get('material_overrides',{}))
        if config.get('task',{}).get('tool_revision') and 'blade_half_size' in overrides:
            raise ValueError('Shared tool geometry cannot be overridden by a material group')
        if set(overrides)-allowed:raise ValueError('Only explicit elastic/density/blade overrides are supported')
        if 'blade_half_size' in overrides:overrides['blade_half_size']=tuple(overrides['blade_half_size'])
        cfg=replace(cfg,**overrides)
        cfg=replace(cfg,dt=banana_dt(cfg,cfg.grid_spacing,.002))
    if config.get('task',{}).get('tool_revision'):
        from .shared_knife import HALF,BEVEL,NECK,REVISION
        if config['task']['tool_revision'] != REVISION:
            raise ValueError('Unsupported shared tool')
        if not hasattr(cfg,'blade_edge_half_width'):
            raise ValueError('Shared knife needs the explicit-profile BladeMaster solver')
        cfg=replace(cfg,blade_half_size=HALF,blade_bevel_height=BEVEL,blade_edge_half_width=NECK)
    if 'contact_friction_mu' in config:
        mu=float(config['contact_friction_mu'])
        if not math.isfinite(mu) or mu<0:raise ValueError('Invalid contact friction')
        cfg=replace(cfg,friction_static=mu,friction_dynamic=mu)
    if config.get('episode_conditions') is not None:
        from .episode_conditions import validate
        limits=validate(config['episode_conditions'])
        stiffest=replace(cfg,young_modulus=cfg.young_modulus*limits['youngs_modulus'][1],
                         density=cfg.density*limits['density'][0])
        cfg=replace(cfg,dt=banana_dt(stiffest,cfg.grid_spacing,.002))
    cfg = replace(cfg, external_tool=True, tool_substep='hold', grid_storage='compact')
    if abs(round(.002/cfg.dt)*cfg.dt-.002) > 1e-12:
        raise ValueError('BladeMaster material dt must divide the 2 ms coupling interval')
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip())
    return cfg, Solver, ExternalToolPort, ToolState, {'commit':revision,'dirty':dirty}


class MaterialArrayView:
    def __init__(self,array,index):self.array,self.index=array,index
    def numpy(self):
        import warp as wp
        return wp.to_torch(self.array.data)[self.index].cpu().numpy().copy()


class MaterialWorldView:
    def __init__(self,solver,index):self.solver,self.index=solver,index
    def state(self):
        names=dict(position='x',velocity='v',deformation='F',plastic_deformation='Fp',
                   sides='codes',damage='damage',blade_position='tool_pos',
                   blade_velocity='tool_vel',blade_orientation='tool_rot',blade_omega='tool_omega')
        result={name:getattr(self,source).numpy() for name,source in names.items()}
        if any(not np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError('Nonfinite batched material world')
        if np.linalg.det(result['deformation']).min()<=0:
            raise FloatingPointError('Inverted batched material world')
        return result
    def __getattr__(self,name):
        from blademaster_warp.batched import BatchArray
        value=getattr(self.solver,name)
        return MaterialArrayView(value,self.index) if isinstance(value,BatchArray) else value


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
        groups=config.get('groups') or [dict(name=config['material'],worlds=self.worlds,
                                              material=config['material'])]
        if sum(g['worlds'] for g in groups)!=self.worlds:
            raise ValueError('BladeMaster group world counts must sum to worlds')
        self.groups=groups
        self.condition_sampler=None
        if config.get('episode_conditions') is not None:
            from .episode_conditions import EpisodeConditions
            if config.get('execution') != 'batched':
                raise ValueError('Randomized BladeMaster episodes require batched execution')
            self.condition_sampler=EpisodeConditions(config['episode_conditions'], groups)
        self.material_configs=[];self.reference_points=[];self.group_names=[];self.group_index=[]
        material_groups=[]
        for group_index,group in enumerate(groups):
            if type(group['worlds']) is not int or group['worlds']<1:
                raise ValueError('Positive integer group worlds required')
            member={**config,**group}
            cfg,self.Solver,self.Port,self.ToolState,revision=load_material(member)
            asset=member.get('geometry_asset')
            points=None
            geometry=None
            if asset:
                import inspect
                if 'reference_points' not in inspect.signature(self.Solver).parameters:
                    raise ValueError('This BladeMaster source lacks the intact mesh sampling extension')
                with np.load(asset) as data:points=data['reference_points'].copy()
                geometry=dict(path=asset,sha256=hashlib.sha256(Path(asset).read_bytes()).hexdigest(),
                              particle_count=len(points),predefined_cut_interface=False)
                # Blade edge lies along material y. Cover the whole initial body with margin.
                if np.max(np.abs(points[:,1]))+cfg.grid_spacing/4>=cfg.blade_half_size[1]:
                    raise ValueError('Blade edge does not cover the initial object width')
            material_groups.append(dict(name=group['name'],worlds=group['worlds'],
                preset=member['material'],config=asdict(cfg),source_revision=revision,geometry=geometry))
            for _ in range(group['worlds']):
                self.material_configs.append(cfg);self.reference_points.append(points)
                self.group_names.append(group['name']);self.group_index.append(group_index)
        self.material_config=self.material_configs[0]
        if any((c.blade_half_size,c.blade_bevel_height)!=(self.material_config.blade_half_size,
             self.material_config.blade_bevel_height) for c in self.material_configs):
            raise ValueError('Shared UR5e batch must use one fixed blade mounting and geometry')
        self.material_metadata=dict(source_root=config['source_root'],groups=material_groups)
        if config['task'].get('tool_revision'):
            self.material_metadata['tool_revision']=config['task']['tool_revision']
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
        shared=config['task'].get('tool_revision')
        if shared:
            from .shared_knife import canonical_mount,disect_from_blade
            _,self.ee_from_blade,base_d=canonical_mount(root,self.chain,self.knife_chain,t)
            mounted=compose_pose(frames[self.chain.tip],self.ee_from_blade.expand(self.worlds,-1))

        board = fixed_pose_from_urdf(t['robot_xml'],t['robot_base'],'cutting_board_disect',
                                    device=self.device,dtype=self.dtype)
        self.base_from_material = mounted[0].clone()
        self.base_from_material[2] = board[2]
        self.base_from_material[3:] = mounted.new_tensor([0.,0.,0.,1.])
        if shared:
            self.base_from_material[3:]=compose_pose(base_d,disect_from_blade(base_d))[3:]

        self.material_from_base = inverse_pose(self.base_from_material)
        self.vertices = torch.tensor(blade_vertices(self.material_config.blade_half_size,
                                  self.material_config.blade_bevel_height,
                                  getattr(self.material_config,'blade_edge_half_width',None)),device=self.device,dtype=self.dtype)
        self.stop_gap = float(config['task']['board']['stop_gap_m'])
        self.depth_span = float(max(c.size[2] for c in self.material_configs)+self.clearance)
        wanted = frames[self.chain.tip].clone()
        wanted[:,2] += board[2]+self.depth_span+self.material_config.blade_half_size[2]-mounted[:,2]
        self.robot.set_joint_positions(self.solve_pose(self.robot.q.double(),wanted))
        self.controller = FDCCController(self.chain,self.config)
        self.mapper = SlicingActionMapper(max_twist=t['controller']['max_twist'],
            pitch_control=config['task'].get('pitch_control'),
            agent_control_dt=t['environment']['agent_control_dt'])
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
        self.reward_depth_span = (gap_from_pose(self.initial_pose,self.board_geometry)-self.stop_gap)[:,None]
        self.depth_task = config['task']['version'] in POSE_DEPTH_VERSIONS
        # pose_depth_v3 software board-stop consistency: post-latch samples report the
        # frozen reading (as the DiSECt adapter does), instead of contributing nothing.
        from cutting_gpu_backend.tool_mount import continuing_board, current_board
        self.board_continuing = continuing_board(config['task']['board'])
        self.board_current_continuation = current_board(config['task']['board'])
        self.board_latch_v3 = config['task']['version'] == 'pose_depth_v3' and not self.board_continuing
        if not bool((torch.isfinite(self.reward_depth_span) & (self.reward_depth_span > 0)).all()):
            raise ValueError('Reset blade must start above its board stopping height')
        self.target = self.initial_pose.clone()
        self.target[:,2] -= gap_from_pose(self.target,self.board_geometry)-self.stop_gap
        self.force_limits = torch.tensor(config['force_limits'],device=self.device,dtype=self.dtype)[None].repeat(self.worlds,1)
        self.desired_wrench = torch.tensor(t['controller']['desired_force_torque'],device=self.device,dtype=self.dtype)[None].expand(self.worlds,-1)
        self.batched = config.get('execution', 'serial') == 'batched'
        if config.get('execution', 'serial') not in ('serial','batched'):
            raise ValueError('Unknown BladeMaster execution mode')
        self.batch_groups = []
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
        if self.condition_sampler is None:
            speed=float(config['target_speed_m_s'])
            if not math.isfinite(speed) or speed<=0:raise ValueError('Target speed must be finite and positive')
            self.target_speed=torch.full((self.worlds,),speed,device=self.device,dtype=self.dtype)
        else:
            # The first reset samples every world's speed before any transition.
            self.target_speed=torch.zeros(self.worlds,device=self.device,dtype=self.dtype)
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

    def prescribed_blade_pose(self):
        """Material-frame blade pose at the task target; its local x axis is the prescribed
        cut-plane normal (BladeMaster side 1 is blade-local x < 0, batch_contact.py:464-467)."""
        return compose_pose(self.material_from_base,compose_pose(
            self.target,self.ee_from_blade.expand(self.worlds,-1)))

    def _new_material(self, index):
        kwargs={} if self.reference_points[index] is None else {'reference_points':self.reference_points[index]}
        solver = self.Solver(self.material_configs[index],device='cuda:0',**kwargs)
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

    def _reset_randomized_batched(self, mask):
        from blademaster_warp.batched import BatchedSolver
        if not self.batch_groups:
            offset=0
            for group in self.groups:
                count=group['worlds']
                solver=BatchedSolver(self.material_configs[offset],count,device='cuda:0',
                                     reference_points=self.reference_points[offset])
                if not hasattr(solver,'set_material_scales'):
                    raise ValueError('BladeMaster source lacks per-world material parameters')
                indices=slice(offset,offset+count)
                self.batch_groups.append((indices,solver))
                for local,index in enumerate(range(offset,offset+count)):
                    self.solvers[index]=MaterialWorldView(solver,local)
                offset+=count
        selected=mask.cpu().numpy()
        values=self.condition_sampler.values
        for indices,solver in self.batch_groups:
            local=selected[indices]
            if not local.any():
                continue
            solver.reset(local,raw=True)
            solver.set_material_scales(local,
                youngs_modulus=values['youngs_modulus'][indices],
                density=values['density'][indices],
                cutting_energy=values['cut_strength'][indices])
            pose=self.tool_pose()[indices].cpu().numpy()
            packed=np.zeros((len(local),49),dtype=np.float32)
            packed[:,:7]=pose
            for _ in range(round(self.settle_seconds/.002)):
                out=solver.advance(packed,local.astype(np.int32),cutting=False).numpy()
                self._check_batch_result(out[local])
                if (np.linalg.norm(out[local,:3]/.002,axis=1)>1e-6).any():
                    raise ValueError('Blade touches randomized material during settling')
            solver.save_reset(mask=local)

    def _reset_batched(self, mask):
        if self.condition_sampler is not None:
            self._reset_randomized_batched(mask)
            return
        from blademaster_warp.batched import BatchedSolver
        if not self.batch_groups:
            offset=0
            for group in self.groups:
                count=group['worlds']
                solver=BatchedSolver(self.material_configs[offset],count,device='cuda:0',
                    reference_points=self.reference_points[offset])
                indices=slice(offset,offset+count)
                pose=self.tool_pose()[indices].cpu().numpy()
                packed=np.zeros((count,49),dtype=np.float32);packed[:,:7]=pose
                for _ in range(round(self.settle_seconds/.002)):
                    out=solver.advance(packed,np.ones(count,dtype=np.int32),cutting=False).numpy()
                    if (np.linalg.norm(out[:,:3]/.002,axis=1)>1e-6).any():
                        raise ValueError('Blade touches material during batched settling')
                    self._check_batch_result(out)
                solver.save_reset()
                self.batch_groups.append((indices,solver))
                for local,index in enumerate(range(offset,offset+count)):
                    self.solvers[index]=MaterialWorldView(solver,local)
                offset+=count
            if offset!=self.worlds:raise ValueError('Grouped BladeMaster world count mismatch')
        else:
            masks=mask.cpu().numpy()
            for indices,solver in self.batch_groups:
                if masks[indices].any():solver.reset(masks[indices])

    @staticmethod
    def _check_batch_result(out):
        if not np.isfinite(out).all() or (out[:,6:9]!=0).any():
            raise FloatingPointError('Invalid batched BladeMaster material interval')
        if (out[:,-1]>1e-6).any():
            raise FloatingPointError('Batched blade grid constraint failed to converge')

    def _advance_batched(self,pose,linear,angular,active):
        packed=torch.zeros((self.worlds,49),device=self.device,dtype=torch.float32)
        packed[:,:7]=pose
        packed[:,7:10]=linear
        packed[:,10:13]=angular
        host=packed.cpu().numpy();enabled=active.cpu().numpy()
        for indices,solver in self.batch_groups:
            selected=enabled[indices]
            if not selected.any():continue
            out=solver.advance(host[indices],selected).numpy()
            self._check_batch_result(out[selected])
            wrench=torch.as_tensor(out[:,:6]/.002,device=self.device,dtype=self.dtype)
            self.net[indices]=torch.where(active[indices,None],wrench,self.net[indices])

    @torch.no_grad()
    def reset(self, mask=None):
        if mask is None: mask=torch.ones(self.worlds,device=self.device,dtype=torch.bool)
        if self.condition_sampler is not None:
            values=self.condition_sampler.sample(mask.cpu().numpy())
            self.target_speed[mask]=torch.as_tensor(values['target_speed'],device=self.device,
                                                    dtype=self.dtype)[mask]
        self.robot.reset(mask)
        for k,v in self._controller_initial.items():getattr(self.controller,k)[mask]=v[mask]
        for k,v in self._filter_initial.items():getattr(self.filter,k)[mask]=v[mask]
        self.mapper.added_motion_command[mask]=0
        self.sim_time[mask]=0;self.peak[mask]=0;self.net[mask]=0
        self.feedback['sensor_wrench'][mask]=0;self.wrench_queue[mask]=0
        self.contacted[mask]=False;self.progress[mask]=0
        self.frames=self.chain(self.robot.q.double(),mass_matrix=False)[0]
        if self.batched:
            self._reset_batched(mask)
        else:
            for index in mask.nonzero().flatten().tolist():self._new_material(index)
        return self.frame()

    def frame(self):
        pose=self.frames[self.chain.tip]
        gap=gap_from_pose(pose,self.board_geometry)
        return dict(joint_q=self.robot.q.clone(),joint_qd=self.robot.qd.clone(),
            target_pose=self.target.clone(),ee_pose=pose.clone(),
            wrench_history=self.wrench_queue[:,:self.task['environment']['wrench_hist_size']].clone(),
            peak_wrench=self.peak.clone(),motion_command=self.mapper.added_motion_command.clone(),
            cut_progress=self.progress.clone(),depth_span=self.reward_depth_span.clone(),knife_board_gap=gap[:,None],
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
            active=~collision if getattr(self,'board_continuing',False) else ~(collision|self.contacted)
            if not bool(active.any()):break
            old_tool=self.tool_pose()
            for _ in range(2):
                moving=active if getattr(self,'board_continuing',False) else active & ~self.contacted
                if not bool(moving.any()):break
                old_q=self.robot.q.double().clone()
                command=self.controller_command(arguments,moving)
                self.robot.step(command,active=moving)
                new_q=self.robot.q.double().clone()
                fk=lambda q:self.chain(q,mass_matrix=False)[0][self.chain.tip]
                continuing=getattr(self,'board_continuing',False)
                if continuing:
                    from cutting_gpu_backend.tool_mount import continuing_joint_step
                    corrected,hit=continuing_joint_step(old_q,new_q,fk,self.board_geometry,self.reach_bound,moving)
                else:
                    fraction,hit=swept_joint_fraction(old_q,new_q,fk,self.board_geometry,
                                                       self.reach_bound,moving)
                if bool(hit.any()):
                    # Robot uses float32; retain positive numerical clearance.
                    if not continuing:
                        alpha=(fraction-1e-5).clamp_min(0)
                        corrected=torch.where(hit[:,None],old_q+alpha[:,None]*(new_q-old_q),new_q)
                    if continuing:
                        from cutting_gpu_backend.tool_mount import project_joint_velocity
                        velocity=project_joint_velocity(corrected,self.robot.qd.double(),fk,self.board_geometry,hit)
                        self.robot.set_joint_positions(corrected,mask=hit,velocity=velocity)
                    else:
                        self.robot.set_joint_positions(corrected,mask=hit)
                self.frames=self.chain(self.robot.q.double(),mass_matrix=False)[0]
                gap=gap_from_pose(self.frames[self.chain.tip],self.board_geometry)
                if continuing:
                    # State property of the end pose (tool_mount.board_contact), as in DiSECt and
                    # the WM; project_continue_v1 (evaluate-only) keeps its projected-AND-gap rule.
                    from cutting_gpu_backend.tool_mount import board_contact
                    current=board_contact(gap) if getattr(self,'board_current_continuation',True) else hit & (gap<=1e-5)
                    self.contacted=torch.where(moving,current,self.contacted)
                else:
                    self.contacted |= hit & (gap<=1e-5)
            tool_pose=self.tool_pose()
            linear=(tool_pose[:,:3]-old_tool[:,:3])/.002
            angular=quat_log(quat_mul(tool_pose[:,3:],quat_inverse(old_tool[:,3:])))/.002
            if self.batched:
                self._advance_batched(tool_pose,linear,angular,active)
            else:
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
            if not self.depth_task:
                self.progress=geometric_progress(self.frames[self.chain.tip],self.board_geometry,
                                                 self.depth_span,self.progress)
            if sample_callback is not None:sample_callback(self,active)
        if getattr(self,'board_latch_v3',False):
            from .pose_depth import latched_interval_peak
            held=self.contacted&~collision
            self.peak,collision=latched_interval_peak(self.peak,self.feedback['sensor_wrench'],
                                                      self.contacted,collision,self.force_limits)
            # Safety margin reads the same samples as peak (DiSECt v3 does the same).
            self.metrics.check_wrench(self.feedback['sensor_wrench'].abs(),self.force_limits,held)
        if bool((gap_from_pose(self.frames[self.chain.tip],self.board_geometry)<-2e-6).any()):
            raise FloatingPointError('BladeMaster knife penetrated the board')
        if self.depth_task:
            from .pose_depth import depth_progress
            self.progress=depth_progress(self.frames[self.chain.tip],self.board_geometry,
                                         self.reward_depth_span,self.progress)
        self.step_metrics=self.metrics.as_dict()
        return self.frame(),collision,peak_force

    def close(self):
        self.ports.clear();self.solvers.clear();self.batch_groups.clear()


class BladeMasterEnvironment(GPUEnvironment):
    @torch.no_grad()
    def __init__(self, backend_root, config):
        if config['task']['version'] not in ('blademaster_pose_v1',) + POSE_DEPTH_VERSIONS:
            raise ValueError('BladeMaster needs its own task identity')
        validate_task(config['task'])
        self.physics=BladeMasterBatch(backend_root,config)
        self.worlds,self.device=self.physics.worlds,self.physics.device
        self.condition_sampler=self.physics.condition_sampler
        from types import SimpleNamespace
        group_names=list(dict.fromkeys(self.physics.group_names))
        self.groups=[SimpleNamespace(name=n) for n in group_names]
        self.group_index=torch.tensor(self.physics.group_index,device=self.device)
        self.object_names=list(self.physics.group_names)
        self.metadata=copy.deepcopy(self.physics.task)
        self.metadata.update(physical_backend='mujoco_warp_blademaster',
            task_interface_revision=config['task']['version'],state_layout='coupled_recurrent_v1',
            material_feedback='blademaster_net_sensor_v1',reset_observation='post_reset',
            free_cut_task=validate_task(config['task']),goal_dim=8,
            board_pose_geometry=self.physics.board_geometry,
            progress_semantics='maximum_board_normal_blade_depth_v1',
            depth_span_m=self.physics.depth_span,
            blademaster=self.physics.material_metadata,
            backend_config=copy.deepcopy(config))
        if config['task']['version'] in POSE_DEPTH_VERSIONS:
            from .pose_depth import PROGRESS_SEMANTICS
            self.metadata['progress_semantics'] = PROGRESS_SEMANTICS
            self.metadata.pop('depth_span_m')
        if config['task'].get('tool_revision'):
            self.metadata['tool_revision']=config['task']['tool_revision']
        if self.condition_sampler is not None:
            from .episode_conditions import metadata
            self.metadata['episode_conditions']=metadata(self.condition_sampler.config)
        self.metadata=configure_force_observation(self.metadata,config.get('force_observation'))
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
        self.depth_span=self.physics.reward_depth_span.clone()
        from .scene_alignment import initial_clearances
        self.start_clearance_m=initial_clearances(config['task'], config.get('start_clearance_m'),
                                                self.worlds, self.device)
        # Physics initialization has already produced settled reset states.
        frame=self.physics.frame()
        from .scene_alignment import align_reset
        frame=align_reset(self,frame,torch.ones(self.worlds,device=self.device,dtype=torch.bool))
        self.previous_pose=self.fk(frame['joint_q'].double())
        self.depth_span=self.physics.reward_depth_span.clone()
        self.current=self._observe(frame,torch.zeros((self.worlds,6),device=self.device))

    def _observe(self, frame, action):
        saved=self.free_task
        self.free_task=None
        try:frame=GPUEnvironment._observe(self,frame,action)
        finally:self.free_task=saved
        frame['observation']=self.free_task.observation(frame['observation'],self.step_count)
        frame['target_speed']=self.target_speed.clone()
        if 'completion_position' in self.free_task.config:
            frame['completion_xy_error'] = self.free_task.completion_distance(frame['ee_pose'],frame['target_pose'])[:,None]
        if self.condition_sampler is not None:
            frame['condition_episode']=torch.as_tensor(
                self.condition_sampler.next_episode-1,device=self.device)[:,None].clone()
        if self.free_task.config['version'] == 'pose_depth_v3':
            from .pose_depth import observe_board_latch
            observe_board_latch(self,frame)
        return frame

    @torch.no_grad()
    def step(self, action, reference=False, sample_callback=None):
        if bool(self.needs_reset.any()):raise RuntimeError('Reset ended worlds before stepping')
        action=action.to(self.device).clamp(-1,1)
        frame,collision,force=self.physics.step(action,reference,sample_callback)
        self.step_count+=1
        frame=self._observe(frame,action)
        physical_progress = None if self.free_task.config['version'] in POSE_DEPTH_VERSIONS else frame['cut_progress'][:,0]
        events=self.free_task.events(frame['observation'],frame['peak_wrench'],physical_progress,
            self.step_count,collision,pose=frame['ee_pose'],board=frame,success_counter=self.success_counter,
            object_slip=(frame['object_slip'][:,0] if 'object_slip' in frame else None),
            target_pose=frame['target_pose'])
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
        if 'completion_xy_error_m' in events:
            info['completion_xy_error_m'] = events['completion_xy_error_m']
            info['tool_motion_failure'] = events['tool_motion_failure']
        if 'object_slip_failure' in events:
            info.update({k:events[k] for k in ('object_slip_failure','object_slip_m')})
        info.update(peak_force=force,depth_progress=frame['cut_progress'][:,0],**self.physics.step_metrics)
        # Native fracture is diagnostic only: do not relabel pose success as separation.
        if bool(events['terminated'].any()):
            from blademaster_warp.topology import components
            count=torch.full((self.worlds,),-1,dtype=torch.long,device=self.device)
            released=torch.full((self.worlds,),float('nan'),device=self.device)
            detailed=self.free_task.config['version']=='pose_depth_v3'
            if detailed:
                from .blademaster_separation import separation_diagnostics
                plane=self.physics.prescribed_blade_pose()
                normal=quat_apply(plane[:,3:],plane.new_tensor([1.,0.,0.]).expand(self.worlds,3))
                plane,normal=plane.cpu().numpy(),normal.cpu().numpy()
                diagnostics=[None]*self.worlds
            for index in events['terminated'].nonzero().flatten().tolist():
                solver=self.physics.solvers[index]
                codes,damage=solver.codes.numpy(),solver.damage.numpy()
                if detailed:
                    cfg=solver.config
                    diagnostics[index]=separation_diagnostics(
                        solver.node_numpy,solver.weight_numpy,codes,damage,solver.ref.numpy(),
                        plane[index,:3],normal[index],cfg.grid_spacing,cfg.cut_band_cells,
                        cut_index=solver.cut_index,components=components)
                    count[index]=len(diagnostics[index]['component_sizes'])
                else:
                    _,sizes=components(solver.node_numpy,solver.weight_numpy,codes)
                    count[index]=len(sizes)
                released[index]=float((damage>=1).mean())
            info.update(material_components=count,damage_saturated_fraction=released)
            if detailed:
                # Per-world dicts (None for running worlds); logging only, never reward/success/input.
                info['separation_diagnostics']=diagnostics
        return frame,reward.float(),events['terminated'],torch.zeros_like(events['terminated']),info

    @staticmethod
    def goal(frame):
        return torch.cat((frame['target_pose'],frame['target_speed'][:,None]),-1).float()

    def conditions(self, worlds):
        rows=[]
        for w in worlds:
            cfg=self.physics.material_configs[w]
            row=dict(group=self.object_names[w],target_speed_mps=float(self.target_speed[w]),
                     young_modulus_pa=cfg.young_modulus,density_kg_m3=cfg.density,
                     cutting_energy_j_m2=cfg.cutting_energy)
            if self.start_clearance_m is not None:
                row['start_clearance_m']=float(self.start_clearance_m[w])
            if self.condition_sampler is not None:
                values=self.condition_sampler.values
                row.update(condition_episode=int(self.condition_sampler.next_episode[w]-1),
                           youngs_modulus=float(values['youngs_modulus'][w]),
                           density=float(values['density'][w]),
                           cutting_energy_scale=float(values['cut_strength'][w]),
                           young_modulus_pa=cfg.young_modulus*values['youngs_modulus'][w],
                           density_kg_m3=cfg.density*values['density'][w],
                           cutting_energy_j_m2=cfg.cutting_energy*values['cut_strength'][w])
            rows.append(row)
        return rows

    def restore_frame(self, frame):
        raise NotImplementedError('BM frame lacks deformation/topology state; reset physical episodes on resume')

    def settle(self, seconds, mask=None):
        raise NotImplementedError('Set BladeMaster settle_seconds explicitly in the backend config')

    def close(self):
        self.physics.close()
