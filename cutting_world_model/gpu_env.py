"""SliceIt task heads on the ROS-free packed UR5e/DiSECt backend."""
import torch

from cutting_gpu_backend.environment import CuttingBatch
from .kinematics import SerialKinematics, observation_velocity
from .task_heads import REWARD_KEYS, SliceItHeads, pose_error, TASK_INTERFACE_REVISION, validate_force_limits


class GPUEnvironment:
    def __init__(self, backend_root, worlds=1, reference_trajectory=None, reward=None, scene=None,
                 initial_lift=None, force_limits=None, sim_dt=None, force_model='scalar',
                 friction_velocity=None, settle_seconds=0., task=None):
        """``reward`` optionally selects the force penalty, e.g. {'force_penalty': 'linear',
        'force_penalty_scale': 50.}; None keeps the source SliceIt reward and metadata.
        ``scene`` is an exported DiSECt scene relative to ``backend_root``; None keeps the
        source prism. The scene is not part of the task metadata."""
        # initial_lift (m, per world): knife start height (CuttingBatch). force_limits (6 values)
        # replaces the task's force/torque limits for every world (one hardware limit).
        # sim_dt (s): DiSECt material substep of this scene, None keeps the exported one.
        # force_model / friction_velocity: material force signal and knife friction (CuttingBatch).
        from .free_cut import validate_task, FreeCutTask, validate_topology
        task = None if task is None else validate_task(task)
        if task is not None and (force_model != 'net' or reward):
            raise ValueError('free_cut requires net feedback and its own reward settings')
        force_limits = validate_force_limits(force_limits, required=task is not None)
        scene = {} if scene is None else {'scene': scene}
        self.physics = CuttingBatch(backend_root, worlds, reference_trajectory=reference_trajectory,
                                    initial_lift=initial_lift, sim_dt=sim_dt, force_model=force_model,
                                    friction_velocity=friction_velocity, settle_seconds=settle_seconds, **scene)
        if force_limits is not None:
            limits = force_limits
            self.physics.task = {**self.physics.task, 'controller': {**self.physics.task['controller'],
                                                                     'max_force_torque': limits}}
            self.physics.force_limits = torch.tensor(limits, device=self.physics.device,
                                                     dtype=self.physics.dtype)[None].repeat(worlds, 1)
        self.target_speed = None  # per-world target cutting speed (m/s), set by a task sampler
        self.worlds, self.device = worlds, self.physics.device
        self.metadata = dict(self.physics.task)
        if reward:
            unknown = set(reward) - set(REWARD_KEYS)
            if unknown:
                raise ValueError('Unknown reward settings: ' + ', '.join(sorted(unknown)))
            self.metadata['environment'] = {**self.metadata['environment'], **reward}
        self.metadata['task_interface_revision'] = TASK_INTERFACE_REVISION
        self.metadata['physical_backend'] = 'mujoco_warp_disect'
        self.metadata['contact_discretization_revision'] = self.physics.material.contact_discretization_revision
        self.metadata['state_layout'] = 'coupled_recurrent_v1'
        self.metadata['material_feedback'] = 'tensor_filter_controller'
        if force_model != 'scalar' or friction_velocity is not None:
            self.metadata['material_contact'] = {
                'revision': 'contact_wrench_v1',
                'force_model': force_model,
                'friction_velocity_override_mps': friction_velocity,
                'effective_friction_velocity_mps': sorted(
                    self.physics.material.model.sdf_kf.detach().cpu().unique().tolist()),
            }
        if force_model == 'net':
            self.metadata['material_feedback'] = 'net_contact_sensor_v1'
        if settle_seconds:
            self.metadata['settle_seconds'] = float(settle_seconds)
        self.metadata['reset_observation'] = 'post_reset'
        self.environment = self.metadata['environment']
        self.controller = self.metadata['controller']
        self.heads = SliceItHeads(self.environment, self.controller)
        self.free_task = None
        if task is not None:
            validate_topology(self.physics.material)
            self.metadata['free_cut_task'] = task
            self.metadata['task_interface_revision'] = task['version']
            self.metadata['goal_dim'] = 8
            self.free_task = FreeCutTask(task, self.heads)
            if task['version'] == 'free_cut_v4':
                self.physics.enable_board(task['board'])
                self.metadata['knife_board_revision'] = 'swept_stop_path_v1'
        m = self.metadata
        self.fk = SerialKinematics(m['robot_xml'], m['robot_base'], m['robot_tip'], m['joint_names']).to(self.device)
        self.previous_pose = self.fk(self.physics.robot.q.double())
        self.previous_velocity = torch.zeros((worlds, 6), device=self.device, dtype=torch.float64)
        self.previous_acceleration = torch.zeros_like(self.previous_velocity)
        self.success_counter = torch.zeros(worlds, device=self.device, dtype=torch.float64)
        self.step_count = torch.zeros(worlds, device=self.device, dtype=torch.long)
        self.needs_reset = torch.zeros(worlds, device=self.device, dtype=torch.bool)
        self.current = None
        self.reset()

    def _observe(self, frame, action):
        pose = self.fk(frame['joint_q'].double())
        dt = self.environment['agent_control_dt']
        velocity = observation_velocity(pose, self.previous_pose, dt)
        acceleration = (velocity - self.previous_velocity) / dt
        jerk = (acceleration - self.previous_acceleration) / dt
        observation = self.heads.observation(pose_error(frame['target_pose'], pose), velocity, jerk,
                                            action, frame['wrench_history'])
        if self.free_task is not None:
            observation = self.free_task.observation(observation, self.step_count)
        frame.update(ee_pose=pose, previous_ee_pose=pose, previous_velocity=velocity,
                     previous_acceleration=acceleration, observation=observation,
                     success_counter=self.success_counter[:, None].clone(),
                     step_count=self.step_count[:, None].clone())
        if self.free_task is not None:
            frame['cut_progress'] = (frame['cut_spring_ke'] == 0).to(pose.dtype).mean(-1, keepdim=True)
            if self.free_task.config['version'] == 'free_cut_v4':
                frame['separation_progress'] = frame['cut_progress'].clone()
                frame['cut_progress'] = frame['knife_path_progress'].to(pose.dtype)
        return frame

    @torch.no_grad()
    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.worlds, dtype=torch.bool, device=self.device)
        mask = mask.to(device=self.device, dtype=torch.bool)
        physical = self.physics.reset(mask)
        pose = self.fk(physical['joint_q'].double())
        self.previous_pose[mask] = pose[mask]
        self.previous_velocity[mask] = 0
        self.previous_acceleration[mask] = 0
        self.success_counter[mask] = 0
        self.step_count[mask] = 0
        self.needs_reset[mask] = False
        frame = self._observe(physical, torch.zeros((self.worlds, 6), device=self.device))
        if self.current is not None:
            for key in frame:
                frame[key][~mask] = self.current[key][~mask]
        self.current = frame
        return frame


    @torch.no_grad()
    def restore_frame(self, frame):
        self.physics.restore_frame(frame)
        self.previous_pose = frame['previous_ee_pose'].clone()
        self.previous_velocity = frame['previous_velocity'].clone()
        self.previous_acceleration = frame['previous_acceleration'].clone()
        self.success_counter = frame['success_counter'][:, 0].clone()
        self.step_count = frame['step_count'][:, 0].long().clone()
        self.needs_reset.zero_()
        self.current = {k: v.clone() for k, v in frame.items()}

    @torch.no_grad()
    def settle(self, seconds, mask=None):
        """Settle selected worlds from the export with their current material scales.
        All other physical and observation states remain unchanged."""
        if mask is None:
            mask = torch.ones(self.worlds, dtype=torch.bool, device=self.device)
        mask = mask.to(self.device, torch.bool)
        self.reset(mask)
        self.physics.settle(seconds, mask)
        self.metadata['settle_seconds'] = float(seconds)
        self.metadata['reset_material_protocol'] = 'export_per_material_settle_v1'
        return self.reset(mask)

    @torch.no_grad()
    def step(self, action, reference=False, sample_callback=None):
        if self.needs_reset.any():
            raise RuntimeError('Reset completed worlds before the next batched action')
        if self.free_task is not None and (self.target_speed is None or not bool(
                (torch.isfinite(self.target_speed) & (self.target_speed > 0)).all())):
            raise ValueError('free_cut requires finite positive target speeds')
        action = action.to(self.device)
        low, high = action.new_tensor(self.metadata['action_bounds'])
        action = torch.maximum(torch.minimum(action, high), low)
        diagnostics = {} if sample_callback is None else {'sample_callback': sample_callback}
        frame, collision, peak_force = self.physics.step(action, reference=reference, **diagnostics)
        self.step_count += 1
        frame = self._observe(frame, action)
        task_info = {}
        if self.free_task is not None:
            events = self.free_task.events(frame['observation'], frame['peak_wrench'],
                                           (frame['separation_progress'] if self.free_task.config['version'] == 'free_cut_v4'
                                            else frame['cut_progress'])[:, 0], self.step_count, collision, pose=frame['ee_pose'],
                                           board=frame if self.free_task.config['version'] == 'free_cut_v4' else None)
            reward = self.free_task.reward(self.current['cut_progress'][:, 0],
                                          frame['cut_progress'][:, 0], frame['observation'],
                                          frame['peak_wrench'], self.target_speed, events)
            terminated = events['terminated']
            success, outside, collision = events['success'], events['outside'], events['collision']
            counter = self.success_counter
            truncated = torch.zeros_like(terminated)
            task_info = {k: events[k] for k in ('separated', 'timeout')}
            task_info['cut_progress'] = frame['cut_progress'][:, 0]
            if self.free_task.config['version'] == 'free_cut_v4':
                task_info.update({k: events[k] for k in ('board_contact', 'board_incomplete', 'board_penetration')})
        else:
            terminated, success, outside, counter = self.heads.termination(
                frame['observation'], self.success_counter, collision)
            success = success & ~collision
            if self.environment['termination_on_negative_reward']:
                success.zero_()
            reward = self.heads.reward(frame['observation'], collision, success, outside, self.target_speed)
            truncated = (self.step_count >= self.environment['steps_per_episode']) & ~terminated
        self.success_counter = counter
        frame['success_counter'] = counter[:, None].clone()
        self.previous_pose = frame['previous_ee_pose'].clone()
        self.previous_velocity = frame['previous_velocity'].clone()
        self.previous_acceleration = frame['previous_acceleration'].clone()
        self.needs_reset = terminated | truncated
        self.current = frame
        return frame, reward.float(), terminated, truncated, {
            'success': success, 'collision': collision, 'outside': outside,
            'peak_force': peak_force, 'released_springs': (frame['cut_spring_ke'] == 0).sum(-1),
            **({'jerkiness': self.heads.jerkiness(frame['observation'])}
               if self.environment['jerkiness_as_obs'] else {}),
            **self.physics.step_metrics,
            **task_info,
        }
