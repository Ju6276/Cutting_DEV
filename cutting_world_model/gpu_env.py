"""SliceIt task heads on the ROS-free packed UR5e/DiSECt backend."""
import torch

from cutting_gpu_backend.environment import CuttingBatch
from .kinematics import SerialKinematics, observation_velocity
from .task_heads import REWARD_KEYS, SliceItHeads, pose_error, TASK_INTERFACE_REVISION


class GPUEnvironment:
    def __init__(self, backend_root, worlds=1, reference_trajectory=None, reward=None, scene=None,
                 initial_lift=None, force_limit_scale=None):
        """``reward`` optionally selects the force penalty, e.g. {'force_penalty': 'linear',
        'force_penalty_scale': 50.}; None keeps the source SliceIt reward and metadata.
        ``scene`` is an exported DiSECt scene relative to ``backend_root``; None keeps the
        source prism. The scene is not part of the task metadata."""
        # initial_lift (m) and force_limit_scale are per-world task conditions (CuttingBatch);
        # with a force-limit scale the observation normalizes forces by each world's limits.
        scene = {} if scene is None else {'scene': scene}
        self.physics = CuttingBatch(backend_root, worlds, reference_trajectory=reference_trajectory,
                                    initial_lift=initial_lift, force_limit_scale=force_limit_scale, **scene)
        self.limit_scale = None if force_limit_scale is None else self.physics.force_limit_scale
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
        self.metadata['state_layout'] = 'coupled_recurrent_v1'
        self.metadata['material_feedback'] = 'tensor_filter_controller'
        self.metadata['reset_observation'] = 'post_reset'
        self.environment = self.metadata['environment']
        self.controller = self.metadata['controller']
        self.heads = SliceItHeads(self.environment, self.controller)
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
                                            action, frame['wrench_history'], self.limit_scale)
        if self.limit_scale is not None:
            frame['force_limit_scale'] = self.limit_scale[:, None].to(observation).clone()
        frame.update(ee_pose=pose, previous_ee_pose=pose, previous_velocity=velocity,
                     previous_acceleration=acceleration, observation=observation,
                     success_counter=self.success_counter[:, None].clone(),
                     step_count=self.step_count[:, None].clone())
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
    def step(self, action, reference=False):
        if self.needs_reset.any():
            raise RuntimeError('Reset completed worlds before the next batched action')
        action = action.to(self.device)
        low, high = action.new_tensor(self.metadata['action_bounds'])
        action = torch.maximum(torch.minimum(action, high), low)
        frame, collision, peak_force = self.physics.step(action, reference=reference)
        self.step_count += 1
        frame = self._observe(frame, action)
        terminated, success, outside, counter = self.heads.termination(
            frame['observation'], self.success_counter, collision)
        success = success & ~collision
        if self.environment['termination_on_negative_reward']:
            success.zero_()
        reward = self.heads.reward(frame['observation'], collision, success, outside, self.limit_scale,
                                   self.target_speed)
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
        }
