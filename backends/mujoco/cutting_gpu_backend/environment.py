"""Batched UR5e, source FDCC and DiSECt transitions at the SliceIt control periods."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
import warp as wp

from .batched_disect import BatchedDiSECt
from .controller.fdcc import (ControllerConfig, FDCCController, SerialChain, SlicingActionMapper,
                              pose_error, quat_apply)
from .coupling import SliceItCoupling
from .metrics import StepMetrics
from .reference import KnifeTrajectory
from .robot import Robot


class CuttingBatch:
    @torch.no_grad()
    def __init__(self, root, worlds=1, scene='runs/disect_source_v4/scene.pt', reference_trajectory=None,
                 initial_lift=None):
        """``initial_lift`` (m, per world): start the knife this far above the captured start
        along DiSECt +y (vertical), same orientation, by inverse kinematics before the reset
        state is settled and saved."""
        root = Path(root)
        self.worlds = worlds
        self.scene = scene
        self.device = torch.device('cuda:0')
        self.dtype = torch.float64
        wp.config.kernel_cache_dir = os.environ.get(
            'WARP_CACHE_PATH', str(root / 'runs/cache/warp'))
        self.task = json.loads((root / 'runs/source/task.json').read_text())
        episode = np.load(root / 'runs/source/episode_000.npz')
        task = self.task
        self.material = BatchedDiSECt(root / scene, worlds)
        self.robot = Robot(root / 'runs/model_coupled/robot.mjb', episode['joint_q'][0], episode['joint_qd'][0], worlds)
        self.config = ControllerConfig.from_parameters(json.loads((root / 'runs/controller_source/controller_params.json').read_text()))
        self.chain = SerialChain(task['robot_xml'], task['robot_base'], task['robot_tip'], task['joint_names']).to(self.device, self.dtype)
        self.knife_chain = SerialChain(task['robot_xml'], task['robot_base'], 'b_bot_knife_sim', task['joint_names']).to(self.device, self.dtype)
        self.controller = FDCCController(self.chain, self.config)
        self.mapper = SlicingActionMapper(max_twist=task['controller']['max_twist'])
        self.mapper.reset(worlds, self.device, self.dtype)
        self.coupling = SliceItCoupling.from_urdf(task['robot_xml'], self.material.spine_height, device=self.device, dtype=self.dtype)
        self.initial_lift = torch.zeros(worlds, device=self.device, dtype=self.dtype)
        if initial_lift is not None:
            self.initial_lift = torch.as_tensor(initial_lift, device=self.device, dtype=self.dtype).expand(worlds).clone()
            self.robot.set_joint_positions(self.lifted_configuration(self.robot.q.to(self.dtype), self.initial_lift))
        raw = -self.robot.sensor_wrench(self.config.sensor_ref_link).to(self.dtype)
        self.coupling.reset_filter(raw)
        hold = self.robot.q.clone()
        for _ in range(self.coupling.wrench_filter.window):
            self.robot.step(hold)
            self.robot.step(hold)
            raw = -self.robot.sensor_wrench(self.config.sensor_ref_link).to(self.dtype)
            self.coupling.wrench_filter.update(raw)
        self.coupling.wrench_filter.tare()
        self.controller.reset(self.robot.q.to(self.dtype), self.robot.qd.to(self.dtype))
        self.frames = self.chain(self.robot.q.to(self.dtype), mass_matrix=False)[0]
        knife = self.knife_chain(self.robot.q.to(self.dtype), mass_matrix=False)[0]['b_bot_knife_sim']
        self.material.set_motion(torch.zeros((worlds, 6), device=self.device), self.coupling.reset_pose(knife).float())
        self.material.advance(.002)
        self.feedback = self._feedback(raw)
        self.wrench_queue = self.feedback['sensor_wrench'][:, None].repeat(1, 25, 1)
        self.target = torch.as_tensor(episode['target_pose'][0], device=self.device, dtype=self.dtype)[None].repeat(worlds, 1)
        self.reference_start = self.frames[self.chain.tip].clone()
        self.set_reference_trajectory(reference_trajectory)
        self.force_limits = torch.tensor(task['controller']['max_force_torque'], device=self.device,
                                         dtype=self.dtype)[None].repeat(worlds, 1)
        self.desired_wrench = torch.tensor(task['controller']['desired_force_torque'], device=self.device, dtype=self.dtype)[None].expand(worlds, -1)
        self.coupling.previous_pose_base = knife.clone()
        self.motion_initialized = torch.zeros(worlds, device=self.device, dtype=torch.bool)
        self.sim_time = torch.zeros(worlds, device=self.device, dtype=self.dtype)
        self.peak = torch.zeros_like(self.desired_wrench)
        self.metrics = StepMetrics(worlds, self.device, self.dtype)
        self.step_metrics = self.metrics.as_dict()
        self.robot.save_reset_state()
        self.material.save_reset_state()
        self._initial_controller = {k: v.clone() for k, v in vars(self.controller).items() if isinstance(v, torch.Tensor)}
        self._initial_filter = {k: getattr(self.coupling.wrench_filter, k).clone() for k in ('history', 'zi', 'offset')}
        self._initial_feedback = {k: v.clone() for k, v in self.feedback.items()}
        self._initial_queue = self.wrench_queue.clone()
        self._initial_previous_knife = knife.clone()
        self.spring_reference = self.material.state_field('cut_spring_ke').clone()

    @torch.no_grad()
    def lifted_configuration(self, q, lift, iterations=50, tolerance=1e-7):
        """Joint positions placing the knife ``lift`` metres higher along DiSECt +y, same orientation."""
        tip = 'b_bot_knife_sim'
        start = self.knife_chain(q, mass_matrix=False)[0][tip]
        up = quat_apply(self.coupling.base_from_disect[3:].to(q).expand(len(q), -1),
                        q.new_tensor([0., 1., 0.]).expand(len(q), -1))
        target = start.clone()
        target[:, :3] += lift[:, None].to(q) * up
        for _ in range(iterations):
            frames, jacobian, _ = self.knife_chain(q, mass_matrix=False)
            error = pose_error(frames[tip], target)
            if float(error.abs().max()) < tolerance:
                break
            jjt = jacobian @ jacobian.transpose(-1, -2) + 1e-10 * torch.eye(6, device=q.device, dtype=q.dtype)
            q = q + (jacobian.transpose(-1, -2) @ torch.linalg.solve(jjt, error[..., None])).squeeze(-1)
        else:
            raise RuntimeError('Inverse kinematics for the lifted start did not converge')
        return q

    def set_reference_trajectory(self, trajectory=None):
        """None restores the scene's constant knife velocity over its duration."""
        if trajectory is None:
            trajectory = KnifeTrajectory.constant(
                self.material.reference_linear_velocity.reshape(-1, 3)[0].to(self.dtype),
                self.material.reference_duration, kind='scene_constant_velocity')
        self.reference_trajectory = trajectory

    def reference_target(self, time):
        """Controller target after ``time`` seconds of the DiSECt-frame knife trajectory."""
        target = self.reference_start.clone()
        target[:, :3] += self.reference_trajectory.displacement(time, self.coupling.base_from_disect[3:])
        return target

    def _feedback(self, raw):
        return self.coupling.feedback(self.material.state_field('knife_f').to(self.dtype),
                                     self.frames[self.config.sensor_ref_link], self.frames[self.chain.tip], raw)

    @torch.no_grad()
    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.worlds, dtype=torch.bool, device=self.device)
        self.robot.reset(mask)
        self.material.reset(mask)
        for k, initial in self._initial_controller.items():
            getattr(self.controller, k)[mask] = initial[mask]
        for k, initial in self._initial_filter.items():
            getattr(self.coupling.wrench_filter, k)[mask] = initial[mask]
        for k, initial in self._initial_feedback.items():
            self.feedback[k][mask] = initial[mask]
        self.wrench_queue[mask] = self._initial_queue[mask]
        self.mapper.added_motion_command[mask] = 0
        self.motion_initialized[mask] = False
        self.coupling.previous_pose_base[mask] = self._initial_previous_knife[mask]
        self.sim_time[mask] = 0
        self.peak[mask] = 0
        self.frames = self.chain(self.robot.q.to(self.dtype), mass_matrix=False)[0]
        return self.frame()

    def frame(self):
        frame = {
            'particle_q': self.material.state_field('particle_q').clone(),
            'particle_qd': self.material.state_field('particle_qd').clone(),
            'cut_spring_ke': self.material.state_field('cut_spring_ke').clone(),
            'spring_reference_ke': self.spring_reference.clone(),
            'joint_q': self.robot.q.clone(), 'joint_qd': self.robot.qd.clone(),
            'ee_pose': self.frames[self.chain.tip].clone(),
            'knife_pose': torch.cat((self.material.position, self.material.rotation), -1).clone(),
            'knife_velocity': torch.cat((self.material.linear_velocity, self.material.angular_velocity), -1).clone(),
            'wrench_history': self.wrench_queue[:, :self.task['environment']['wrench_hist_size']].clone(),
            'peak_wrench': self.peak.clone(),
            'motion_command': self.mapper.added_motion_command.clone(),
            'target_pose': self.target.clone(),
        }
        for name in ('q', 'velocity', 'previous_error', 'previous_position_command', 'cached_tip_pose', 'tare_compensation'):
            frame['controller_' + name] = getattr(self.controller, name).clone()
        for name in ('history', 'zi', 'offset'):
            frame['filter_' + name] = getattr(self.coupling.wrench_filter, name).clone()
        frame['wrench_queue'] = self.wrench_queue.clone()
        frame['robot_warmstart'] = wp.to_torch(self.robot.data.qacc_warmstart).clone()
        frame['coupling_previous_pose'] = self.coupling.previous_pose_base.clone()
        frame['coupling_initialized'] = self.motion_initialized[:, None].float().clone()
        frame['robot_time'] = wp.to_torch(self.robot.data.time)[:, None].clone()
        frame['sim_time'] = self.sim_time[:, None].clone()
        return frame




    @torch.no_grad()
    def restore_frame(self, frame):
        for name in ('particle_q', 'particle_qd', 'cut_spring_ke'):
            self.material.state_field(name).copy_(frame[name])
        self.material.position.copy_(frame['knife_pose'][:, :3])
        self.material.rotation.copy_(frame['knife_pose'][:, 3:])
        self.material.set_motion(frame['knife_velocity'])
        self.robot.q.copy_(frame['joint_q'])
        self.robot.qd.copy_(frame['joint_qd'])
        wp.to_torch(self.robot.data.qacc_warmstart).copy_(frame['robot_warmstart'])
        wp.to_torch(self.robot.data.time).copy_(frame['robot_time'][:, 0])
        for name in ('q', 'velocity', 'previous_error', 'previous_position_command', 'cached_tip_pose', 'tare_compensation'):
            getattr(self.controller, name).copy_(frame['controller_' + name])
        for name in ('history', 'zi', 'offset'):
            getattr(self.coupling.wrench_filter, name).copy_(frame['filter_' + name])
        self.wrench_queue.copy_(frame['wrench_queue'])
        self.coupling.previous_pose_base.copy_(frame['coupling_previous_pose'])
        self.motion_initialized.copy_(frame['coupling_initialized'][:, 0].bool())
        self.mapper.added_motion_command.copy_(frame['motion_command'])
        self.sim_time.copy_(frame['sim_time'][:, 0])
        self.peak.copy_(frame['peak_wrench'])
        self.frames = self.chain(self.robot.q.to(self.dtype), mass_matrix=False)[0]
        self.feedback['sensor_wrench'] = self.wrench_queue[:, -1].clone()
        from .coupling import source_wrench_transform
        self.feedback['observation_wrench'] = source_wrench_transform(
            self.feedback['sensor_wrench'], self.frames[self.chain.tip])

    def controller_command(self, arguments, active):
        if bool(active.all()):
            command = self.controller.update(self.robot.q.to(self.dtype),
                sensor_wrench=self.feedback['sensor_wrench'], target_wrench=self.desired_wrench, **arguments)
            return self.controller.enforce_position_limits(command, .001)
        selected = copy.copy(self.controller)
        names = [k for k, v in vars(self.controller).items() if isinstance(v, torch.Tensor)]
        for k in names:
            setattr(selected, k, getattr(self.controller, k)[active].clone())
        args = {k: v[active] if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == self.worlds else v
                for k, v in arguments.items()}
        command = selected.update(self.robot.q[active].to(self.dtype),
            sensor_wrench=self.feedback['sensor_wrench'][active], target_wrench=self.desired_wrench[active], **args)
        command = selected.enforce_position_limits(command, .001)
        result = self.robot.q.clone()
        result[active] = command.to(result.dtype)
        for k in names:
            getattr(self.controller, k)[active] = getattr(selected, k)
        return result

    def capture_recurrence(self, mask):
        return (
            {k: v[mask].clone() for k, v in vars(self.controller).items() if isinstance(v, torch.Tensor)},
            {k: getattr(self.coupling.wrench_filter, k)[mask].clone() for k in ('history', 'zi', 'offset')},
            {k: v[mask].clone() for k, v in self.feedback.items()},
            self.wrench_queue[mask].clone(), self.coupling.previous_pose_base[mask].clone(),
            self.motion_initialized[mask].clone(),
        )

    def restore_recurrence(self, mask, saved):
        controller, filt, feedback, queue, pose, initialized = saved
        for k, value in controller.items():
            getattr(self.controller, k)[mask] = value
        for k, value in filt.items():
            getattr(self.coupling.wrench_filter, k)[mask] = value
        for k, value in feedback.items():
            self.feedback[k][mask] = value
        self.wrench_queue[mask] = queue
        self.coupling.previous_pose_base[mask] = pose
        self.motion_initialized[mask] = initialized

    @torch.no_grad()
    def step(self, action, reference=False):
        action = action.to(device=self.device, dtype=self.dtype)
        if action.shape != (self.worlds, 6):
            raise ValueError('Expected one six-component action per environment')
        self.peak.zero_()
        self.metrics.zero_()
        if reference:
            dt = self.task['environment']['agent_control_dt']
            args = {'target_pose': self.reference_target(self.sim_time + dt)}
        else:
            args = self.mapper(action, self.target, self.frames[self.chain.tip])
        collision = torch.zeros(self.worlds, dtype=torch.bool, device=self.device)
        terminal = None
        peak_force = torch.zeros(self.worlds, device=self.device)
        intervals = round(self.task['environment']['agent_control_dt'] / .002)
        for _ in range(intervals):
            observed = self.feedback['observation_wrench'].abs()
            self.peak = torch.where(collision[:, None], self.peak, torch.maximum(self.peak, observed))
            self.metrics.check_wrench(observed, self.force_limits, ~collision)
            newly_done = (~collision) & (observed > self.force_limits).any(-1)
            if newly_done.any():
                current = self.frame()
                if terminal is None:
                    terminal = current
                else:
                    for k in current:
                        terminal[k][newly_done] = current[k][newly_done]
                collision |= newly_done
            if collision.all():
                break
            active = ~collision
            paused = self.capture_recurrence(collision) if collision.any() else None
            for _ in range(2):
                command = self.controller_command(args, active)
                self.robot.step(command, active=active)
            self.frames = self.chain(self.robot.q.to(self.dtype), mass_matrix=False)[0]
            knife = self.knife_chain(self.robot.q.to(self.dtype), mass_matrix=False)[0]['b_bot_knife_sim']
            motion = self.coupling.motion(knife)
            twist = torch.cat((motion['linear_velocity'], motion['angular_velocity']), -1)
            twist[~self.motion_initialized] = 0
            self.motion_initialized[:] = True
            self.material.set_motion(twist.float(), active=active)
            self.material.advance(.002, active=active)
            raw = -self.robot.sensor_wrench(self.config.sensor_ref_link).to(self.dtype)
            self.feedback = self._feedback(raw)
            self.metrics.accumulate(self.feedback['force_magnitude'], twist[:, :3], active, .002)
            self.wrench_queue = torch.cat((self.wrench_queue[:, 1:], self.feedback['sensor_wrench'][:, None]), 1)
            peak_force = torch.where(collision, peak_force, torch.maximum(peak_force, self.feedback['force_magnitude'].float()))
            self.sim_time += active * .002
            if paused is not None:
                self.restore_recurrence(collision, paused)
        frame = self.frame()
        if terminal is not None:
            for k in frame:
                frame[k][collision] = terminal[k][collision]
        # Read by callers after step(); the return signature is unchanged.
        self.step_metrics = self.metrics.as_dict()
        return frame, collision, peak_force
