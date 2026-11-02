"""Batched UR5e, source FDCC and DiSECt transitions at the SliceIt control periods."""
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from .batched_disect import BatchedDiSECt
from .controller.fdcc import ControllerConfig, FDCCController, SerialChain, SlicingActionMapper, quat_apply
from .coupling import SliceItCoupling
from .robot import Robot


class CuttingBatch:
    @torch.no_grad()
    def __init__(self, root, worlds=1, scene='runs/disect_source_v4/scene.pt'):
        root = Path(root)
        self.worlds = worlds
        self.device = torch.device('cuda:0')
        self.dtype = torch.float64
        wp.config.kernel_cache_dir = '/dataHDD/dong/.cache/warp/cutting_gpu_backend'
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
        self.force_limits = torch.tensor(task['controller']['max_force_torque'], device=self.device, dtype=self.dtype)
        self.desired_wrench = torch.tensor(task['controller']['desired_force_torque'], device=self.device, dtype=self.dtype)[None].expand(worlds, -1)
        self.motion_initialized = torch.zeros(worlds, device=self.device, dtype=torch.bool)
        self.sim_time = torch.zeros(worlds, device=self.device, dtype=self.dtype)
        self.peak = torch.zeros_like(self.desired_wrench)
        self.robot.save_reset_state()
        self.material.save_reset_state()
        self._initial_controller = {k: v.clone() for k, v in vars(self.controller).items() if isinstance(v, torch.Tensor)}
        self._initial_filter = {k: getattr(self.coupling.wrench_filter, k).clone() for k in ('history', 'zi', 'offset')}
        self._initial_feedback = {k: v.clone() for k, v in self.feedback.items()}
        self._initial_queue = self.wrench_queue.clone()
        self.spring_reference = self.material.state_field('cut_spring_ke').clone()

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
        self.sim_time[mask] = 0
        self.peak[mask] = 0
        self.frames = self.chain(self.robot.q.to(self.dtype), mass_matrix=False)[0]
        return self.frame()

    def frame(self):
        return {
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

    @torch.no_grad()
    def step(self, action, reference=False):
        action = action.to(device=self.device, dtype=self.dtype)
        if action.shape != (self.worlds, 6):
            raise ValueError('Expected one six-component action per environment')
        self.peak.zero_()
        if reference:
            velocity = quat_apply(self.coupling.base_from_disect[3:], self.material.reference_linear_velocity.to(self.dtype))
            target = self.reference_start.clone()
            dt = self.task['environment']['agent_control_dt']
            target[:, :3] += velocity * (self.sim_time + dt).clamp(max=self.material.reference_duration)[:, None]
            args = {'target_pose': target}
        else:
            args = self.mapper(action, self.target, self.frames[self.chain.tip])
        collision = torch.zeros(self.worlds, dtype=torch.bool, device=self.device)
        terminal = None
        peak_force = torch.zeros(self.worlds, device=self.device)
        intervals = round(self.task['environment']['agent_control_dt'] / .002)
        for _ in range(intervals):
            observed = self.feedback['observation_wrench'].abs()
            self.peak = torch.where(collision[:, None], self.peak, torch.maximum(self.peak, observed))
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
            for _ in range(2):
                command = self.controller.update(self.robot.q.to(self.dtype),
                    sensor_wrench=self.feedback['sensor_wrench'], target_wrench=self.desired_wrench, **args)
                self.robot.step(self.controller.enforce_position_limits(command, .001))
            self.frames = self.chain(self.robot.q.to(self.dtype), mass_matrix=False)[0]
            knife = self.knife_chain(self.robot.q.to(self.dtype), mass_matrix=False)[0]['b_bot_knife_sim']
            motion = self.coupling.motion(knife)
            twist = torch.cat((motion['linear_velocity'], motion['angular_velocity']), -1)
            twist[~self.motion_initialized] = 0
            self.motion_initialized[:] = True
            self.material.set_motion(twist.float())
            self.material.advance(.002)
            raw = -self.robot.sensor_wrench(self.config.sensor_ref_link).to(self.dtype)
            self.feedback = self._feedback(raw)
            self.wrench_queue = torch.cat((self.wrench_queue[:, 1:], self.feedback['sensor_wrench'][:, None]), 1)
            peak_force = torch.where(collision, peak_force, torch.maximum(peak_force, self.feedback['force_magnitude'].float()))
            self.sim_time += (~collision) * .002
        frame = self.frame()
        if terminal is not None:
            for k in frame:
                frame[k][collision] = terminal[k][collision]
        return frame, collision, peak_force
