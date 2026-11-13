"""Non-learning FDCC reference controllers for one free-cut physical world.

Uses the existing FDCC port and its exported fixed parameters, no model or MPC.
The reference is in DiSECt coordinates; execution remains closed-loop through
robot FK, sensor wrench filtering and the FDCC joint command update.
"""
from dataclasses import asdict
import math

import torch

from .benchmark import ReferencePolicy


def reference_trajectory(distance, speed, dt, horizon, saw=False, amplitude=.02, frequency=1.):
    """Stop vertical feed at the specified endpoint; hold thereafter, no retraction."""
    from cutting_gpu_backend.reference import KnifeTrajectory
    if (not all(math.isfinite(v) for v in (distance, speed, dt, amplitude, frequency))
            or distance < 0 or speed <= 0 or dt <= 0 or horizon < 1
            or amplitude < 0 or frequency <= 0):
        raise ValueError('Invalid finite reference trajectory parameters')
    times = torch.arange(horizon + 1, dtype=torch.float64) * dt
    stop = distance / speed
    if 0 < stop < float(times[-1]):
        times = torch.cat((times, times.new_tensor([stop]))).unique(sorted=True)
    moving_time = times.clamp(max=stop)
    displacement = torch.zeros(len(times), 3, dtype=torch.float64)
    displacement[:, 1] = -speed * moving_time
    if saw:
        displacement[:, 2] = amplitude * torch.sin(2 * math.pi * frequency * moving_time)
    velocity = (displacement[1:] - displacement[:-1]) / (times[1:] - times[:-1])[:, None]
    return KnifeTrajectory(times, velocity, {
        'method': 'fdcc-saw' if saw else 'fdcc-press',
        'distance_m': distance, 'feed_mps': speed,
        'saw_amplitude_m': amplitude if saw else 0.,
        'saw_frequency_hz': frequency if saw else 0.,
        'source': 'prescribed engineering reference; not a reproduced cutting paper controller',
    })


class FDCCReferencePolicy(ReferencePolicy):
    def __init__(self, env, saw=False, amplitude=.02, frequency=1.):
        if env.worlds != 1 or not env.metadata.get('free_cut_task'):
            raise ValueError('FDCC reference requires one free-cut world')
        # Validate trajectory arguments before starting physics.
        reference_trajectory(0., .01, .05, 1, saw, amplitude, frequency)
        self.env, self.saw, self.amplitude, self.frequency = env, saw, amplitude, frequency
        self.commands, self.trajectories = [], []
        self.episode = -1
        self.controller_parameters = asdict(env.physics.config)
        self.manifest = {
            'implementation': 'existing SliceIt/FZI FDCC tensor port',
            'upstream': 'https://github.com/cambel/cartesian_controllers',
            'upstream_revision': 'f4359b966bb3b8d8a07d829df463da814c2bee33',
            'origin': 'https://github.com/fzi-forschungszentrum-informatik/cartesian_controllers',
            'controller_parameters': self.controller_parameters,
            'learning': False, 'mpc': False, 'uses_simulator_cut_progress': False,
            'trajectory': 'press_plus_sine' if saw else 'press',
            'saw_amplitude_m': amplitude if saw else 0.,
            'saw_frequency_hz': frequency if saw else 0.,
            'target_speed': 'episode command', 'endpoint': 'task target height',
            'feedback': 'same measured sensor wrench and robot FK as learned-controller branch',
            'action_interface': 'FDCC target_pose; six recorded action placeholders are not commands',
            'completion': 'physical evaluator only; no progress-triggered control decisions',
        }

    def reset(self, frame):
        from cutting_gpu_backend.controller.fdcc import quat_apply
        physics = self.env.physics
        pose = frame['previous_ee_pose'].to(physics.reference_start)
        # Re-anchor every episode to the actual reset robot pose (including its lift).
        physics.reference_start.copy_(pose)
        rotation = physics.coupling.base_from_disect[3:]
        down = quat_apply(rotation[None], rotation.new_tensor([[0., -1., 0.]]))[0]
        if not bool(torch.allclose(down, down.new_tensor([0., 0., -1.]), atol=1e-5, rtol=0)):
            raise ValueError('This reference requires DiSECt down to map to robot-base -z')
        distance = float(pose[0, 2] - frame['target_pose'][0, 2])
        if distance <= 0:
            raise ValueError('Reset knife must be above the task endpoint')
        speed = float(frame['target_speed'][0, 0])
        trajectory = reference_trajectory(distance, speed,
                    self.env.metadata['environment']['agent_control_dt'],
                    self.env.metadata['free_cut_task']['horizon_steps'],
                    self.saw, self.amplitude, self.frequency)
        physics.set_reference_trajectory(trajectory)
        self.episode += 1
        self.step = 0
        self.trajectories.append({'episode': self.episode,
            'start_pose_base': pose[0].tolist(), 'sha256': trajectory.digest(),
            **trajectory.to_dict()})

    def act(self, frame):
        physics = self.env.physics
        t = physics.sim_time + self.env.metadata['environment']['agent_control_dt']
        self.commands.append({'episode': self.episode, 'step': self.step,
                              'target_pose_base': physics.reference_target(t)[0].tolist(),
                              'time_seconds': float(t[0])})
        self.step += 1
        return super().act(frame)

    def unchanged(self):
        return self.controller_parameters == asdict(self.env.physics.config)
