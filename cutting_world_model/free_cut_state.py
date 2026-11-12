"""Compact WM state for free_cut_v1; policy observations remain 24-dimensional."""
import torch

from .control_state import ControlStateSpec
from .state_spec import CuttingStateSpec
from .free_cut import FreeCutTask
from .kinematics import observation_velocity
from .task_heads import pose_error


class FreeCutStateSpec(ControlStateSpec):
    physical_fields = ControlStateSpec.physical_fields + ('cut_progress',)
    terminal_feature_dim = 11  # six force margins, three workspace margins, rho, time remaining

    @classmethod
    def fields_for(cls, metadata):
        return cls.physical_fields, CuttingStateSpec.history_fields + ('step_count',)

    def __init__(self, metadata):
        super().__init__(metadata)
        self.free_task = FreeCutTask(metadata['free_cut_task'], self.heads)

    def integrate_physical(self, state, delta):
        fields = super().integrate_physical(state, delta)
        rho = self.field(state, 'cut_progress')
        fraction = delta[..., self.delta_slices['cut_progress']].sigmoid()
        fields['cut_progress'] = rho + (1 - rho) * fraction
        return fields

    def derive(self, old, fields, action, goal):
        fields = dict(fields, control_history=self.shifted_history(old))
        pose = self.fk(fields['joint_q'])
        velocity = observation_velocity(pose, self.field(old, 'previous_ee_pose'), self.dt)
        acceleration = (velocity - self.field(old, 'previous_velocity')) / self.dt
        jerk = (acceleration - self.field(old, 'previous_acceleration')) / self.dt
        target, speed = self.goal_parts(goal)
        obs = self.heads.observation(pose_error(target, pose), velocity, jerk, action, fields['wrench_history'])
        count = self.field(old, 'step_count') + 1
        direction = action.new_tensor([1., 0., 0., 0., 0., 0.])
        rotation = action.new_tensor([0., 0., 0., 0., -1., 0.])
        limits = self.metadata['controller']['max_twist']
        motion = self.field(old, 'motion_command') + self.metadata['environment']['robot_control_dt'] * (
            limits[0] * action[..., 4:5] * direction + limits[1] * action[..., 5:6] * rotation)
        fields.update(motion_command=motion, previous_ee_pose=pose, previous_velocity=velocity,
                      previous_acceleration=acceleration, observation=obs,
                      success_counter=torch.zeros_like(self.field(old, 'success_counter')), step_count=count)
        next_state = self.assemble(fields)
        reward = self.free_task.step_reward(self.field(old, 'cut_progress')[..., 0],
                                            fields['cut_progress'][..., 0], obs,
                                            fields['peak_wrench'], speed)
        failure, timeout = self.hard_events(next_state)
        reward = reward - failure.to(reward.dtype) * self.free_task.weights['failure']
        return next_state, reward, failure | timeout

    def hard_events(self, state):
        failure = ((self.field(state, 'peak_wrench').abs() > self.force_limits).any(-1)
                   | self.free_task.workspace(self.field(state, 'observation')))
        timeout = self.field(state, 'step_count')[..., 0] >= self.free_task.horizon
        return failure, timeout

    def hard_termination(self, state):
        failure, timeout = self.hard_events(state)
        return failure | timeout

    def completion_reward(self, state, logits):
        # Expected reward on entering completion, weighted by survival in imagine/MPPI.
        # No exact rho==1 test and no invented progress threshold.
        failure, _ = self.hard_events(state)
        return (~failure).to(logits.dtype) * logits.sigmoid() * self.free_task.weights['success']

    def terminal_features(self, state, goal=None):
        obs = self.field(state, 'observation')
        workspace = ((obs[..., :3] * obs.new_tensor(self.heads.max_distance)).abs()
                     / obs.new_tensor(self.heads.env['workspace_limit']) - 1)
        force = self.field(state, 'peak_wrench').abs() / self.limits(goal) - 1
        remaining = (1 - self.field(state, 'step_count') / self.free_task.horizon).clamp(0, 1)
        return torch.cat((force, workspace, self.field(state, 'cut_progress'), remaining), -1)

