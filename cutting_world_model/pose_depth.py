"""Shared pose task: geometric depth shaping, independent of fracture state."""
import torch
from .control_state import ControlStateSpec
from .state_spec import CuttingStateSpec
from .free_cut import FreeCutTask
from .free_cut_state import FreeCutStateSpec
from .pose_cut import PoseCutStateSpec, project_board, gap_from_pose

REVISION = 'pose_depth_v2'
PROGRESS_SEMANTICS = 'maximum_blade_depth_from_episode_start_v2'

def depth_progress(pose, geometry, depth_span, previous):
    """Monotone progress from the reset blade clearance to its board stop.
    Re-entering previously reached depths earns nothing; this is not separation.
    """
    gap = gap_from_pose(pose, geometry)[..., None]
    depth = (1 - (gap - geometry['stop_gap_m']) / depth_span).clamp(0, 1)
    return torch.maximum(previous, depth)

class PoseDepthStateSpec(ControlStateSpec):
    terminal_feature_dim = 14

    @classmethod
    def fields_for(cls, metadata):
        physical = cls.physical_fields
        if 'object_slip' in metadata.get('free_cut_task', {}):
            physical = physical + ('object_slip',)
        history = CuttingStateSpec.history_fields + ('step_count', 'cut_progress', 'depth_span')
        if 'completion_position' in metadata.get('free_cut_task', {}):
            history += ('completion_xy_error',)
        return physical, history

    def __init__(self, metadata):
        if metadata.get('physical_backend') not in ('mujoco_warp_disect', 'mujoco_warp_blademaster'):
            raise ValueError('Depth task requires a supported physical backend identity')
        if metadata.get('progress_semantics') != PROGRESS_SEMANTICS:
            raise ValueError('Depth task requires versioned geometric progress semantics')
        super().__init__(metadata)
        self.free_task = FreeCutTask(metadata['free_cut_task'], self.heads)
        self.board_geometry = metadata['board_pose_geometry']
        if 'completion_position' in self.free_task.config:
            self.terminal_feature_dim += 1
        if 'object_slip' in self.free_task.config:
            self.terminal_feature_dim += 1

    @torch.compiler.disable
    def integrate_physical(self, state, delta):
        fields = project_board(self, state, super().integrate_physical(state, delta))
        if 'object_slip' in fields:
            fields['object_slip'] = fields['object_slip'].clamp_min(0)
        fields['depth_span'] = self.field(state, 'depth_span')
        fields['cut_progress'] = depth_progress(self.fk(fields['joint_q']),
            self.board_geometry, fields['depth_span'], self.field(state, 'cut_progress'))
        return fields

    derive = PoseCutStateSpec.derive
    pose_success = PoseCutStateSpec.pose_success
    def hard_events(self, state):
        failure, timeout = FreeCutStateSpec.hard_events(self, state)
        if 'object_slip' in self.free_task.config:
            failure = failure | self.free_task.slip_failure(
                self.field(state, 'object_slip')[..., 0], failure)
        if 'completion_position' in self.free_task.config:
            failure = failure | self.free_task.tool_motion_failure(
                self.field(state,'completion_xy_error')[...,0],failure)
        return failure, timeout

    hard_termination = PoseCutStateSpec.hard_termination
    completion_reward = PoseCutStateSpec.completion_reward
    def terminal_features(self, state, goal=None):
        features = PoseCutStateSpec.terminal_features(self, state, goal)
        if 'object_slip' in self.free_task.config:
            margin = self.field(state, 'object_slip') / self.free_task.config['object_slip']['failure_m'] - 1
            features = torch.cat((features, margin), -1)
        if 'completion_position' in self.free_task.config:
            margin = self.field(state,'completion_xy_error') / self.free_task.config['completion_position']['radius_m'] - 1
            features = torch.cat((features,margin),-1)
        return features
