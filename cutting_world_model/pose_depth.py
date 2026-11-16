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
        return cls.physical_fields, CuttingStateSpec.history_fields + (
            'step_count', 'cut_progress', 'depth_span')

    def __init__(self, metadata):
        if metadata.get('physical_backend') not in ('mujoco_warp_disect', 'mujoco_warp_blademaster'):
            raise ValueError('Depth task requires a supported physical backend identity')
        if metadata.get('progress_semantics') != PROGRESS_SEMANTICS:
            raise ValueError('Depth task requires versioned geometric progress semantics')
        super().__init__(metadata)
        self.free_task = FreeCutTask(metadata['free_cut_task'], self.heads)
        self.board_geometry = metadata['board_pose_geometry']

    @torch.compiler.disable
    def integrate_physical(self, state, delta):
        fields = project_board(self, state, super().integrate_physical(state, delta))
        fields['depth_span'] = self.field(state, 'depth_span')
        fields['cut_progress'] = depth_progress(self.fk(fields['joint_q']),
            self.board_geometry, fields['depth_span'], self.field(state, 'cut_progress'))
        return fields

    derive = PoseCutStateSpec.derive
    pose_success = PoseCutStateSpec.pose_success
    hard_events = FreeCutStateSpec.hard_events
    hard_termination = PoseCutStateSpec.hard_termination
    completion_reward = PoseCutStateSpec.completion_reward
    terminal_features = PoseCutStateSpec.terminal_features
