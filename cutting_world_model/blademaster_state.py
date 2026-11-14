"""BM predicts robot and wrench dynamics; blade-depth progress is analytic, not fracture."""
from .control_state import ControlStateSpec
from .state_spec import CuttingStateSpec
from .free_cut import FreeCutTask
from .free_cut_state import FreeCutStateSpec
from .pose_cut import PoseCutStateSpec, project_board, gap_from_pose
import torch

class BladeMasterStateSpec(ControlStateSpec):
    terminal_feature_dim = 14

    @classmethod
    def fields_for(cls, metadata):
        return cls.physical_fields, CuttingStateSpec.history_fields + ('step_count', 'cut_progress')

    def __init__(self, metadata):
        if metadata.get('physical_backend') != 'mujoco_warp_blademaster':
            raise ValueError('BladeMaster state requires its own physical backend identity')
        super().__init__(metadata)
        self.free_task = FreeCutTask(metadata['free_cut_task'], self.heads)
        self.board_geometry = metadata['board_pose_geometry']
        self.depth_span = float(metadata['depth_span_m'])
        if self.depth_span <= 0:
            raise ValueError('Positive blade depth span required')

    @torch.compiler.disable
    def integrate_physical(self, state, delta):
        fields = project_board(self, state, super().integrate_physical(state, delta))
        depth = (1-gap_from_pose(self.fk(fields['joint_q']), self.board_geometry)/self.depth_span).clamp(0,1)
        fields['cut_progress'] = torch.maximum(self.field(state,'cut_progress'), depth[...,None])
        return fields

    derive = PoseCutStateSpec.derive
    pose_success = PoseCutStateSpec.pose_success
    hard_events = FreeCutStateSpec.hard_events
    hard_termination = PoseCutStateSpec.hard_termination
    completion_reward = PoseCutStateSpec.completion_reward
    terminal_features = PoseCutStateSpec.terminal_features
