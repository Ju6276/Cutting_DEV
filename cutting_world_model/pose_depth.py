"""Shared pose task: geometric depth shaping, independent of fracture state."""
import torch
from .control_state import ControlStateSpec
from .state_spec import CuttingStateSpec
from .free_cut import FreeCutTask, POSE_DEPTH_VERSIONS
from .free_cut_state import FreeCutStateSpec
from .pose_cut import PoseCutStateSpec, project_board, gap_from_pose

REVISION = 'pose_depth_v2'
REVISION_V3 = 'pose_depth_v3'
assert POSE_DEPTH_VERSIONS == (REVISION, REVISION_V3)
# Unchanged in v3: geometric progress is the same quantity.
PROGRESS_SEMANTICS = 'maximum_blade_depth_from_episode_start_v2'
# Board-arrival clearance of the software board stop (m). Same criterion as the physical
# latch: backends/mujoco/cutting_gpu_backend/knife_board.py register_stop and
# blademaster_backend.BladeMasterBatch.step (limited step AND end clearance <= 10 um).
BOARD_LATCH_GAP_M = 1e-5

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


def pose_depth_spec_class(version):
    """State spec of a pose_depth family version; v2 keeps its original class."""
    if version == REVISION:
        return PoseDepthStateSpec
    if version == REVISION_V3:
        return PoseDepthV3StateSpec
    raise ValueError('Not a pose_depth task version: ' + str(version))


def latched_interval_peak(peak, reading, latched, collision, limits):
    """Post-latch peak rule; one implementation shared with the DiSECt adapter
    (cutting_gpu_backend.knife_board, which cannot import this package)."""
    from cutting_gpu_backend.knife_board import latched_interval_peak as rule
    return rule(peak, reading, latched, collision, limits)


def observe_board_latch(env, frame):
    """pose_depth_v3 frame fields: the latch flag (0/1) and the force-limit signal at the
    last 2 ms sample of the interval (frozen reading for latched worlds; the initialised
    reading after reset). This is a software board-stop flag, not a board contact force."""
    physics = env.physics
    if env.metadata['physical_backend'] == 'mujoco_warp_blademaster':
        reading = physics.feedback['sensor_wrench']
    else:
        if physics.force_model != 'net':
            raise ValueError('pose_depth_v3 needs the net force signal')
        # Net model: the observation wrench is the filtered sensor wrench that the
        # force-limit check reads (coupling.observation_wrench).
        reading = physics.feedback['observation_wrench']
    dtype = frame['peak_wrench'].dtype
    frame['last_wrench'] = reading.to(dtype).clone()
    frame['board_contact'] = frame['knife_board_contact'].reshape(-1, 1).to(dtype).clone()
    return frame


class PoseDepthV3StateSpec(PoseDepthStateSpec):
    """pose_depth_v2 plus a software board-stop boundary consistency fix.

    The physical adapters stop stepping a world once the software board stop latches;
    its pose, recurrence and force reading stay frozen while the task clock continues.
    v3 imagines the same rule: a latched state keeps q, wrench history and the last
    force-limit reading, has zero joint velocity and reports |last reading| as its
    interval peak. Only the first-contact step is predicted by the dynamics.
    """
    board_latch = True
    # Success (+100), failure (-100) and timeout are all settled analytically in derive /
    # hard_termination, so imagined survival is the analytic absorption alone. The learned
    # terminal head is still fitted with the unchanged BCE (and, with terminal_gradient
    # 'dynamics', its gradient still reaches the dynamics); its probabilities are logged
    # but do not weight the policy objective, GAE targets or MPPI cost.
    analytic_survival = True

    @classmethod
    def fields_for(cls, metadata):
        physical, history = super().fields_for(metadata)
        return physical + ('last_wrench',), history + ('board_contact',)

    def __init__(self, metadata):
        if metadata.get('free_cut_task', {}).get('version') != REVISION_V3:
            raise ValueError('pose_depth_v3 state requires pose_depth_v3 task metadata')
        shapes = metadata.get('state_shapes', {})
        if shapes.get('last_wrench') != [6] or shapes.get('board_contact') != [1]:
            raise ValueError('pose_depth_v3 state needs last_wrench[6] and board_contact[1]; '
                             'pose_depth_v2 replay/WM/critic data cannot be loaded')
        super().__init__(metadata)

    def contact(self, state):
        return self.field(state, 'board_contact') > .5

    @torch.compiler.disable
    def integrate_physical(self, state, delta):
        fields = CuttingStateSpec.integrate_physical(self, state, delta)
        fields, limited, end_gap = project_board(self, state, fields, with_contact=True)
        if 'object_slip' in fields:
            fields['object_slip'] = fields['object_slip'].clamp_min(0)
        fields['depth_span'] = self.field(state, 'depth_span')
        fields['cut_progress'] = depth_progress(self.fk(fields['joint_q']),
            self.board_geometry, fields['depth_span'], self.field(state, 'cut_progress'))
        return self.apply_board_latch(state, fields, limited, end_gap)

    def apply_board_latch(self, state, fields, limited, end_gap):
        """Latch update and frozen physical fields; history fields are derived as usual."""
        held = self.contact(state)
        arrived = (limited & (end_gap <= BOARD_LATCH_GAP_M))[..., None]
        last = self.field(state, 'last_wrench')
        fields['joint_q'] = torch.where(held, self.field(state, 'joint_q'), fields['joint_q'])
        fields['joint_qd'] = torch.where(held, torch.zeros_like(fields['joint_qd']), fields['joint_qd'])
        fields['wrench_history'] = torch.where(held[..., None], self.field(state, 'wrench_history'),
                                               fields['wrench_history'])
        fields['last_wrench'] = torch.where(held, last, fields['last_wrench'])
        fields['peak_wrench'] = torch.where(held, last.abs(), fields['peak_wrench'])
        fields['cut_progress'] = torch.where(held, self.field(state, 'cut_progress'), fields['cut_progress'])
        if 'object_slip' in fields:
            # Latched physical worlds no longer advance their material either.
            fields['object_slip'] = torch.where(held, self.field(state, 'object_slip'), fields['object_slip'])
        fields['board_contact'] = (held | arrived).to(fields['joint_q'].dtype)
        return fields
