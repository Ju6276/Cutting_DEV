"""Control observations with episode-local causal history; SI units before scaling."""
import numpy as np
import torch

from third_party.supermimic.trainer_config import PolicyTrainerCfg
from .state_spec import CuttingStateSpec


class ControlStateSpec(CuttingStateSpec):
    physical_fields = ('joint_q', 'joint_qd', 'wrench_history', 'peak_wrench')

    @classmethod
    def fields_for(cls, metadata):
        return cls.physical_fields, CuttingStateSpec.history_fields

    @classmethod
    def metadata_from_frame(cls, task, frame):
        if task.get('free_cut_task') and cls is ControlStateSpec:
            if task['free_cut_task']['version'] == 'blademaster_pose_v1':
                from .blademaster_state import BladeMasterStateSpec
                return BladeMasterStateSpec.metadata_from_frame(task, frame)
            from .free_cut_state import FreeCutStateSpec
            return FreeCutStateSpec.metadata_from_frame(task, frame)
        result = dict(task)
        result['prediction_layout'] = 'control_history_v2'
        result['prediction_history'] = PolicyTrainerCfg().history
        physical, history = cls.fields_for(result)
        result['state_shapes'] = {key: list(np.asarray(frame[key]).shape)
                                  for key in physical + history}
        result.pop('spring_reference_ke', None)
        return result

    def __init__(self, metadata):
        if metadata.get('prediction_layout') != 'control_history_v2':
            raise ValueError('Control predictor requires control_history_v2 metadata')
        super().__init__(metadata)
        self.history_length = metadata['prediction_history']
        if self.history_length < 1:
            raise ValueError('Prediction history must include the current frame')
        self.base_dim = self.state_dim
        self.base_fields = self.physical_fields + self.history_fields
        self.shapes = dict(self.shapes)
        self.shapes['control_history'] = [self.history_length - 1, self.base_dim]
        self.layout['control_history'] = slice(self.base_dim, self.base_dim * self.history_length)
        self.history_fields = self.history_fields + ('control_history',)
        self.state_dim = self.base_dim * self.history_length
        self.in_scale = self.in_scale.repeat(self.history_length)
        indices = []
        for name in self.physical_fields + ('motion_command',):
            indices.extend(range(self.layout[name].start, self.layout[name].stop))
        # SliceIt observation: error, velocity, jerk, then the six executed actions.
        action_start = self.layout['observation'].start + 3 * len(self.heads.dims)
        if not self.heads.env['last_action_as_obs'] or not self.heads.env['jerkiness_as_obs']:
            raise ValueError('Control history requires the source action/jerk observation layout')
        indices.extend(range(action_start, action_start + 6))
        self.register_buffer('dynamics_indices', torch.tensor(indices, dtype=torch.long))

    @property
    def dynamics_dim(self):
        return self.history_length * len(self.dynamics_indices)

    def dynamics_features(self, state):
        frames = self.canonical(state).reshape(*state.shape[:-1], self.history_length, self.base_dim)
        return frames.index_select(-1, self.dynamics_indices).flatten(-2)

    def _base(self, frame, device=None, dtype=torch.float32):
        values = np.concatenate([np.asarray(frame[name]).reshape(-1) for name in self.base_fields])
        return torch.as_tensor(values, dtype=dtype, device=device)

    def pack(self, frame, device=None, dtype=torch.float32):
        base = self._base(frame, device, dtype)
        return base.repeat(self.history_length)

    def shifted_history(self, previous_state):
        history = self.field(previous_state, 'control_history')
        if self.history_length == 1:
            return history
        return torch.cat((history[..., 1:, :], previous_state[..., None, :self.base_dim]), -2)

    def pack_transition(self, frame, previous_state):
        base = self._base(frame, previous_state.device, previous_state.dtype)
        history = self.shifted_history(previous_state)
        return torch.cat((base, history.flatten(-2)), -1)

    def _base_batch(self, frame):
        batch_shape = frame['joint_q'].shape[:-1]
        return torch.cat([frame[name].reshape(*batch_shape, -1)
                          for name in self.base_fields], -1)

    def pack_batch(self, frame):
        base = self._base_batch(frame)
        return base.repeat(*([1] * (base.ndim - 1)), self.history_length)

    def pack_batch_transition(self, frame, previous_state):
        base = self._base_batch(frame).to(previous_state)
        history = self.shifted_history(previous_state)
        return torch.cat((base, history.flatten(-2)), -1)

    def reset_batch_history(self, frame, previous_state, mask):
        initial = self.pack_batch(frame).to(previous_state)
        return torch.where(mask[..., None], initial, previous_state)

    def pack_episode(self, data, device=None, dtype=torch.float32):
        states = []
        for t in range(len(data['joint_q'])):
            frame = {key: data[key][t] for key in self.base_fields}
            state = self.pack(frame, device, dtype) if t == 0 else self.pack_transition(frame, states[-1])
            states.append(state)
        return torch.stack(states)

    def derive(self, old, fields, action, goal):
        fields = dict(fields, control_history=self.shifted_history(old))
        return super().derive(old, fields, action, goal)
