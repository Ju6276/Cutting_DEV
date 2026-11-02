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
        result = dict(task)
        result['prediction_layout'] = 'control_history_v1'
        result['prediction_history'] = PolicyTrainerCfg().history
        physical, history = cls.fields_for(result)
        result['state_shapes'] = {key: list(np.asarray(frame[key]).shape)
                                  for key in physical + history}
        result.pop('spring_reference_ke', None)
        return result

    def __init__(self, metadata):
        if metadata.get('prediction_layout') != 'control_history_v1':
            raise ValueError('Control predictor requires control_history_v1 metadata')
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
