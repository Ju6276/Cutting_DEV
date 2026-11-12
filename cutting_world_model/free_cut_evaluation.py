"""One physical world from an exact training distribution, for frozen controllers.

Only execution batch size, group selection, process mode and RNG seed may differ.
Task/physics metadata remain checked; this is not an allow-mismatch checkpoint load.
"""
from copy import deepcopy

from .benchmark import check_contract, copy_frame
from .control_state import ControlStateSpec


def selected_config(conditions, group_name, seed):
    if conditions.get('task', {}).get('version') != 'free_cut_v3':
        raise ValueError('This evaluator requires an explicit free_cut_v3 task')
    names = [g['name'] for g in conditions['groups']]
    if len(names) != len(set(names)) or group_name not in names:
        raise ValueError('Select one unique group from the training conditions')
    result = deepcopy(conditions)
    result['groups'] = [deepcopy(conditions['groups'][names.index(group_name)])]
    result['groups'][0]['worlds'] = 1
    result['parallel'] = False
    result['seed'] = int(seed)
    return result


def checked_metadata(actual, saved, conditions, group_name):
    """Validate real task metadata, then retain the checkpoint's distribution identity."""
    check_contract(conditions, saved.get('task_conditions'), 'training_conditions')
    expected = deepcopy(saved)
    expected.pop('task_conditions')
    contacts = expected.get('group_material_contact', [])
    selected = [g for g in contacts if g['name'] == group_name]
    if len(selected) != 1:
        raise ValueError('Checkpoint has no unique contact model for the selected group')
    expected['group_material_contact'] = selected
    check_contract(expected, actual, 'free_cut_evaluation')
    return deepcopy(saved)


class FreeCutEvaluationEnvironment:
    """Randomized episode resets with raw single-object arrays for episode recording."""
    reset_is_fixed = False

    def __init__(self, backend_root, conditions, group_name, seed):
        from .multi_env import MultiConditionEnvironment
        self.training_conditions = deepcopy(conditions)
        self.group_name = group_name
        self.runtime_conditions = selected_config(conditions, group_name, seed)
        self.env = MultiConditionEnvironment.from_config(backend_root, self.runtime_conditions)
        self.worlds, self.device = self.env.worlds, self.env.device
        self.metadata = self.env.metadata
        self.physics = self.env.members[0].env.physics
        self.current = self._recordable(self.env.current)

    def _recordable(self, frame):
        from .multi_env import PER_OBJECT_KEYS
        material_frame = self.env.members[0].env.current
        return {**frame, **{k: material_frame[k] for k in PER_OBJECT_KEYS if k in material_frame}}

    def checkpoint_metadata(self, saved_metadata):
        actual = ControlStateSpec.metadata_from_frame(self.metadata, copy_frame(self.current))
        return checked_metadata(actual, saved_metadata, self.training_conditions, self.group_name)

    def reset(self):
        self.current = self._recordable(self.env.reset())
        return self.current

    def step(self, action):
        frame, reward, terminated, truncated, info = self.env.step(action)
        self.current = self._recordable(frame)
        return self.current, reward, terminated, truncated, info

    def conditions(self, worlds):
        return self.env.conditions(worlds)

    def close(self):
        self.env.close()
