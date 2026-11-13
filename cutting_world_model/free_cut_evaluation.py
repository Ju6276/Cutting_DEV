"""Frozen one-world evaluation, optionally on an explicitly unseen object asset.

Training identity stays exact. A declared transfer changes only the object scene;
task, action, contact implementation and state-interface checks remain strict.
"""
from copy import deepcopy
from pathlib import PurePosixPath

from .benchmark import check_contract, copy_frame
from .control_state import ControlStateSpec


def selected_config(conditions, group_name, seed, transfer_scene=None):
    if conditions.get('task', {}).get('version') != 'free_cut_v3':
        raise ValueError('This evaluator requires an explicit free_cut_v3 task')
    names = [g['name'] for g in conditions['groups']]
    if len(names) != len(set(names)) or group_name not in names:
        raise ValueError('Select one unique group from the training conditions')
    result = deepcopy(conditions)
    result['groups'] = [deepcopy(conditions['groups'][names.index(group_name)])]
    if transfer_scene is not None:
        scene = PurePosixPath(transfer_scene)
        if scene.is_absolute() or '..' in scene.parts or scene.suffix != '.pt':
            raise ValueError('Transfer scene must be a relative backend .pt asset path')
        if str(scene) in [str(PurePosixPath(g['scene'])) for g in conditions['groups']]:
            raise ValueError('Transfer scene must be absent from the training groups')
        result['groups'][0]['scene'] = str(scene)
        result['groups'][0]['name'] = group_name + '__transfer'
    result['groups'][0]['worlds'] = 1
    result['parallel'] = False
    result['seed'] = int(seed)
    return result


def checked_metadata(actual, saved, conditions, group_name, transfer_scene=None):
    """Validate real task metadata, then retain the checkpoint's distribution identity."""
    check_contract(conditions, saved.get('task_conditions'), 'training_conditions')
    expected = deepcopy(saved)
    expected.pop('task_conditions')
    contacts = expected.get('group_material_contact', [])
    selected = [g for g in contacts if g['name'] == group_name]
    if len(selected) != 1:
        raise ValueError('Checkpoint has no unique contact model for the selected group')
    source = next(g for g in conditions['groups'] if g['name'] == group_name)
    check_contract(source['scene'], selected[0]['scene'], 'checkpoint_source_scene')
    runtime = selected_config(conditions, group_name, 0, transfer_scene)['groups'][0]
    selected[0]['name'], selected[0]['scene'] = runtime['name'], runtime['scene']
    expected['group_material_contact'] = selected
    check_contract(expected, actual, 'free_cut_evaluation')
    return deepcopy(saved)


class FreeCutEvaluationEnvironment:
    """Randomized episode resets with raw single-object arrays for episode recording."""
    reset_is_fixed = False

    def __init__(self, backend_root, conditions, group_name, seed, transfer_scene=None):
        from .multi_env import MultiConditionEnvironment
        self.training_conditions = deepcopy(conditions)
        self.group_name = group_name
        self.transfer_scene = transfer_scene
        self.runtime_conditions = selected_config(conditions, group_name, seed, transfer_scene)
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
        return checked_metadata(actual, saved_metadata, self.training_conditions, self.group_name,
                                self.transfer_scene)

    def reset(self):
        self.current = self._recordable(self.env.reset())
        return self.current

    def step(self, action, reference=False):
        frame, reward, terminated, truncated, info = self.env.step(action, reference=reference)
        self.current = self._recordable(frame)
        return self.current, reward, terminated, truncated, info

    def conditions(self, worlds):
        return self.env.conditions(worlds)

    def close(self):
        self.env.close()
