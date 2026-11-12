from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cutting_world_model.free_cut_evaluation import checked_metadata, selected_config
from cutting_world_model.benchmark import ConstantPolicy, evaluate, checkpoint_metadata, save_episode
from cutting_world_model.mpc import MPPIPolicy, MPPIConfig, ModelRollout
from cutting_world_model.world_model import CuttingWorldModel
from test_benchmark import Clock, PhysicalEpisode
from test_free_cut_v3 import v3_fixture


def distribution():
    return dict(task={'version': 'free_cut_v3'}, seed=0, parallel=True,
                groups=[dict(name='cucumber', scene='cucumber.pt', worlds=256, material={'E': [1, 2]}),
                        dict(name='potato', scene='potato.pt', worlds=256)],
                force_limits=[50, 50, 50, 5, 5, 5])


def test_selecting_single_world_preserves_distribution_and_task():
    original = distribution()
    frozen = deepcopy(original)
    selected = selected_config(original, 'cucumber', 17)
    assert selected['groups'] == [{**original['groups'][0], 'worlds': 1}]
    assert selected['task'] == original['task'] and selected['force_limits'] == original['force_limits']
    assert selected['seed'] == 17 and selected['parallel'] is False
    assert original == frozen
    with pytest.raises(ValueError, match='unique group'):
        selected_config(original, 'absent', 0)
    with pytest.raises(ValueError, match='free_cut_v3'):
        selected_config({**original, 'task': {'version': 'free_cut_v2'}}, 'cucumber', 0)


def test_checkpoint_projection_never_bypasses_physics_or_task():
    conditions = distribution()
    contacts = [{'name': g['name'], 'scene': g['scene'], 'friction': .01} for g in conditions['groups']]
    saved = {'task_conditions': conditions, 'group_material_contact': contacts, 'gamma': 1,
             'limits': conditions['force_limits'], 'horizon': 180}
    actual = {k: deepcopy(v) for k, v in saved.items() if k != 'task_conditions'}
    actual['group_material_contact'] = contacts[:1]
    assert checked_metadata(actual, saved, conditions, 'cucumber') == saved
    for key, value in [('gamma', .99), ('horizon', 100), ('limits', [100]*6),
                       ('group_material_contact', [{**contacts[0], 'friction': 50.}])]:
        with pytest.raises(ValueError, match='free_cut_evaluation'):
            checked_metadata({**actual, key: value}, saved, conditions, 'cucumber')
    wrong = deepcopy(conditions)
    wrong['groups'][0]['scene'] = 'held_cucumber.pt'
    with pytest.raises(ValueError, match='training_conditions'):
        checked_metadata(actual, saved, wrong, 'cucumber')


def test_random_episode_resets_and_actual_terminal_labels_are_recorded(tmp_path):
    class RandomEpisode(PhysicalEpisode):
        reset_is_fixed = False

        def reset(self):
            frame = super().reset()
            frame['observation'][0, 0] = self.resets*.01
            return frame

        def conditions(self, worlds):
            return [{'youngs_modulus': self.resets*.5, 'group': 'cucumber'}]

        def step(self, action):
            frame, reward, term, trunc, info = super().step(action)
            # Timeout is a natural terminal; separation and success remain distinct.
            term = term | trunc
            info['timeout'] = trunc.clone()
            info['separated'] = torch.tensor([False])
            trunc.zero_()
            return frame, reward, term, trunc, info

    clock = Clock()
    env = RandomEpisode(clock, 'truncated')
    report = evaluate(env, ConstantPolicy(), tmp_path, 2, 100, clock)
    assert report['state'] == 'completed'
    assert report['summary']['timeout_rate'] == 1
    assert report['summary']['separated_rate'] == report['summary']['success_rate'] == 0
    a, b = report['episodes']
    assert a['conditions'] != b['conditions']
    assert a['terminated'] and not a['truncated']
    with np.load(tmp_path/'episode_000.npz') as data:
        assert data['timeout'].tolist() == [False, True]
        assert len(data['observation']) == len(data['reward'])+1


def test_mppi_v3_uses_exact_reward_completion_and_absorbs_timeout():
    model, state, goal = v3_fixture()
    model.spec.field(state, 'step_count').fill_(model.spec.free_task.horizon-1)
    model.spec.field(state, 'cut_progress').fill_(.6)
    rollout = ModelRollout(model, 1.)
    augmented = rollout.initial(state, goal)[None]
    action = state.new_zeros(1, 6)
    _, reward, _ = model.predict(state, action, goal)
    nxt = rollout.dynamics(augmented, action, 0)
    torch.testing.assert_close(nxt[:, -1], -reward)
    assert bool(nxt[0, -2]) and float(nxt[0, -3]) == 0
    twice = rollout.dynamics(nxt, action, 1)
    assert float(twice[0, -1]) == 0
    torch.testing.assert_close(twice[:, :-1], nxt[:, :-1])
    with pytest.raises(ValueError, match='discount'):
        ModelRollout(model, .99)


def test_real_mppi_optimizer_accepts_v3_goal_time_progress_and_freezes_weights(tmp_path, monkeypatch):
    pytest.importorskip('pytorch_mppi')
    wm, state, goal = v3_fixture()
    meta = deepcopy(wm.spec.metadata)
    frame = {key: wm.spec.field(state, key).detach().float().reshape(1, *meta['state_shapes'][key])
             for key in wm.spec.base_fields}
    frame['target_pose'], frame['target_speed'] = goal[:, :7].float(), goal[:, 7:].float()
    saved = {'metadata': meta, 'learner': {'world_model': wm.state_dict(), 'policy': {'w': torch.ones(1)},
                                         'updates': 0, 'training_stage': {'wm_updates': 341}}}
    path = tmp_path/'wm.pt'
    torch.save(saved, path)
    monkeypatch.setattr('cutting_world_model.mpc.CuttingWorldModel',
                        lambda metadata: CuttingWorldModel(metadata, hidden=(12, 12)))
    env = SimpleNamespace(worlds=1, metadata=meta, current=frame, device=torch.device('cpu'))
    assert checkpoint_metadata(env, meta) == meta
    planner = MPPIPolicy(path, env, MPPIConfig(horizon=3, samples=16, discount=1.), path)
    planner.reset(frame)
    action = planner.act(frame)
    assert action.shape == (1, 6) and torch.isfinite(action).all()
    assert (action.abs() <= 1).all()
    assert planner.unchanged() and planner.manifest['information']['uses_simulator_cut_progress']
    nxt = {k: v.clone() for k, v in frame.items()}
    nxt['step_count'] += 1
    nxt['cut_progress'] += .1
    planner.observe(nxt)
    torch.testing.assert_close(planner.spec.field(planner.state, 'cut_progress'), nxt['cut_progress'])
    assert planner.plans[0]['planning_seconds'] > 0


def test_transition_progress_does_not_overwrite_state_trajectory(tmp_path):
    states = [{'cut_progress': np.array([rho]), 'observation': np.array([0.])}
              for rho in (0., .5, 1.)]
    transitions = [{'cut_progress': np.array(rho), 'reward': 1.} for rho in (.5, 1.)]
    save_episode(tmp_path, {'episode': 0}, states, transitions)
    with np.load(tmp_path/'episode_000.npz') as data:
        np.testing.assert_array_equal(data['cut_progress'][:, 0], [0., .5, 1.])
        np.testing.assert_array_equal(data['transition_cut_progress'], [.5, 1.])


def test_unseen_scene_requires_explicit_declaration_preserves_every_other_setting():
    conditions = distribution()
    original = deepcopy(conditions)
    scene = 'runs/free_prism/scene.pt'
    runtime = selected_config(conditions, 'cucumber', 0, scene)
    assert runtime['groups'] == [{**conditions['groups'][0], 'worlds': 1,
                                  'scene': scene, 'name': 'cucumber__transfer'}]
    assert conditions == original
    for invalid in ('cucumber.pt', 'potato.pt', '/absolute.pt', '../outside.pt', 'scene.json'):
        with pytest.raises(ValueError):
            selected_config(conditions, 'cucumber', 0, invalid)
    saved = {'task_conditions': conditions, 'group_material_contact': [
        {'name': g['name'], 'scene': g['scene'], 'friction': .01}
        for g in conditions['groups']], 'horizon': 180,
        'limits': conditions['force_limits'], 'revision': 'edge_coordinate_consistent_v2'}
    actual = {k: deepcopy(v) for k, v in saved.items() if k != 'task_conditions'}
    actual['group_material_contact'] = [
        {**saved['group_material_contact'][0], 'name': 'cucumber__transfer', 'scene': scene}]
    with pytest.raises(ValueError, match='free_cut_evaluation'):
        checked_metadata(actual, saved, conditions, 'cucumber')
    assert checked_metadata(actual, saved, conditions, 'cucumber', scene) == saved
    inconsistent = deepcopy(saved)
    inconsistent['group_material_contact'][0]['scene'] = 'wrong.pt'
    with pytest.raises(ValueError, match='checkpoint_source_scene'):
        checked_metadata(actual, inconsistent, conditions, 'cucumber', scene)
    for key, value in [('horizon', 900), ('limits', [100]*6), ('revision', 'legacy'),
                       ('group_material_contact', [{**actual['group_material_contact'][0], 'friction': 50}])]:
        with pytest.raises(ValueError, match='free_cut_evaluation'):
            checked_metadata({**actual, key: value}, saved, conditions, 'cucumber', scene)
