import json

import numpy as np
import pytest
import torch

from cutting_world_model.benchmark import (BENCHMARK_SCENES, ConstantPolicy, ReferencePolicy, benchmark_scene,
                                           check_contract, evaluate)


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class PhysicalEpisode:
    worlds = 1
    device = torch.device('cpu')
    metadata = {'action_bounds': [[-1.] * 6, [1.] * 6]}

    def __init__(self, clock, outcome='collision'):
        self.clock = clock
        self.outcome = outcome
        self.resets = 0
        self.reset()
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.steps = 0
        self.current = {'observation': torch.zeros(1, 24),
                        'sim_time': torch.zeros(1, 1, dtype=torch.float64),
                        'cut_spring_ke': torch.tensor([[0., 1., 1.]])}
        return self.current

    def step(self, action):
        assert self.steps < 2
        self.steps += 1
        self.clock.now += 1.
        self.current['observation'].fill_(self.steps)
        self.current['sim_time'].fill_(.05 if self.steps == 1 else .062)
        self.current['cut_spring_ke'][0, self.steps] = 0
        end = self.steps == 2
        collision = end and self.outcome == 'collision'
        success = end and self.outcome == 'success'
        truncated = end and self.outcome == 'truncated'
        return (self.current, torch.tensor([-25. if collision else 2.]),
                torch.tensor([collision or success]), torch.tensor([truncated]),
                {'collision': torch.tensor([collision]), 'success': torch.tensor([success]),
                 'outside': torch.tensor([False]), 'peak_force': torch.tensor([self.steps * 7.]),
                 'released_springs': torch.tensor([self.steps + 1]),
                 # Step integrals: step 1 is 0.05 s at 4 N, step 2 is 0.012 s at 10 N.
                 'material_impulse': torch.tensor([.2 if self.steps == 1 else .12], dtype=torch.float64),
                 'knife_work': torch.tensor([.01 if self.steps == 1 else -.002], dtype=torch.float64),
                 'knife_gross_work': torch.tensor([.01 if self.steps == 1 else .004], dtype=torch.float64),
                 'safety_margin': torch.tensor([.5 if self.steps == 1 else 1.2], dtype=torch.float64),
                 'jerkiness': torch.tensor([.3 if self.steps == 1 else .1], dtype=torch.float64)})


def test_terminal_frame_time_and_transition_alignment(tmp_path):
    clock = Clock()
    env = PhysicalEpisode(clock)
    report = evaluate(env, ConstantPolicy(), tmp_path, 2, 100., clock)
    assert report['state'] == 'completed' and report['completed_episodes'] == 2
    assert report['successes'] == 0 and report['physics_seconds'] == 4.
    assert env.resets == 2 and report['optimizer_updates'] == 0
    for row in report['episodes']:
        assert row['steps'] == 2 and row['collision'] and row['return'] == -23.
        assert row['simulation_seconds'] == .062
        assert row['material_peak_force_n'] == 14.
        assert row['released_springs'] == 3 and row['newly_released_springs'] == 2
        assert row['material_impulse_ns'] == pytest.approx(.32)
        assert row['mean_material_force_n'] == pytest.approx(.32 / .062)
        assert row['mean_material_force_n'] < row['material_peak_force_n']
        assert row['knife_work_j'] == pytest.approx(.008)
        assert row['knife_gross_work_j'] == pytest.approx(.014)
        assert row['safety_margin'] == pytest.approx(1.2)
        assert row['total_cut_springs'] == 3 and row['cut_fraction'] == 1.
        assert row['mean_jerk'] == pytest.approx(.2)
    summary = report['summary']
    assert summary['episodes'] == 2
    assert summary['success_rate'] == 0. and summary['collision_rate'] == 1.
    assert summary['knife_work_j'] == {'mean': pytest.approx(.008), 'std': pytest.approx(0.)}
    assert summary['return'] == {'mean': -23., 'std': 0.}
    assert 'success' not in summary and 'episode' not in summary
    assert json.loads((tmp_path / 'report.json').read_text())['summary']['episodes'] == 2
    with np.load(tmp_path / 'episode_000.npz') as data:
        np.testing.assert_array_equal(data['observation'][:, 0], [0, 1, 2])
        np.testing.assert_array_equal(data['reward'], [2., -25.])
        np.testing.assert_array_equal(data['terminated'], [False, True])
        np.testing.assert_array_equal(data['action'], np.zeros((2, 6)))
        assert data['sim_time'][-1, 0] == .062


def test_budget_preserves_incomplete_episode_without_counting_success(tmp_path):
    clock = Clock()
    env = PhysicalEpisode(clock, 'success')
    report = evaluate(env, ConstantPolicy(), tmp_path, 1, .5, clock)
    assert report['state'] == 'budget_complete'
    assert report['completed_episodes'] == report['successes'] == 0
    assert report['episodes'][0]['steps'] == 1
    assert not report['episodes'][0]['terminated']
    assert not report['episodes'][0]['truncated']
    assert report['summary'] == {'episodes': 0, 'success_rate': None, 'collision_rate': None}


def test_task_time_limit_stays_distinct_from_terminal_failure(tmp_path):
    clock = Clock()
    report = evaluate(PhysicalEpisode(clock, 'truncated'), ConstantPolicy(),
                      tmp_path, 1, 100., clock)
    row = report['episodes'][0]
    assert row['completed'] and row['truncated'] and not row['terminated']
    assert not row['success'] and not row['collision']


def test_wrong_material_contract_and_changed_policy_are_errors(tmp_path):
    with pytest.raises(ValueError, match='benchmark.material.dt'):
        check_contract({'material': {'dt': 1e-5}}, {'material': {'dt': 1e-4}})
    policy = ConstantPolicy()
    policy.unchanged = lambda: False
    clock = Clock()
    with pytest.raises(RuntimeError, match='changed policy'):
        evaluate(PhysicalEpisode(clock), policy, tmp_path, 1, 100., clock)
    assert json.loads((tmp_path / 'report.json').read_text())['state'] == 'failed'


def test_exception_keeps_valid_physical_transition_prefix(tmp_path):
    clock = Clock()
    env = PhysicalEpisode(clock)
    original = env.step
    def broken_step(action):
        if env.steps:
            raise RuntimeError('physical failure')
        return original(action)
    env.step = broken_step
    with pytest.raises(RuntimeError, match='physical failure'):
        evaluate(env, ConstantPolicy(), tmp_path, 1, 100., clock)
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['state'] == 'failed' and report['completed_episodes'] == 0
    assert report['episodes'][0]['steps'] == 1
    assert not report['episodes'][0]['completed']
    with np.load(tmp_path / 'episode_000.npz') as data:
        assert len(data['observation']) == 2 and len(data['action']) == len(data['reward']) == 1


def test_reset_change_is_rejected(tmp_path):
    clock = Clock()
    env = PhysicalEpisode(clock)
    original = env.reset
    def changed_reset():
        frame = original()
        frame['cut_spring_ke'][0, 1] = .5
        return frame
    env.reset = changed_reset
    with pytest.raises(ValueError, match='Physical reset changed cut_spring_ke'):
        evaluate(env, ConstantPolicy(), tmp_path, 1, 100., clock)


def test_reference_method_uses_the_reference_control_branch(tmp_path):
    clock = Clock()
    branches = []

    def recording_episode():
        env = PhysicalEpisode(clock, 'success')
        original = env.step
        def step(action, reference=False):
            branches.append(reference)
            return original(action)
        env.step = step
        return env

    report = evaluate(recording_episode(), ReferencePolicy(), tmp_path / 'reference', 1, 100., clock)
    assert branches == [True, True] and report['control_branch'] == 'knife_reference'
    assert report['successes'] == 1 and report['episodes'][0]['steps'] == 2
    branches.clear()
    report = evaluate(recording_episode(), ConstantPolicy(), tmp_path / 'constant', 1, 100., clock)
    assert branches == [False, False] and report['control_branch'] == 'policy_action'


def test_summary_mean_and_std_over_mixed_outcomes(tmp_path):
    clock = Clock()
    outcomes = iter(['success', 'success', 'collision'])  # __init__ resets once

    class Mixed(PhysicalEpisode):
        def reset(self):
            self.outcome = next(outcomes, self.outcome)
            return super().reset()

    env = Mixed(clock, 'success')
    report = evaluate(env, ConstantPolicy(), tmp_path, 2, 100., clock)
    summary = report['summary']
    assert summary['success_rate'] == .5 and summary['collision_rate'] == .5
    assert summary['return']['mean'] == pytest.approx((4. - 23.) / 2)
    assert summary['return']['std'] == pytest.approx(13.5)


def test_missing_jerk_observation_reports_no_mean_jerk(tmp_path):
    clock = Clock()

    class NoJerk(PhysicalEpisode):
        def step(self, action):
            frame, reward, terminated, truncated, info = super().step(action)
            info.pop('jerkiness')
            return frame, reward, terminated, truncated, info

    report = evaluate(NoJerk(clock, 'success'), ConstantPolicy(), tmp_path, 1, 100., clock)
    row = report['episodes'][0]
    assert row['completed'] and row['mean_jerk'] is None
    assert 'mean_jerk' not in report['summary'] and row['knife_work_j'] == pytest.approx(.008)


def test_benchmark_names_select_distinct_scenes():
    assert len(set(BENCHMARK_SCENES.values())) == len(BENCHMARK_SCENES)
    assert benchmark_scene({'name': 'public-prism-ur5e-v1'}) == 'runs/disect_source_v4/scene.pt'
    with pytest.raises(ValueError, match='Unknown benchmark'):
        benchmark_scene({'name': 'public-prism-ur5e-v0'})
