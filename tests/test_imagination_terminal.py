from types import SimpleNamespace

import pytest
import torch

from cutting_world_model.state_spec import CuttingStateSpec
from cutting_world_model.trainer import OnlineLearner
from third_party.supermimic.objective import gae_targets, policy_objective
from test_state_spec import fixture


@pytest.mark.parametrize('event', ['force', 'workspace', 'goal', 'horizon'])
def test_known_task_terminal_is_absorbing_even_when_classifier_predicts_survival(event):
    metadata, frame = fixture()
    spec = CuttingStateSpec(metadata)
    state = spec.pack(frame)[None]
    state[:, spec.layout['success_counter']] = spec.heads.env['successes_threshold'] if event == 'goal' else 0
    state[:, spec.layout['observation']] = 0
    goal = torch.zeros((1, 7))
    goal[:, -1] = 1

    class Predictor:
        def __init__(self):
            self.spec = spec

        def predict(self, old, action, goal):
            new = old.clone()
            new[:, spec.layout['observation']] = 0
            new[:, spec.layout['peak_wrench']] = 0
            obs = spec.field(new, 'observation')
            if event == 'force':
                new[:, spec.layout['peak_wrench']] = 2 * spec.force_limits
            elif event == 'workspace':
                obs[:, 0] = 2 * spec.heads.env['workspace_limit'][0] / spec.heads.max_distance[0]
            elif event == 'horizon':
                obs[:, 1] = 2 * spec.heads.env['goal_threshold'][1] / spec.heads.max_distance[1]
            collision = (spec.field(new, 'peak_wrench') > spec.force_limits).any(-1)
            done, success, outside, count = spec.heads.termination(
                obs, spec.field(old, 'success_counter'), collision)
            new[:, spec.layout['success_counter']] = count[:, None]
            reward = spec.heads.reward(obs, collision, success, outside)
            return new, reward, torch.full((1,), -100.)

    predictor = Predictor()
    learner = SimpleNamespace(wm=predictor, pcfg=SimpleNamespace(horizon=4),
                              policy=lambda history, target: torch.zeros((1, 6)))
    ro = OnlineLearner.imagine(learner, state, state[None].repeat(10, 1, 1), goal)
    values = torch.full((5, 1), -10000.)
    target = gae_targets(ro.rewards, ro.dhat[1:], values, .99, .95)
    bootstrap = torch.tensor([-10000.], requires_grad=True)
    objective = policy_objective(ro, bootstrap, .99)
    objective.backward()
    if event == 'horizon':
        assert ro.survival[-1].item() == 1
        assert bootstrap.grad.item() == pytest.approx(.99 ** 4)
        assert target[0].item() < -1000
    else:
        assert ro.survival[1:].count_nonzero() == 0
        assert ro.rewards[1:].count_nonzero() == 0
        torch.testing.assert_close(ro.states[1:], ro.states[1:2].expand_as(ro.states[1:]))
        torch.testing.assert_close(target[0], ro.rewards[0])
        assert target[1:].count_nonzero() == 0
        assert bootstrap.grad.item() == 0
