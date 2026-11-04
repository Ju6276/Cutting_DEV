import math

import pytest
import torch

from scripts.check_control_model import evaluate
from cutting_world_model.world_model import CuttingWorldModel
from test_control_state import compact_fixture


def test_early_predicted_failure_counts_reward_once_but_keeps_raw_state_errors():
    metadata, frame = compact_fixture()
    model = CuttingWorldModel(metadata, hidden=(16, 16)).double()
    spec = model.spec
    initial = spec.pack(frame, dtype=torch.float64)
    goal = torch.tensor([.06, 0., -.05, 0., 0., 0., 1.], dtype=torch.float64)
    actions = torch.zeros(4, 6, dtype=torch.float64)
    states = initial.repeat(5, 1)
    with torch.no_grad():
        _, ordinary_reward, _ = model.step(states[:1], actions[:1], goal[None])
        peak = spec.delta_slices['peak_wrench'].start
        model.dynamics.mlp.net[-1].bias[peak] = 1.1
        _, entering_reward, _ = model.step(states[:1], actions[:1], goal[None])
    episode = {'path': 'physical-prefix', 'states': states, 'actions': actions,
               'goal': goal, 'rewards': ordinary_reward.repeat(4),
               'done': torch.zeros(4, dtype=torch.bool)}
    row = evaluate(model, [episode], horizons=(4,))[0]
    expected = abs(float(entering_reward) - 4 * float(ordinary_reward))
    assert row['return_rmse'] == pytest.approx(expected, abs=1e-5)
    expected_force = 4 * 1.1 * float(spec.force_limits[0]) / math.sqrt(3)
    assert row['peak_wrench_force_rmse_N'] == pytest.approx(expected_force)
