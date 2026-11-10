"""Multi-condition task goal: per-object force limits and the target-speed reward."""
import copy

import numpy as np
import torch

from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.task_heads import SliceItHeads
from cutting_world_model.world_model import CuttingWorldModel
from test_control_state import compact_fixture


def heads_with_speed(weight=1.):
    metadata, _ = compact_fixture()
    environment = dict(metadata['environment'], w_speed=weight)
    return SliceItHeads(environment, metadata['controller']), metadata


def observation(heads, depth_error, down_speed):
    obs = torch.zeros(1, 24)
    obs[0, 1] = depth_error / heads.max_distance[1]
    velocity = -down_speed
    obs[0, 4] = velocity / heads.max_velocity[1] if heads.env['normalize_velocity'] else velocity
    return obs


def test_speed_reward_tracks_target_only_while_cutting():
    heads, _ = heads_with_speed()
    target = torch.tensor([.05])
    assert torch.allclose(heads.speed_reward(observation(heads, .03, .05), target), torch.zeros(1))
    slow = heads.speed_reward(observation(heads, .03, .025), target)
    assert torch.allclose(slow, -torch.tanh(torch.tensor([.5])))
    # Within the depth tolerance the knife should stop: no speed term.
    assert torch.allclose(heads.speed_reward(observation(heads, .001, 0.), target), torch.zeros(1))


def test_reward_weights_unchanged_without_speed_weight():
    metadata, _ = compact_fixture()
    plain = SliceItHeads(metadata['environment'], metadata['controller'])
    obs = observation(plain, .03, .05)
    flags = torch.zeros(1, dtype=torch.bool)
    assert plain.reward_weights(obs).shape == (4,)
    weighted, _ = heads_with_speed()
    assert weighted.reward_weights(obs).shape == (5,)
    assert torch.isfinite(weighted.reward(obs, flags, flags, flags, target_speed=torch.tensor([.05]))).all()


def test_goal_carries_target_speed_and_one_force_limit():
    metadata, frame = compact_fixture()
    metadata = dict(metadata, goal_dim=8)
    spec = ControlStateSpec(metadata)
    pose = torch.as_tensor(np.asarray(frame["previous_ee_pose"]), dtype=torch.float32)[None].repeat(2, 1)
    goal = torch.cat((pose, torch.tensor([[.035], [.065]])), -1)
    target, speed = spec.goal_parts(goal)
    assert torch.equal(target, pose) and torch.allclose(speed, torch.tensor([.035, .065]))
    assert torch.equal(spec.limits(goal), spec.force_limits)
    state = spec.pack_batch({k: torch.as_tensor(np.asarray(v), dtype=torch.float32)[None].repeat(2, *([1] * np.asarray(v).ndim))
                             for k, v in frame.items() if k in spec.base_fields})
    fields = {name: spec.field(state, name) for name in spec.physical_fields + spec.history_fields}
    fields['peak_wrench'] = 2 * spec.force_limits.expand(2, -1)
    _, _, done = spec.derive(state, dict(fields), torch.zeros(2, 6), goal)
    assert bool(done.all())  # the same limit applies to every world
    wm = CuttingWorldModel(metadata, hidden=(16, 16))
    next_state, reward, logits = wm.step(state, torch.zeros(2, 6), goal)
    assert next_state.shape == state.shape and logits.shape == (2,)
    try:
        spec.goal_parts(torch.zeros(2, 9))
    except ValueError:
        pass
    else:
        raise AssertionError('A 9-dimensional goal must be rejected')
