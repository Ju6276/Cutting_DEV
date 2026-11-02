import copy

import numpy as np
import torch

from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.state_spec import CuttingStateSpec
from cutting_world_model.world_model import CuttingWorldModel
from test_state_spec import fixture


def compact_fixture():
    metadata, frame = fixture()
    return ControlStateSpec.metadata_from_frame(metadata, frame), frame


def base(spec, state):
    fields = [name for name in spec.physical_fields + spec.history_fields
              if name != 'control_history']
    return torch.cat([spec.field(state, name).reshape(*state.shape[:-1], -1)
                      for name in fields], dim=-1)


def test_compact_recording_history_is_causal_and_matches_streaming():
    metadata, initial = compact_fixture()
    spec = ControlStateSpec(metadata)
    frames = []
    for step in range(13):
        frame = copy.deepcopy(initial)
        frame['joint_q'][:] = .01 * step
        frame['joint_qd'][:] = .2 * step
        frame['observation'][9:15] = step / 20
        frame['wrench_history'][:] = step
        frame['peak_wrench'][:] = step + 1
        frames.append(frame)
    data = {name: np.stack([frame[name] for frame in frames]) for name in initial}
    episode = spec.pack_episode(data)
    streamed = [spec.pack(frames[0])]
    for frame in frames[1:]:
        streamed.append(spec.pack_transition(frame, streamed[-1]))
    torch.testing.assert_close(episode, torch.stack(streamed))
    for step, state in enumerate(streamed):
        expected = torch.stack([base(spec, streamed[max(0, step - 9 + lag)])
                                for lag in range(9)])
        torch.testing.assert_close(spec.field(state, 'control_history'), expected)
    reset = spec.pack(initial)
    torch.testing.assert_close(spec.field(reset, 'control_history'),
                               base(spec, reset).expand(9, -1))
    changed = copy.deepcopy(data)
    changed['joint_q'][8:] += 10
    torch.testing.assert_close(spec.pack_episode(changed)[:8], episode[:8])


def test_compact_task_heads_match_full_state_and_shift_memory_analytically():
    metadata, frame = compact_fixture()
    compact = ControlStateSpec(metadata).double()
    full_metadata, _ = fixture()
    full = CuttingStateSpec(full_metadata).double()
    frame['joint_qd'][:] = .03
    frame['wrench_history'][:] = np.arange(18).reshape(3, 6) / 10
    frame['peak_wrench'][0] = 51
    compact_state = compact.pack(frame).double()[None]
    full_state = full.pack(frame).double()[None]
    action = torch.full((1, 6), .2, dtype=torch.float64)
    goal = torch.tensor([[.06, 0., -.05, 0., 0., 0., 1.]], dtype=torch.float64)
    compact_fields = compact.integrate_physical(
        compact_state, torch.zeros(1, compact.delta_dim, dtype=torch.float64))
    full_fields = full.integrate_physical(
        full_state, torch.zeros(1, full.delta_dim, dtype=torch.float64))
    result, reward, done = compact.derive(compact_state, compact_fields, action, goal)
    full_result, full_reward, full_done = full.derive(full_state, full_fields, action, goal)
    torch.testing.assert_close(reward, full_reward)
    torch.testing.assert_close(done, full_done)
    assert done.item()
    for name in ('observation', 'previous_ee_pose', 'previous_velocity',
                 'previous_acceleration', 'motion_command', 'success_counter'):
        torch.testing.assert_close(compact.field(result, name), full.field(full_result, name))
    torch.testing.assert_close(compact.terminal_features(result), full.terminal_features(full_result))
    torch.testing.assert_close(compact.field(result, 'control_history')[..., -1, :],
                               base(compact, compact_state))


def test_compact_outputs_ignore_material_mesh_and_history_targets():
    metadata, frame = compact_fixture()
    spec = ControlStateSpec(metadata)
    state = spec.pack(frame)
    frame['particle_q'] = np.ones((19, 3))
    frame['particle_qd'] = np.ones((19, 3))
    frame['cut_spring_ke'] = np.ones(31)
    torch.testing.assert_close(spec.pack(frame), state)
    other = state.clone()
    other[spec.layout['control_history']] += 1
    torch.testing.assert_close(spec.error_vector(other, state), torch.zeros(36))
    assert spec.delta_dim == 36


def test_compact_multistep_action_gradients_pass_through_history_and_heads():
    torch.manual_seed(17)
    metadata, frame = compact_fixture()
    model = CuttingWorldModel(metadata, hidden=(12, 12)).double()
    assert isinstance(model.spec, ControlStateSpec)
    torch.nn.init.normal_(model.dynamics.mlp.net[-1].weight, std=1e-4)
    state = model.spec.pack(frame).double()[None]
    goal = torch.tensor([[.06, 0., -.05, 0., 0., 0., 1.]], dtype=torch.float64)
    actions = torch.full((2, 1, 6), .1, dtype=torch.float64, requires_grad=True)

    def rollout(value):
        next_state, reward1, logits1 = model.step(state, value[0], goal)
        final_state, reward2, logits2 = model.step(next_state, value[1], goal)
        return torch.cat((base(model.spec, final_state), reward1[:, None],
                          reward2[:, None], logits1[:, None], logits2[:, None]), dim=-1)

    assert torch.autograd.gradcheck(rollout, (actions,), atol=2e-5, rtol=2e-3)
    derivatives = torch.autograd.grad(rollout(actions).sum(), actions)[0]
    assert torch.isfinite(derivatives).all()
    assert derivatives[0].abs().sum() > 0
