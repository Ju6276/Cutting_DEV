import numpy as np
import torch

from cutting_world_model.manipulation import CuttingEnvironment, CuttingMDP
from cutting_world_model.control_state import ControlStateSpec
from test_state_spec import fixture


def mdp_fixture():
    metadata, frame = fixture()
    metadata['environment']['steps_per_episode'] = 500
    metadata = ControlStateSpec.metadata_from_frame(metadata, frame)
    mdp = CuttingMDP(metadata)
    state = torch.cat((mdp.spec.pack(frame), torch.zeros(1)))[None]
    goal = torch.tensor([[.06, 0., -.05, 0., 0., 0., 1.]])
    return mdp, state, goal


def test_manipulation_delta_targets_reconstruct_physics_and_shift_history():
    mdp, state, goal = mdp_fixture()
    action = torch.full((1, 6), .2)
    delta = torch.linspace(0, .01, mdp.delta_dim)[None]
    new = mdp.integrate(state, delta, action, goal)
    torch.testing.assert_close(mdp.targets(state, new), delta)
    torch.testing.assert_close(new[:, -1], state[:, -1] + 1)
    torch.testing.assert_close(mdp.spec.field(new, 'control_history')[:, -1],
                               state[:, :mdp.spec.base_dim])
    assert mdp.state_error(new, new).count_nonzero() == 0
    assert mdp.inputs(state, action, goal).shape == (1, mdp.input_dim)


def test_manipulation_terminal_reward_uses_pre_counter_and_finite_clock():
    mdp, state, goal = mdp_fixture()
    spec = mdp.spec
    state[:, spec.layout['success_counter']] = spec.heads.env['successes_threshold'] - 1
    new = state.clone()
    new[:, spec.layout['observation']] = 0
    new[:, spec.layout['success_counter']] += 1
    _, done = mdp.reward(state, torch.zeros(1, 6), new, goal)
    assert not done.item()  # Reading POST counter here would count this step twice.
    state[:, spec.layout['success_counter']] += 1
    _, done = mdp.reward(state, torch.zeros(1, 6), new, goal)
    assert done.item()
    new[:, spec.layout['peak_wrench']] = 2 * spec.force_limits
    reward, done = mdp.reward(state, torch.zeros(1, 6), new, goal)
    assert done.item() and reward.item() < -99
    new[:, spec.layout['peak_wrench']] = 0
    new[:, spec.layout['observation']][:, 1] = .5
    new[:, -1] = 500
    _, done = mdp.reward(state, torch.zeros(1, 6), new, goal)
    assert done.item()


def test_manipulation_action_gradient_through_two_predicted_steps():
    mdp, state, goal = mdp_fixture()
    mdp.spec.double()
    state, goal = state.double(), goal.double()
    action = torch.full((1, 6), .1, dtype=torch.float64, requires_grad=True)

    def forward(a):
        delta = torch.cat((a * .001, a * .01, a.repeat(1, 4) * .1), -1)
        first = mdp.integrate(state, delta, a, goal)
        second = mdp.integrate(first, delta, a, goal)
        reward, _ = mdp.reward(first, a, second, goal)
        return reward

    assert torch.autograd.gradcheck(forward, (action,), atol=2e-5, rtol=2e-3)
    assert torch.autograd.grad(forward(action).sum(), action)[0].abs().sum() > 0


def test_manipulation_autoreset_preserves_terminal_post_and_other_world_history(monkeypatch):
    import cutting_world_model.manipulation as module

    mdp, _, goal = mdp_fixture()
    metadata, single = fixture()
    metadata['environment']['steps_per_episode'] = 500
    selected = {key: np.stack([single[key], single[key]]).tolist()
                for key in mdp.spec.base_fields}
    selected.update(step_count=[[0], [0]], target_pose=goal.repeat(2, 1).tolist())
    post = {key: np.array(value) for key, value in selected.items()}
    post['joint_q'] += .02
    post['step_count'][:] = 1
    next_frame = {key: np.array(value) for key, value in post.items()}
    for key in selected:
        next_frame[key][0] = np.asarray(selected[key])[0]

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, operation, **kwargs):
            if operation == 'init':
                return dict(metadata=metadata, model_metadata=mdp.spec.metadata,
                            frame=selected, overflow=[0, 0])
            assert operation == 'step'
            return dict(post=post, frame=next_frame, terminated=[True, False],
                        truncated=[False, False], reward=[-100., -.5],
                        overflow=[0, 0], episode_metrics={}, material_force_max=51.)

        def close(self):
            pass

    monkeypatch.setattr(module, 'JsonProcess', Client)
    env = CuttingEnvironment('python', '.', worlds=2, scene='scene', capture='capture', device='cpu')
    before, _ = env.read()
    _, _, term, _, _ = env.step(torch.zeros(2, 6))
    assert term.tolist() == [True, False]
    assert (env.mdp.spec.field(env.post_step(), 'joint_q') == .02).all()
    torch.testing.assert_close(env.state[0], before[0])
    torch.testing.assert_close(env.state[1], env.post_step()[1])
    assert env.post_step()[0, -1].item() == 1
    assert env.state[0, -1].item() == 0
