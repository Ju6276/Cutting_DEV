import torch

from cutting_world_model.imagination import critic_targets, fixed_parameters, imagined_return


def test_action_gradient_flows_through_fixed_dynamics_and_terminal_critic():
    model = torch.nn.Linear(1, 1, bias=False)
    critic = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.)
        critic.weight.fill_(3.)
    action = torch.tensor([[.25]], requires_grad=True)
    with fixed_parameters(model, critic):
        state = model(action)
        reward = state.squeeze(-1).unsqueeze(0)
        objective = imagined_return(reward, torch.zeros_like(reward), critic(state).squeeze(-1), gamma=.9)
        objective.backward()
    torch.testing.assert_close(action.grad, torch.tensor([[2. + .9 * 6.]]))
    assert model.weight.grad is None and critic.weight.grad is None
    assert model.weight.requires_grad and critic.weight.requires_grad


def test_terminal_transition_receives_reward_and_stops_bootstrap():
    rewards = torch.tensor([[2.], [100.]])
    done = torch.tensor([[1.], [0.]])
    values = torch.tensor([[7.], [8.], [9.]])
    result = imagined_return(rewards, done, values[-1])
    torch.testing.assert_close(result, torch.tensor(2.))
    targets = critic_targets(rewards, done, values)
    torch.testing.assert_close(targets[0], torch.tensor([2.]))


def test_gae_lambda_one_matches_discounted_return():
    rewards = torch.tensor([[1.], [2.]])
    done = torch.zeros_like(rewards)
    values = torch.tensor([[100.], [-100.], [3.]])
    target = critic_targets(rewards, done, values, gamma=.5, lam=1.)
    torch.testing.assert_close(target, torch.tensor([[2.75], [3.5]]))
