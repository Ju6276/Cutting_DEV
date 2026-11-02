from types import SimpleNamespace
import torch
from third_party.supermimic.objective import gae_targets, policy_objective


def objective(rewards, done, terminal, gamma=.99):
    survival = torch.cat((torch.ones_like(done[:1]), (1 - done).cumprod(dim=0)), dim=0)
    return policy_objective(SimpleNamespace(rewards=rewards, survival=survival), terminal, gamma)


def test_action_gradient_flows_through_fixed_dynamics_and_terminal_critic():
    model = torch.nn.Linear(1, 1, bias=False)
    critic = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.)
        critic.weight.fill_(3.)
    action = torch.tensor([[.25]], requires_grad=True)
    model.requires_grad_(False)
    critic.requires_grad_(False)
    state = model(action)
    reward = state.squeeze(-1).unsqueeze(0)
    loss = objective(reward, torch.zeros_like(reward), critic(state).squeeze(-1), gamma=.9)
    loss.backward()
    torch.testing.assert_close(action.grad, torch.tensor([[2. + .9 * 6.]]))
    assert model.weight.grad is None and critic.weight.grad is None


def test_terminal_transition_receives_reward_and_stops_bootstrap():
    rewards = torch.tensor([[2.], [100.]])
    done = torch.tensor([[1.], [0.]])
    values = torch.tensor([[7.], [8.], [9.]])
    result = objective(rewards, done, values[-1])
    torch.testing.assert_close(result, torch.tensor(2.))
    targets = gae_targets(rewards, done, values, .99, .95)
    torch.testing.assert_close(targets[0], torch.tensor([2.]))


def test_gae_lambda_one_matches_discounted_return():
    rewards = torch.tensor([[1.], [2.]])
    done = torch.zeros_like(rewards)
    values = torch.tensor([[100.], [-100.], [3.]])
    target = gae_targets(rewards, done, values, gamma=.5, lam=1.)
    torch.testing.assert_close(target, torch.tensor([[2.75], [3.5]]))
