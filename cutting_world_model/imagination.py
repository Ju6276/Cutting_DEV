"""SuperMimic Eq. 2–3: differentiable returns and bootstrapped critic targets."""
from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def fixed_parameters(*modules):
    parameters = [parameter for module in modules for parameter in module.parameters()]
    flags = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


def imagined_return(rewards, terminal_probabilities, terminal_value, gamma=.99):
    survival = torch.ones_like(terminal_value)
    result = torch.zeros_like(terminal_value)
    for step in range(len(rewards)):
        result = result + gamma**step * survival * rewards[step]
        survival = survival * (1 - terminal_probabilities[step])
    return (result + gamma**len(rewards) * survival * terminal_value).mean()


@torch.no_grad()
def critic_targets(rewards, terminal_probabilities, values, gamma=.99, lam=.95):
    target = values[-1]
    targets = []
    for step in reversed(range(len(rewards))):
        bootstrap = (1 - lam) * values[step + 1] + lam * target
        target = rewards[step] + gamma * (1 - terminal_probabilities[step]) * bootstrap
        targets.append(target)
    return torch.stack(list(reversed(targets)))


class CuttingPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        size = 24
        for width in (2048, 1024, 512, 256):
            layers.extend((nn.Linear(size, width), nn.SiLU()))
            size = width
        layers.extend((nn.Linear(size, 6), nn.Tanh()))
        self.network = nn.Sequential(*layers)

    def forward(self, observation):
        return self.network(observation)


class CuttingCritic(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("scale", scale.clone())
        self.network = nn.Sequential(nn.Linear(len(mean), 256), nn.SiLU(),
                                     nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, state):
        return self.network((state - self.mean) / self.scale).squeeze(-1)


def imagine(model, policy, critic, initial, observation_slice, horizon=32):
    states = [initial]
    rewards, probabilities = [], []
    begin, end = observation_slice
    for _ in range(horizon):
        action = policy(states[-1][:, begin:end])
        next_state, reward, logits = model(states[-1], action)
        states.append(next_state)
        rewards.append(reward)
        probabilities.append(logits.sigmoid())
    states = torch.stack(states)
    rewards = torch.stack(rewards)
    probabilities = torch.stack(probabilities)
    values = critic(states)
    objective = imagined_return(rewards, probabilities, values[-1])
    targets = critic_targets(rewards.detach(), probabilities.detach(), values.detach())
    return objective, states.detach(), targets
