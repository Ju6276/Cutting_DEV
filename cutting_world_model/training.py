"""Eight-step trajectory windows and supervised world-model fitting."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dynamics import CuttingDynamics, pack_trajectory
from .state_update import structured_update


class EpisodeWindows:
    def __init__(self, paths, window=8):
        self.paths = [Path(path).resolve() for path in paths]
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("Duplicate episodes")
        self.episodes = []
        self.indices = []
        self.layout = None
        self.window = window
        for index, path in enumerate(self.paths):
            state, actions, layout = pack_trajectory(path)
            if self.layout is not None and layout != self.layout:
                raise ValueError("Episodes use different state layouts")
            self.layout = layout
            with np.load(path) as data:
                rewards = data["rewards"].astype(np.float32)
                terminated = data["terminated"].astype(np.float32)
                truncated = data["truncated"].astype(bool)
            if rewards.shape != (len(actions),) or terminated.shape != rewards.shape or truncated.shape != rewards.shape:
                raise ValueError("Transition labels must align with actions")
            if np.any(terminated[:-1]) or np.any(truncated[:-1]):
                raise ValueError("A trajectory crosses an episode boundary")
            if not np.isfinite(rewards).all() or not np.isin(terminated, [0, 1]).all():
                raise ValueError("Invalid reward or termination labels")
            self.episodes.append((state, actions, rewards, terminated))
            self.indices.extend((index, start) for start in range(len(actions) - window + 1))
        if not self.indices:
            raise ValueError("No complete training windows")

    def batch(self, indices, device):
        states, actions, rewards, done = [], [], [], []
        for index in indices:
            episode, start = self.indices[index]
            s, a, r, d = self.episodes[episode]
            end = start + self.window
            states.append(s[start:end + 1])
            actions.append(a[start:end])
            rewards.append(r[start:end])
            done.append(d[start:end])
        return tuple(torch.as_tensor(np.stack(values, axis=1), device=device)
                     for values in (states, actions, rewards, done))

    def statistics(self):
        state = np.concatenate([episode[0] for episode in self.episodes]).astype(np.float64)
        reward = np.concatenate([episode[2] for episode in self.episodes]).astype(np.float64)
        mean = state.mean(axis=0)
        variance = state.var(axis=0)
        scale = np.empty_like(mean)
        for name, (begin, end) in self.layout.items():
            groups = [(begin, end)]
            if name in ("knife_pose", "ee_pose", "previous_ee_pose", "target_pose"):
                groups = [(begin, begin + 3), (begin + 3, end)]
            elif name in ("knife_velocity", "wrench_history", "previous_velocity", "previous_acceleration"):
                groups = [(offset, min(offset + 3, end)) for offset in range(begin, end, 3)]
            for start, stop in groups:
                pooled_scale = float(np.sqrt(variance[start:stop].mean()))
                scale[start:stop] = pooled_scale or 1.
        return mean.astype(np.float32), scale.astype(np.float32), float(reward.mean()), float(reward.std() or 1)


class CoupledWorldModel(nn.Module):
    def __init__(self, upstream_root, mean, scale, reward_mean, reward_scale, layout, action_increment):
        super().__init__()
        self.transition = CuttingDynamics(upstream_root, mean, scale)
        self.layout = layout
        self.register_buffer("action_increment", torch.as_tensor(action_increment, dtype=torch.float32))
        size = len(mean)
        self.reward_head = nn.Sequential(nn.Linear(size + 6, 256), nn.SiLU(),
                                         nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
        self.termination_head = nn.Sequential(nn.Linear(size, 256), nn.SiLU(),
                                              nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
        self.register_buffer("reward_mean", torch.tensor(reward_mean, dtype=torch.float32))
        self.register_buffer("reward_scale", torch.tensor(reward_scale, dtype=torch.float32))

    def normalize(self, state):
        return (state - self.transition.mean) / self.transition.scale

    def forward(self, state, action):
        proposal = self.transition(state, action)
        next_state = structured_update(state, proposal, action, self.layout, self.action_increment)
        reward_normalized = self.reward_head(torch.cat((self.normalize(state), action), dim=-1)).squeeze(-1)
        terminal_logits = self.termination_head(self.normalize(next_state)).squeeze(-1)
        reward = reward_normalized * self.reward_scale + self.reward_mean
        return next_state, reward, terminal_logits

    def window_loss(self, states, actions, rewards, terminated):
        state = states[0]
        losses = []
        for index, action in enumerate(actions):
            state, reward, logits = self(state, action)
            state_error = ((state - states[index + 1]) / self.transition.scale).square().mean()
            reward_error = ((reward - rewards[index]) / self.reward_scale).square().mean()
            terminal_error = nn.functional.binary_cross_entropy_with_logits(logits, terminated[index])
            losses.append(state_error + reward_error + terminal_error)
        return torch.stack(losses).sum()


@torch.no_grad()
def evaluate(model, dataset, device):
    totals = {name: [0., 0., 0] for name in dataset.layout}
    reward_error = 0.
    terminal_error = 0.
    invalid = {"negative_stiffness": 0, "increasing_stiffness": 0,
               "quaternion_norm_max_error": 0.}
    transitions = 0
    for start in range(0, len(dataset.indices), 32):
        states, actions, rewards, terminated = dataset.batch(
            range(start, min(start + 32, len(dataset.indices))), device)
        state = states[0]
        persistence = state
        for index, action in enumerate(actions):
            previous = state
            state, reward, logits = model(state, action)
            if not torch.isfinite(state).all() or not torch.isfinite(reward).all():
                raise FloatingPointError("Non-finite validation rollout")
            for name, (begin, end) in dataset.layout.items():
                error = (state[:, begin:end] - states[index + 1, :, begin:end]).double()
                base = (persistence[:, begin:end] - states[index + 1, :, begin:end]).double()
                totals[name][0] += error.square().sum().item()
                totals[name][1] += base.square().sum().item()
                totals[name][2] += error.numel()
            begin, end = dataset.layout["cut_spring_ke"]
            invalid["negative_stiffness"] += int((state[:, begin:end] < 0).sum())
            invalid["increasing_stiffness"] += int((state[:, begin:end] > previous[:, begin:end]).sum())
            begin, end = dataset.layout["knife_pose"]
            quat_error = (state[:, begin + 3:end].norm(dim=-1) - 1).abs().max().item()
            invalid["quaternion_norm_max_error"] = max(invalid["quaternion_norm_max_error"], quat_error)
            reward_error += (reward - rewards[index]).double().square().sum().item()
            terminal_error += (logits.sigmoid() - terminated[index]).double().square().sum().item()
            transitions += len(action)
    return {"fields": {name: {"rmse": (squared / count)**.5,
                                "persistence_rmse": (baseline / count)**.5}
                       for name, (squared, baseline, count) in totals.items()},
            "reward_rmse": (reward_error / transitions)**.5,
            "termination_brier": terminal_error / transitions,
            "physical_predictions": invalid, "window_transitions": transitions}


@torch.no_grad()
def evaluate_episode_rollouts(model, dataset, device):
    reports = []
    for path, (states, actions, rewards, terminated) in zip(dataset.paths, dataset.episodes):
        predicted = torch.as_tensor(states[:1], device=device)
        predictions = []
        for action in actions:
            predicted, _, _ = model(predicted, torch.as_tensor(action[None], device=device))
            if not torch.isfinite(predicted).all():
                raise FloatingPointError("Non-finite full-episode model rollout")
            predictions.append(predicted[0].cpu().numpy())
        predictions = np.asarray(predictions)
        metrics = {}
        for name, (begin, end) in dataset.layout.items():
            error = predictions[:, begin:end].astype(np.float64) - states[1:, begin:end]
            baseline = states[0, begin:end].astype(np.float64) - states[1:, begin:end]
            metrics[name] = {"rmse": float(np.sqrt(np.mean(error**2))),
                             "persistence_rmse": float(np.sqrt(np.mean(baseline**2)))}
        reports.append({"episode": str(path), "steps": len(actions), "fields": metrics})
    return reports
