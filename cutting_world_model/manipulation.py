"""SliceIt task contract for the external supermimic_manipulation learner."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from .control_state import ControlStateSpec
from .sac_gpu_bridge import JsonProcess, clean_python_environment

SOURCE_REVISION = 'eda7da480c6d92a68cc41c9628b7efdae08cae23'


@dataclass
class EnvironmentConfig:
    metadata: dict
    scene: str
    capture: str
    reset_bank: dict | None = None
    auto_reset: bool = True
    is_finite_horizon: bool = True


class CuttingMDP:
    task = 'ur5e_apple'
    reward_matches_native = True
    action_dim = 6
    context_dim = 7

    def __init__(self, metadata, device='cpu'):
        self.spec = ControlStateSpec(metadata).to(device)
        self.state_dim = self.spec.state_dim + 1  # Episode clock, advanced analytically.
        self.delta_dim = self.spec.delta_dim
        self.obs_dim = self.spec.shapes['observation'][0]
        self.input_dim = self.spec.dynamics_dim + self.action_dim + self.context_dim
        self.critic_dim = self.state_dim + self.context_dim + self.obs_dim
        self.error_floor = torch.full((self.delta_dim,), 1e-3, device=device)
        for key in ('joint_q', 'joint_qd'):
            self.error_floor[self.spec.delta_slices[key]] = 1e-2
        self.smooth_temperature = 0.01
        self.success_gradient_scale = 0.02

    @staticmethod
    def _event(hard, soft):
        return hard.to(soft.dtype) + (soft - soft.detach())

    def candidate(self, state):
        obs = self.observe(state, None)
        threshold = obs.new_tensor(self.spec.heads.env['goal_threshold'])
        maximum = obs.new_tensor(self.spec.heads.max_distance)
        error = obs[:, :len(threshold)].abs()
        margin = threshold / maximum - error
        return self._event((error * maximum < threshold).all(-1),
                           torch.sigmoid(margin / self.smooth_temperature).prod(-1))

    def events(self, state):
        obs = self.observe(state, None)
        limits = obs.new_tensor(self.spec.heads.env['workspace_limit'])
        maximum = obs.new_tensor(self.spec.heads.max_distance)
        error = obs[:, :len(limits)].abs()
        workspace = error - limits / maximum
        peak = self.spec.field(state, 'peak_wrench')
        force = peak / self.spec.force_limits - 1
        def any_event(hard, margin):
            return self._event(hard,
                1 - torch.sigmoid(-margin / self.smooth_temperature).prod(-1))
        return (any_event((peak > self.spec.force_limits).any(-1), force),
                any_event((error * maximum > limits).any(-1), workspace))

    def physical(self, state):
        return torch.cat([self.spec.field(state, key).flatten(1)
                          for key in self.spec.physical_fields], -1)

    def dynamics_action(self, action):
        executed = self.spec.execute(action)
        gains = executed[..., :4].clamp(0, 1)
        active = executed.new_tensor([
            value != 0 for value in self.spec.metadata['controller']['max_twist']])
        return torch.cat((gains, executed[..., 4:] * active), -1)

    def inputs(self, state, action, context):
        frames = state[:, :-1].reshape(-1, self.spec.history_length, self.spec.base_dim)
        history = frames.index_select(-1, self.spec.dynamics_indices)
        # The last six selected features are each frame's previous action.
        history = torch.cat((history[..., :-6], self.dynamics_action(history[..., -6:])), -1)
        return torch.cat((history.flatten(1), self.dynamics_action(action), context), -1)

    def targets(self, state, next_state):
        return self.physical(next_state) - self.physical(state)

    def state_error(self, prediction, target):
        return self.physical(prediction) - self.physical(target)

    def integrate(self, state, delta, action, context):
        old = state[:, :-1]
        fields = {key: self.spec.field(old, key)
                  for key in self.spec.physical_fields + self.spec.history_fields}
        for key in self.spec.physical_fields:
            fields[key] = fields[key] + delta[:, self.spec.delta_slices[key]].reshape_as(fields[key])
        new, _, _ = self.spec.derive(old, fields, self.spec.execute(action), context)
        counter = self.spec.field(old, 'success_counter') + self.candidate(new)
        new = new.clone()
        new[:, self.spec.layout['success_counter']] = counter[:, None]
        return torch.cat((new, state[:, -1:] + 1), -1)

    def observe(self, state, context):
        return self.spec.field(state, 'observation')

    def advance_obs(self, observation, next_state, action, context):
        return self.observe(next_state, context)

    def reward(self, state, action, next_state, context):
        observation = self.observe(next_state, context)
        collision = (self.spec.field(next_state, 'peak_wrench') > self.spec.force_limits).any(-1)
        done, success, outside, _ = self.spec.heads.termination(
            observation, self.spec.field(state, 'success_counter'), collision)
        reward = self.spec.heads.reward(observation, collision, success, outside)
        candidate = self.candidate(next_state)
        counter = self.spec.field(state, 'success_counter') + candidate
        margin = counter - self.spec.heads.env['successes_threshold']
        counted = self._event(margin > 0, torch.sigmoid(margin / 0.5))
        force_event, outside_event = self.events(next_state)
        success_event = candidate * counted * (1 - force_event)
        success_event *= float(not self.spec.heads.env['termination_on_negative_reward'])
        success_event = (success_event.detach() + self.success_gradient_scale
                         * (success_event - success_event.detach()))
        costs = self.spec.heads.env['cost']
        reward = reward + costs['done'] * (success_event - success_event.detach())
        reward = reward + costs['collision'] * (force_event - force_event.detach())
        reward = reward + costs['collision'] * (outside_event - outside_event.detach())
        # The upstream runner models a finite task horizon, including its timeout.
        timeout = next_state[:, -1] >= self.spec.metadata['environment']['steps_per_episode']
        done = done | timeout
        soft_done = 1 - (1 - success_event) * (1 - force_event) * (1 - outside_event)
        soft_done = soft_done * (~timeout).to(soft_done.dtype)
        if self.spec.heads.env['termination_on_negative_reward']:
            soft_done = soft_done * 0
        return reward, self._event(done, soft_done)

    def critic_features(self, state, context, observation):
        return torch.cat((state, context, observation), -1)

    def contract(self):
        return {
            'task': self.task, 'state_dim': self.state_dim, 'delta_dim': self.delta_dim,
            'input_dim': self.input_dim, 'obs_dim': self.obs_dim,
            'action_dim': self.action_dim, 'context_dim': self.context_dim,
            'critic_dim': self.critic_dim, 'history': self.spec.history_length,
            'predicted_fields': list(self.spec.physical_fields),
            'dynamics_action_features': 'native gain saturation and enabled motion axes v1',
            'delta': 'physical SI increments', 'reward': 'native SliceIt',
            'finite_horizon_steps': self.spec.metadata['environment']['steps_per_episode'],
            'error_floor': self.error_floor.tolist(),
            'event_gradient': 'native forward; normalized-margin sigmoid straight-through backward',
            'smooth_temperature': self.smooth_temperature,
            'success_gradient_scale': self.success_gradient_scale,
        }


class CuttingEnvironment:
    """Batched physical worker with terminal POST captured before automatic reset."""

    def __init__(self, python, backend_root, *, worlds, scene, capture, device='cuda:0',
                 timeout=600, log=None, reset_bank=None, seed=0):
        self.device = torch.device(device)
        self.num_envs = worlds
        self._reward_sum = 0.0
        self._transitions = 0
        self._force_max = 0.0
        self._return_sum = 0.0
        self._episodes = 0
        self.completed_episodes = 0
        self.successful_episodes = 0
        worker = Path(__file__).resolve().parents[1] / 'scripts/manipulation_gpu_worker.py'
        command = [python, worker, '--backend-root', backend_root, '--worlds', str(worlds),
                   '--scene', scene, '--capture', capture, '--seed', str(seed)]
        if reset_bank is not None:
            command += ['--reset-bank', str(reset_bank)]
        self.client = JsonProcess(command, timeout=timeout,
            environment=clean_python_environment(), stderr=log)
        try:
            ready = self.client.request('init')
            self.cfg = EnvironmentConfig(ready['metadata'], str(scene), str(capture), ready.get('reset_bank'))
            self.mdp = CuttingMDP(ready['model_metadata'], self.device)
            self.sim = SimpleNamespace(data=SimpleNamespace(overflow=None))
            self.frame = self._frame(ready['frame'])
            self.state = self._reset_state(ready)
            self.context = self.frame['target_pose']
            self.post = None
            self.sim.data.overflow = self._tensor(ready['overflow'], torch.int32)
        except BaseException:
            self.close()
            raise

    def _tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _frame(self, values):
        return {key: self._tensor(value) for key, value in values.items()}

    def _pack(self, frame, previous=None):
        spec = self.mdp.spec
        state = (spec.pack_batch(frame) if previous is None
                 else spec.pack_batch_transition(frame, previous[:, :-1]))
        return torch.cat((state, frame['step_count']), -1).float()

    def _reset_state(self, reply):
        if 'history' not in reply:
            return self._pack(self.frame)
        return torch.cat((self._tensor(reply['history']), self.frame['step_count']), -1).float()

    def reset(self):
        reply = self.client.request('reset')
        self.frame = self._frame(reply['frame'])
        self.state = self._reset_state(reply)
        self.context = self.frame['target_pose']
        self.sim.data.overflow = self._tensor(reply['overflow'], torch.int32)
        return {'actor': self.mdp.observe(self.state, self.context)}, {}

    def step(self, action):
        reply = self.client.request('step', action=action.detach().cpu().tolist())
        self._reward_sum += sum(reply['reward'])
        self._transitions += len(reply['reward'])
        self._force_max = max(self._force_max, reply['material_force_max'])
        episode_metrics = reply['episode_metrics']
        if episode_metrics:
            count = sum(value for key, value in episode_metrics.items() if key.endswith('_episodes'))
            self._episodes += count
            self.completed_episodes += count
            self.successful_episodes += sum(value for key, value in episode_metrics.items()
                                            if key.endswith('_successes'))
            self._return_sum += count * episode_metrics['Episode_Metrics/return_mean']
        self.post = self._pack(self._frame(reply['post']), self.state)
        terminated = self._tensor(reply['terminated'], torch.bool)
        truncated = self._tensor(reply['truncated'], torch.bool)
        self.frame = self._frame(reply['frame'])
        done = terminated | truncated
        self.state = torch.where(done[:, None], self._reset_state(reply), self.post)
        self.context = self.frame['target_pose']
        self.sim.data.overflow = self._tensor(reply['overflow'], torch.int32)
        extras = {'log': reply['episode_metrics']}
        return ({'actor': self.mdp.observe(self.state, self.context)},
                self._tensor(reply['reward']), terminated, truncated, extras)

    def read(self):
        return self.state.clone(), self.context.clone()

    def post_step(self):
        return self.post

    def validate_observation(self, state, context, observation):
        error = (self.mdp.observe(state, context) - observation).abs().max().item()
        if error:
            raise ValueError(f'Actor observation mismatch: {error}')
        return error

    def close(self):
        self.client.close()

    def take_metrics(self):
        result = {}
        if self._transitions:
            result['Environment/reward_mean'] = self._reward_sum / self._transitions
            result['Environment/material_force_max'] = self._force_max
        if self._episodes:
            result['Episodes/return_mean'] = self._return_sum / self._episodes
            result['Episodes/count'] = self._episodes
        self._reward_sum = self._force_max = self._return_sum = 0.0
        self._transitions = self._episodes = 0
        return result


class CuttingMetricsLogger:
    def __init__(self, logger, environment):
        self.logger = logger
        self.environment = environment
        self.writer = logger.writer

    def log(self, metrics, *, step):
        self.logger.log(dict(metrics, **self.environment.take_metrics()), step=step)

    def close(self):
        self.logger.close()


class ManipulationPolicy:
    def __init__(self, python, source_root, checkpoint, contract, *, action_bounds):
        self.action_bounds = action_bounds
        worker = Path(__file__).resolve().parents[1] / 'scripts/manipulation_policy_worker.py'
        self.client = JsonProcess([python, worker, '--source-root', source_root,
                                   '--checkpoint', checkpoint], environment=clean_python_environment())
        try:
            ready = self.client.request('init', contract=contract)
            self.updates = ready['policy_updates']
        except BaseException:
            self.close()
            raise

    def reset(self, frame):
        pass

    def observe(self, frame):
        pass

    def act(self, frame):
        observation = frame['observation']
        reply = self.client.request('act', observation=observation.cpu().tolist())
        action = torch.as_tensor(reply['action'], device=observation.device, dtype=torch.float32)
        if not torch.isfinite(action).all():
            raise ValueError('Non-finite manipulation policy action')
        low, high = action.new_tensor(self.action_bounds)
        return torch.maximum(torch.minimum(action, high), low)

    def unchanged(self):
        return self.client.request('unchanged')['unchanged']

    def close(self):
        self.client.close()
