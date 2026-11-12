"""Frozen history-based world model with the upstream MPPI action optimizer."""
from dataclasses import asdict, dataclass
import importlib.metadata
import json
import math
import time
from pathlib import Path

import torch

from .benchmark import copy_frame, synchronize
from .control_state import ControlStateSpec
from .checkpoint import checkpoint_identity, check_checkpoint_pair
from .world_model import CuttingWorldModel
from .policy import SliceItCritic
from .task_heads import pose_error


@dataclass(frozen=True)
class MPPIConfig:
    horizon: int = 8
    samples: int = 512
    refinements: int = 1
    temperature: float = 1.
    noise_std: float = .5
    discount: float = .99
    terminal_value: str = 'none'

    def __post_init__(self):
        for name in ('horizon', 'samples', 'refinements'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('temperature', 'noise_std'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.discount) or not 0 < self.discount <= 1:
            raise ValueError('discount must be in (0, 1]')
        if self.terminal_value not in ('none', 'critic'):
            raise ValueError('terminal_value must be none or critic')


class ModelRollout:
    """Augmented state: physical history, survival, absorbed flag, transition cost.

    MPPI evaluates running_cost after dynamics. Carrying the transition cost
    preserves the old success counter and survival weight used for its reward.
    """
    def __init__(self, wm, discount, critic=None):
        self.wm = wm
        self.spec = wm.spec
        self.discount = discount
        self.goal = None
        self.critic = critic

    def initial(self, state, goal):
        if state.shape[0] != 1 or goal.shape != (1, getattr(self.spec, 'goal_dim', 7)):
            raise ValueError('MPPI plans for one physical environment at a time')
        self.goal = goal
        return torch.cat((state, state.new_ones((1, 1)), state.new_zeros((1, 2))), -1)[0]

    def task_action(self, action):
        low, high = self.spec.action_low, self.spec.action_high
        return low + (action + 1) * (high - low) / 2

    def dynamics(self, augmented, action, t):
        state = augmented[..., :-3]
        survival = augmented[..., -3]
        absorbed = augmented[..., -2].bool()
        goal = self.goal.expand(state.shape[0], -1)
        predicted, reward, logits = self.wm.predict(state, self.task_action(action), goal)
        next_state = torch.where(absorbed[:, None], state, predicted)
        reward = torch.where(absorbed, 0., reward)
        collision = (self.spec.field(next_state, 'peak_wrench') > self.spec.force_limits).any(-1)
        if hasattr(self.spec, 'hard_termination'):
            done = self.spec.hard_termination(next_state)
        else:
            done, _, _, _ = self.spec.heads.termination(
                self.spec.field(next_state, 'observation'),
                self.spec.field(state, 'success_counter'), collision)
        absorbed = absorbed | done
        probability = torch.where(absorbed, 1., logits.sigmoid())
        cost = -(self.discount ** t) * survival * reward
        next_survival = survival * (1 - probability)
        result = torch.cat((next_state, next_survival[:, None],
                            absorbed.to(state.dtype)[:, None], cost[:, None]), -1)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError(f'Non-finite WM prediction in MPPI at horizon step {t}')
        return result

    @staticmethod
    def running_cost(state, action, t):
        return state[..., -1]

    def terminal_cost(self, augmented, steps):
        survival = augmented[:, -3]
        value = torch.zeros_like(survival)
        active = survival > 0
        if self.critic is not None and bool(active.any()):
            state = augmented[active, :-3]
            goal = self.goal.expand(len(state), -1)
            error = pose_error(goal, self.spec.field(state, 'previous_ee_pose'))
            value[active] = self.critic(self.spec.canonical(state), goal, error)
        cost = -(self.discount ** steps) * survival * value
        if not bool(torch.isfinite(cost).all()):
            raise FloatingPointError('Non-finite terminal critic prediction in MPPI')
        return cost

    def final_cost(self, states, actions):
        # Upstream deterministic MPPI stores (1, candidates, horizon, state).
        return self.terminal_cost(states[0, :, -1], states.shape[-2])


class MPPIPolicy:
    def __init__(self, checkpoint, env, config, paired_checkpoint=None):
        from pytorch_mppi import MPPI

        distribution = importlib.metadata.distribution('pytorch-mppi')
        source = json.loads(distribution.read_text('direct_url.json') or '{}')
        revision = source.get('vcs_info', {}).get('commit_id')
        if revision != 'e04a569cd4215e705f9013145c496fc59cb25ed6':
            raise ValueError('Install the pinned planner from requirements-mpc.txt')
        if env.worlds != 1:
            raise ValueError('MPPI benchmark requires one physical world')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.checkpoint_identity = checkpoint_identity(saved)
        paired_manifest = None
        if paired_checkpoint is not None:
            paired = torch.load(paired_checkpoint, map_location='cpu', weights_only=False)
            groups = check_checkpoint_pair(saved, paired, config.terminal_value == 'critic')
            paired_manifest = {
                'path': str(Path(paired_checkpoint).resolve()),
                'identity': checkpoint_identity(paired), 'equal_weight_groups': groups}
            del paired
        metadata = ControlStateSpec.metadata_from_frame(env.metadata, copy_frame(env.current))
        if metadata != saved['metadata']:
            raise ValueError('WM checkpoint and benchmark task metadata differ')
        self.wm = CuttingWorldModel(metadata).to(env.device).eval()
        self.weights = saved['learner']['world_model']
        self.wm.load_state_dict(self.weights)
        self.wm.requires_grad_(False)
        self.spec = self.wm.spec
        self.config = config
        self.critic = None
        self.critic_weights = None
        if config.terminal_value == 'critic':
            self.critic = SliceItCritic(self.spec.state_dim).to(env.device).eval()
            self.critic_weights = saved['learner']['critic']
            self.critic.load_state_dict(self.critic_weights)
            self.critic.requires_grad_(False)
        self.rollout = ModelRollout(self.wm, config.discount, self.critic)
        self.plans = []
        self.episode = -1
        # All six optimizer coordinates are centered in [-1, 1]; the affine
        # map to the task's asymmetric bounds is applied only at the WM/robot.
        self.planner = MPPI(
            self.rollout.dynamics, self.rollout.running_cost, self.spec.state_dim + 3,
            noise_sigma=torch.eye(6, device=env.device) * config.noise_std ** 2,
            num_samples=config.samples, horizon=config.horizon, device=env.device,
            lambda_=config.temperature, u_min=-torch.ones(6, device=env.device),
            u_max=torch.ones(6, device=env.device),
            U_init=torch.zeros(config.horizon, 6, device=env.device),
            terminal_state_cost=self.rollout.final_cost if self.critic is not None else None,
            step_dependent_dynamics=True)
        self.manifest = {
            **asdict(config), 'implementation': 'UM-ARM-Lab/pytorch_mppi',
            'package_version': distribution.version,
            'upstream_revision': revision,
            'policy_updates': self.checkpoint_identity['policy_updates'],
            'wm_updates': self.checkpoint_identity['wm_updates'],
            'paired_checkpoint': paired_manifest,
            'terminal_weighting': 'task_absorption_and_learned_survival',
            'initial_plan': 'action_bounds_midpoint', 'action_coordinates': 'normalized_minus1_plus1',
            'information': {
                'state_fields': list(self.spec.base_fields),
                'history_frames': self.spec.history_length,
                'dynamics_features': self.spec.dynamics_dim,
            },
        }

    def reset(self, frame):
        self.state = self.spec.pack_batch(frame).to(next(self.wm.parameters()))
        self.planner.U.zero_()
        self.episode += 1
        self.step = 0

    @torch.no_grad()
    def act(self, frame):
        synchronize(self.state.device)
        started = time.monotonic()
        initial = self.rollout.initial(self.state, frame['target_pose'].float())
        for i in range(self.config.refinements):
            normalized = self.planner.command(
                initial, shift_nominal_trajectory=(self.step > 0 and i == 0))
        # Evaluate the returned weighted-mean plan; it need not equal a sample.
        prediction = initial[None]
        predicted_return = initial.new_zeros(())
        peak_ratio = initial.new_zeros(())
        for t, candidate in enumerate(self.planner.U):
            previous_survival = prediction[0, -3]
            prediction = self.rollout.dynamics(prediction, candidate[None], t)
            predicted_return -= prediction[0, -1]
            ratio = (self.spec.field(prediction[:, :-3], 'peak_wrench') / self.spec.force_limits).max()
            peak_ratio = torch.maximum(peak_ratio, torch.where(previous_survival > 0, ratio, 0.))
        terminal_return = -self.rollout.terminal_cost(prediction, self.config.horizon)[0]
        synchronize(self.state.device)
        self.plans.append({
            'episode': self.episode, 'step': self.step,
            'planning_seconds': time.monotonic() - started,
            'predicted_task_return': float(predicted_return),
            'predicted_terminal_return': float(terminal_return),
            'predicted_total_return': float(predicted_return + terminal_return),
            'predicted_sensor_limit_ratio': float(peak_ratio),
            'predicted_end_survival': float(prediction[0, -3]),
            'predicted_task_terminated': bool(prediction[0, -2]),
            'sample_cost_min_with_mppi_correction': float(self.planner.cost_total.min()),
            'sample_cost_std_with_mppi_correction': float(self.planner.cost_total.std(unbiased=False)),
            'effective_candidates': float(1 / self.planner.omega.square().sum()),
        })
        self.step += 1
        return self.spec.execute(self.rollout.task_action(normalized))[None]

    def observe(self, frame):
        self.state = self.spec.pack_batch_transition(frame, self.state)

    def unchanged(self):
        wm_same = all(torch.equal(value.cpu(), self.weights[name])
                      for name, value in self.wm.state_dict().items())
        critic_same = self.critic is None or all(
            torch.equal(value.cpu(), self.critic_weights[name])
            for name, value in self.critic.state_dict().items())
        return wm_same and critic_same
