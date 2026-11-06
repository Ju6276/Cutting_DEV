"""Physical cutting-prefix states for uniformly sampled task-stage resets."""
from pathlib import Path

import torch

from .benchmark import check_contract


STAGES = ('full_task', 'before_cut', 'cutting', 'closest_cut')
SCHEMA = 'cutting-reset-bank-v1'


@torch.no_grad()
def capture_snapshot(env, history, mask):
    state, contact, motion = env.physics.material.capture_rows(mask)
    groups = {
        'frame': {k: v[mask] for k, v in env.current.items()},
        'material_state': state, 'material_contact': contact, 'material_motion': motion,
        'feedback': {k: v[mask] for k, v in env.physics.feedback.items()},
        'history': {'state': history[mask]},
    }
    return {f'{group}/{key}': value.detach().cpu().clone()
            for group, fields in groups.items() for key, value in fields.items()}


@torch.no_grad()
def restore_snapshot(env, rows, mask, *, new_episode=True):
    """Restore selected worlds; recorded physical clocks keep the prefix time."""
    groups = {}
    for key, value in rows.items():
        group, field = key.split('/', 1)
        groups.setdefault(group, {})[field] = value.to(env.device)
    current = {k: v.clone() for k, v in env.current.items()}
    for key, value in groups['frame'].items():
        current[key][mask] = value
    if new_episode:
        current['step_count'][mask] = 0
        current['success_counter'][mask] = 0
    needs_reset = env.needs_reset.clone()
    env.physics.robot.reset(mask)
    env.restore_frame(current)
    env.physics.target[mask] = groups['frame']['target_pose']
    env.needs_reset[~mask] = needs_reset[~mask]
    env.physics.material.restore_rows(mask, tuple(groups[k] for k in
        ('material_state', 'material_contact', 'material_motion')))
    for key, value in groups['feedback'].items():
        env.physics.feedback[key][mask] = value
    # The next MuJoCo step reconstructs derived dynamics from q/qdot/warmstart.
    return groups['history']['state'].clone()


def stage_masks(env):
    frame = env.current
    dims = len(env.heads.dims)
    error = frame['observation'][:, :dims].abs() * frame['observation'].new_tensor(env.heads.max_distance)
    goal = (error < error.new_tensor(env.environment['goal_threshold'])).all(-1)
    weakened = (frame['cut_spring_ke'] < frame['spring_reference_ke']).any(-1)
    valid = ~env.needs_reset
    return {'before_cut': valid & ~weakened & ~goal,
            'cutting': valid & weakened}


class CuttingResetBank:
    def __init__(self, path, contract):
        self.path = Path(path)
        saved = torch.load(self.path, map_location='cpu', weights_only=True)
        if saved.get('schema') != SCHEMA:
            raise ValueError('Unknown cutting reset-bank schema')
        check_contract(saved['contract'], contract)
        if tuple(saved['states']) != STAGES:
            raise ValueError('Reset bank must contain every cutting stage in order')
        self.states = saved['states']
        self.metadata = {key: saved[key] for key in ('schema', 'split', 'seed', 'provenance')}
        self.metadata['counts'] = {}
        keys = None
        for stage, rows in self.states.items():
            if keys is None:
                keys = set(rows)
            if set(rows) != keys or 'history/state' not in rows:
                raise ValueError(f'Inconsistent snapshot fields in {stage}')
            size = len(rows['history/state'])
            if not size or any(len(v) != size or not torch.isfinite(v).all() for v in rows.values()):
                raise ValueError(f'Empty, nonfinite or misaligned reset stage: {stage}')
            self.metadata['counts'][stage] = size

    def sample(self, count, generator=None, stage=None):
        kinds = (torch.randint(len(STAGES), (count,), generator=generator) if stage is None
                 else torch.full((count,), STAGES.index(stage), dtype=torch.long))
        samples = {}
        for i, name in enumerate(STAGES):
            selected = kinds == i
            n = int(selected.sum())
            if not n:
                continue
            rows = self.states[name]
            index = torch.randint(len(rows['history/state']), (n,), generator=generator)
            for key, value in rows.items():
                if key not in samples:
                    samples[key] = torch.empty((count, *value.shape[1:]), dtype=value.dtype)
                samples[key][selected] = value[index]
        return samples, kinds


class BankEvaluationEnvironment:
    """Physical evaluator reset category; policy observations remain native."""
    def __init__(self, env, bank, stage, seed):
        self.env, self.bank, self.stage = env, bank, stage
        self.generator = torch.Generator().manual_seed(seed)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.worlds, device=self.device, dtype=torch.bool)
        rows, _ = self.bank.sample(int(mask.sum()), self.generator, self.stage)
        restore_snapshot(self.env, rows, mask)
        return self.env.current
