"""Frozen-policy robustness under predefined action perturbations (PLAN, A control pair).

Every case is fixed before any policy runs: index, perturbation scale ``sigma``, first
perturbed step, mode ('iid': fresh N(0, I) per step from the case's own stream;
'persistent': one N(0, I) direction for the whole episode) and seeds. Two policies execute
the same cases and are compared case by case. Each case runs to its own end; success,
force limit, workspace and time-limit outcomes are all kept.
"""
import torch

from cutting_world_model.reliability import metric, paired_bootstrap

SIGMAS = (0.3, 0.6, 1.0)
START_STEPS = (0, 5, 10)
MODES = ('iid', 'persistent')
OUTCOMES = ('success', 'collision', 'outside', 'truncated')
NOISE_SEED_OFFSET = 10_000_000


def make_cases(count, seed_base):
    cases = []
    for index in range(count):
        gen = torch.Generator().manual_seed(seed_base + index)
        pick = lambda options: options[int(torch.randint(len(options), (1,), generator=gen))]
        sigma, start, mode = pick(SIGMAS), pick(START_STEPS), pick(MODES)
        direction = torch.randn(6, generator=gen).tolist() if mode == 'persistent' else None
        cases.append({'case': index, 'sigma': sigma, 'start_step': start, 'mode': mode,
                      'direction': direction, 'noise_seed': seed_base + NOISE_SEED_OFFSET + index})
    return cases


class CaseNoise:
    """Perturbation of one case; the i.i.d. stream advances once per perturbed step, so
    identical cases give identical perturbation sequences for every policy."""

    def __init__(self, case):
        self.case = case
        self.gen = torch.Generator().manual_seed(case['noise_seed'])
        self.direction = None if case['direction'] is None else torch.tensor(case['direction'])

    def __call__(self, step):
        if step < self.case['start_step']:
            return torch.zeros(6)
        z = self.direction if self.case['mode'] == 'persistent' else torch.randn(6, generator=self.gen)
        return self.case['sigma'] * z


def outcome(success, collision, outside, truncated):
    if success:
        return 'success'
    if collision:
        return 'collision'
    if outside:
        return 'outside'
    if truncated:
        return 'truncated'
    raise ValueError('Episode ended without an outcome')


def with_impulse(rows):
    """Rows recorded before the impulse field: mean force over actual simulated time times that time."""
    return [r if 'material_impulse_ns' in r else
            {**r, 'material_impulse_ns': r['mean_material_force_n'] * r['simulation_seconds']} for r in rows]


def summarize_cases(rows):
    rows = with_impulse(rows)
    n = len(rows)
    out = {'cases': n, **{f'{o}_rate': sum(r['outcome'] == o for r in rows) / n for o in OUTCOMES}}
    for key in ('max_material_force_n', 'mean_material_force_n', 'material_impulse_ns', 'safety_margin'):
        out[key + '_mean'] = sum(r[key] for r in rows) / n
    done = [r for r in rows if r['outcome'] == 'success']
    out['success_steps_mean'] = sum(r['steps'] for r in done) / len(done) if done else None
    return out


def compare_cases(rows_a, rows_b, samples=10000, seed=0):
    """Case-paired A − B: outcome rates and force over all cases; steps and force over the
    cases both policies complete successfully."""
    ids_a = {r['id'] for r in rows_a}
    ids_b = {r['id'] for r in rows_b}
    if ids_a != ids_b:
        raise ValueError('The two evaluations ran different cases')
    flag = lambda rows: [{**r, **{o: float(r['outcome'] == o) for o in OUTCOMES}} for r in with_impulse(rows)]
    a, b = flag(rows_a), flag(rows_b)
    result = {'all_cases': {}, 'both_success': {}}
    for key, lower in [('success', False), ('collision', True), ('outside', True), ('truncated', True),
                       ('max_material_force_n', True), ('mean_material_force_n', True),
                       ('material_impulse_ns', True), ('safety_margin', True)]:
        result['all_cases'][key] = paired_bootstrap(a, b, key, lower, samples, seed)
    both = {r['id'] for r in a if r['success']} & {r['id'] for r in b if r['success']}
    a2 = [r for r in a if r['id'] in both]
    b2 = [r for r in b if r['id'] in both]
    result['both_success']['cases'] = len(both)
    for key in ('steps', 'max_material_force_n', 'mean_material_force_n', 'material_impulse_ns', 'safety_margin'):
        result['both_success'][key] = paired_bootstrap(a2, b2, key, True, samples, seed)
    return result


__all__ = ['make_cases', 'CaseNoise', 'outcome', 'summarize_cases', 'compare_cases', 'metric']
