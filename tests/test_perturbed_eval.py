import torch

from cutting_world_model.perturbed_eval import CaseNoise, compare_cases, make_cases, summarize_cases


def test_cases_are_fixed_and_noise_does_not_depend_on_the_policy():
    first, again = make_cases(64, 3_000_000), make_cases(64, 3_000_000)
    assert first == again
    assert {c['mode'] for c in first} == {'iid', 'persistent'}
    other = make_cases(64, 4_000_000)
    assert first != other
    case = next(c for c in first if c['mode'] == 'iid')
    a, b = CaseNoise(case), CaseNoise(case)
    seq_a = torch.stack([a(t) for t in range(20)])
    seq_b = torch.stack([b(t) for t in range(20)])
    assert torch.equal(seq_a, seq_b)
    assert (seq_a[:case['start_step']] == 0).all()
    persistent = next(c for c in first if c['mode'] == 'persistent')
    noise = CaseNoise(persistent)
    later = [noise(t) for t in range(persistent['start_step'], persistent['start_step'] + 3)]
    assert all(torch.equal(later[0], x) for x in later)


def rows(outcomes, force):
    return [{'id': i, 'outcome': o, 'steps': 18 + (o != 'success'), 'max_material_force_n': f,
             'mean_material_force_n': f / 4, 'safety_margin': f / 50}
            for i, (o, f) in enumerate(zip(outcomes, force))]


def test_case_paired_comparison():
    base = rows(['success'] * 30 + ['collision'] * 30, [30.] * 60)
    better = rows(['success'] * 55 + ['collision'] * 5, [28.] * 60)
    result = compare_cases(better, base)
    assert result['all_cases']['success']['outcome'] == 'improved'
    assert result['all_cases']['collision']['outcome'] == 'improved'
    assert result['all_cases']['max_material_force_n']['outcome'] == 'improved'
    assert result['both_success']['cases'] == 30
    summary = summarize_cases(base)
    assert summary['success_rate'] == 0.5 and summary['collision_rate'] == 0.5
    try:
        compare_cases(better[:-1], base)
    except ValueError:
        pass
    else:
        raise AssertionError('mismatched cases must be rejected')
