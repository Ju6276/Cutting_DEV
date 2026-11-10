import torch

from scripts.legacy.perturbed_eval import CaseNoise, compare_cases, make_cases, summarize_cases


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
             'mean_material_force_n': f / 4, 'simulation_seconds': 1.0, 'safety_margin': f / 50}
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


def write_run(directory, cases, rows, unfinished=()):
    import json
    directory.mkdir()
    (directory / 'manifest.json').write_text(json.dumps({'cases': cases}))
    (directory / 'summary.json').write_text(json.dumps({'unfinished_cases': list(unfinished)}))
    (directory / 'cases.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_compare_checks_configurations_and_completion(tmp_path):
    import importlib.util
    import pytest
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        'evaluate_perturbed', Path(__file__).resolve().parents[1] / 'scripts/legacy/evaluate_perturbed.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = make_cases(4, 3_000_000)
    good = rows(['success'] * 4, [30.] * 4)
    write_run(tmp_path / 'a', cases, good)
    write_run(tmp_path / 'b', make_cases(4, 4_000_000), good)
    write_run(tmp_path / 'c', cases, good[:3], unfinished=[3])
    write_run(tmp_path / 'd', cases, good)
    manifest, loaded = module.load_run(tmp_path / 'a')
    assert len(loaded) == 4
    with pytest.raises(ValueError, match='not every predefined case'):
        module.load_run(tmp_path / 'c')
    args = type('Args', (), {'a': tmp_path / 'a', 'b': tmp_path / 'b', 'output': tmp_path / 'out.json'})
    with pytest.raises(ValueError, match='different case configurations'):
        module.compare(args)
    args.b = tmp_path / 'd'
    module.compare(args)


def load_script():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        'evaluate_perturbed_script', Path(__file__).resolve().parents[1] / 'scripts/legacy/evaluate_perturbed.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_perturbation_is_added_to_the_executed_action():
    module = load_script()

    class Spec:
        @staticmethod
        def execute(action):
            return action.clamp(-1., 1.)

    raw = torch.tensor([[1.6, -1.4, 0.2, 1.0, -1.0, 0.0]])
    noise = torch.tensor([[-0.3, 0.3, 0.3, 0.3, -0.3, 0.5]])
    result = module.perturbed(Spec, raw, noise)
    # A saturated raw output (1.6) no longer absorbs a perturbation toward the interior.
    torch.testing.assert_close(result, torch.tensor([[0.7, -0.7, 0.5, 1.0, -1.0, 0.5]]))


def test_compare_refuses_mixed_perturbation_points(tmp_path):
    import json
    import pytest
    module = load_script()
    cases = make_cases(4, 3_000_000)
    good = rows(['success'] * 4, [30.] * 4)
    write_run(tmp_path / 'legacy', cases, good)
    write_run(tmp_path / 'new', cases, good)
    manifest = json.loads((tmp_path / 'new' / 'manifest.json').read_text())
    manifest['perturbation_point'] = module.PERTURBATION_POINT
    (tmp_path / 'new' / 'manifest.json').write_text(json.dumps(manifest))
    args = type('Args', (), {'a': tmp_path / 'new', 'b': tmp_path / 'legacy', 'output': tmp_path / 'out.json'})
    with pytest.raises(ValueError, match='different points'):
        module.compare(args)


def test_legacy_rows_recover_impulse_from_mean_force_and_time():
    from scripts.legacy.perturbed_eval import with_impulse
    legacy = [{'id': 0, 'mean_material_force_n': 6.0, 'simulation_seconds': 0.5}]
    recorded = [{'id': 1, 'mean_material_force_n': 6.0, 'simulation_seconds': 0.5, 'material_impulse_ns': 2.9}]
    assert with_impulse(legacy)[0]['material_impulse_ns'] == 3.0
    assert with_impulse(recorded)[0]['material_impulse_ns'] == 2.9
    summary = summarize_cases(rows(['success'] * 4, [30.] * 4))
    assert summary['material_impulse_ns_mean'] == 7.5
