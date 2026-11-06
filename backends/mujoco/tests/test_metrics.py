import pytest
import torch

from cutting_gpu_backend.metrics import StepMetrics


def run(forces, velocities, active=None, dt=.002):
    worlds = forces.shape[1]
    metrics = StepMetrics(worlds, 'cpu')
    for t in range(forces.shape[0]):
        twist = torch.zeros(worlds, 3, dtype=torch.float64)
        twist[:, 1] = velocities[t]
        mask = torch.ones(worlds, dtype=torch.bool) if active is None else active[t]
        metrics.accumulate(forces[t], twist, mask, dt)
    return metrics.as_dict()


def test_impulse_and_work_integrate_synthetic_sequences():
    forces = torch.tensor([[0.], [2.], [4.], [6.]], dtype=torch.float64)
    velocities = torch.tensor([[-.05], [-.05], [.02], [-.1]], dtype=torch.float64)
    result = run(forces, velocities)
    assert float(result['material_impulse'][0]) == pytest.approx(.002 * 12)
    net = .002 * (2 * .05 + 4 * -.02 + 6 * .1)
    gross = .002 * (2 * .05 + 4 * .02 + 6 * .1)
    assert float(result['knife_work'][0]) == pytest.approx(net)
    assert float(result['knife_gross_work'][0]) == pytest.approx(gross)
    assert gross > net


def test_pressing_down_into_material_does_positive_work():
    # DiSECt y is up; the coupling applies +force_magnitude along +y on the knife.
    forces = torch.full((10, 1), 5., dtype=torch.float64)
    pressing = run(forces, torch.full((10, 1), -.05, dtype=torch.float64))
    lifting = run(forces, torch.full((10, 1), .05, dtype=torch.float64))
    assert float(pressing['knife_work'][0]) == pytest.approx(5 * .05 * .02)
    assert float(lifting['knife_work'][0]) == pytest.approx(-5 * .05 * .02)
    assert float(pressing['knife_gross_work'][0]) == float(lifting['knife_gross_work'][0])


def test_inactive_worlds_do_not_accumulate():
    forces = torch.full((3, 2), 1., dtype=torch.float64)
    velocities = torch.full((3, 2), -1., dtype=torch.float64)
    active = torch.tensor([[True, True], [True, False], [True, False]])
    result = run(forces, velocities, active)
    torch.testing.assert_close(result['material_impulse'], torch.tensor([.006, .002], dtype=torch.float64))
    torch.testing.assert_close(result['knife_work'], torch.tensor([.006, .002], dtype=torch.float64))


def test_safety_margin_is_max_ratio_and_freezes_after_collision():
    metrics = StepMetrics(2, 'cpu')
    limits = torch.tensor([10., 10., 10., 1., 1., 1.], dtype=torch.float64)
    observed = torch.tensor([[2., 0, 0, .5, 0, 0], [5., 0, 0, 0, 0, 1.5]], dtype=torch.float64)
    metrics.check_wrench(observed, limits, torch.tensor([True, True]))
    torch.testing.assert_close(metrics.safety_margin, torch.tensor([.5, 1.5], dtype=torch.float64))
    metrics.check_wrench(observed * 3, limits, torch.tensor([True, False]))
    torch.testing.assert_close(metrics.safety_margin, torch.tensor([1.5, 1.5], dtype=torch.float64))
    metrics.zero_()
    assert not metrics.safety_margin.any() and not metrics.knife_work.any()
