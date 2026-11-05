import importlib.util
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/export_disect_trajectory.py'
spec = importlib.util.spec_from_file_location('export_disect_trajectory', SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def test_exported_intervals_reproduce_disect_substep_displacement():
    frequency, amplitudes, pressing = 7., [.02, .01, .03, 0., .02], [-.05, -.08, -.04, -.06, -.05]
    dt, duration = 1e-5, .6
    trajectory = exporter.export(frequency, amplitudes, pressing, duration, dt, .001, .02)
    assert len(trajectory.linear_velocity) == 600 and trajectory.duration == duration
    substeps = np.arange(round(duration / dt)) * dt
    velocity = exporter.slicing_velocity(substeps, frequency, amplitudes, pressing,
                                         np.linspace(0., duration, 5), .02)
    # DiSECt holds each substep velocity for dt; position after k substeps is the prefix sum.
    for k in (0, 1000, 25000, 60000):
        expected = velocity[:k].sum(0) * dt
        actual = trajectory.displacement(torch.tensor([k * dt], dtype=torch.float64))[0].numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    assert velocity[:, 0].max() == velocity[:, 0].min() == 0.


def test_slicing_velocity_matches_the_disect_formula_at_one_time():
    times = np.linspace(0., .6, 5)
    t = .23
    weights = np.exp(-(t - times) ** 2 / .02)
    v = exporter.slicing_velocity([t], 5., [.02] * 5, [-.05] * 5, times, .02)[0]
    np.testing.assert_allclose(v, [0., weights.sum() * -.05, np.sin(5. * t) * weights.sum() * .02])
