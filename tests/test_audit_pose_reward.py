"""Audit comparison helper: NaN/Inf must fail instead of slipping past a threshold."""
import importlib.util
from pathlib import Path
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    'audit_pose_reward', Path(__file__).resolve().parents[1] / 'scripts/audit_pose_reward.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_close_values_pass():
    ref = torch.tensor([1., 2.], dtype=torch.float64)
    assert audit.checked_error('x', ref, ref + 1e-7, 1e-6) == pytest.approx(1e-7)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
@pytest.mark.parametrize('where', ['reference', 'candidate', 'extra'])
def test_non_finite_values_fail(bad, where):
    ref = torch.tensor([1., 2.], dtype=torch.float64)
    cand = ref.clone()
    extra = [('progress', torch.zeros(2, dtype=torch.float64))]
    {'reference': ref, 'candidate': cand, 'extra': extra[0][1]}[where][0] = bad
    with pytest.raises(FloatingPointError):
        audit.checked_error('x', ref, cand, 1e-3, extra=extra)


def test_large_error_fails():
    ref = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(AssertionError):
        audit.checked_error('x', ref, ref + 1., 1e-3)
