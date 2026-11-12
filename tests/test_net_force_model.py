"""Net material wrench option: rigid transform, configuration plumbing, contact-point moment."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'backends/mujoco'))
from cutting_gpu_backend.coupling import rigid_wrench_transform  # noqa: E402
from cutting_world_model.multi_env import MultiConditionEnvironment  # noqa: E402


def test_rigid_wrench_transform_moves_the_reference_point():
    # Target frame = source frame shifted by p (identity rotation): a force F at the source
    # origin has moment p x F about the target origin.
    pose = torch.tensor([0., 0., .2, 0., 0., 0., 1.], dtype=torch.float64)
    wrench = torch.tensor([[10., 0., 0., 0., 0., 0.]], dtype=torch.float64)
    out = rigid_wrench_transform(wrench, pose)
    assert torch.allclose(out[0, :3], wrench[0, :3])
    assert torch.allclose(out[0, 3:], torch.tensor([0., 2., 0.], dtype=torch.float64))


def test_rigid_wrench_transform_preserves_power_pairing_under_rotation():
    q = torch.tensor([0., 0., 2 ** -.5, 2 ** -.5], dtype=torch.float64)   # 90 deg about z (x, y, z, w)
    pose = torch.cat((torch.zeros(3, dtype=torch.float64), q))
    wrench = torch.tensor([[1., 0., 0., 0., 0., 3.]], dtype=torch.float64)
    out = rigid_wrench_transform(wrench, pose)
    assert torch.allclose(out[0, :3].norm(), wrench[0, :3].norm())
    assert torch.allclose(out[0, 3:], wrench[0, 3:])            # moment along the rotation axis is unchanged
    assert abs(float(out[0, 0])) < 1e-12 and abs(abs(float(out[0, 1])) - 1.) < 1e-12


def test_group_force_model_from_config(monkeypatch):
    captured = {}

    def init(self, backend_root, groups, **kwargs):
        captured['groups'] = groups

    monkeypatch.setattr(MultiConditionEnvironment, '__init__', init)
    config = {'force_limits': [50.] * 3 + [5.] * 3, 'groups': [
        {'name': 'a', 'scene': 'a.pt', 'worlds': 2, 'force_model': 'net', 'friction_velocity': .01},
        {'name': 'b', 'scene': 'b.pt', 'worlds': 2}]}
    MultiConditionEnvironment.from_config(ROOT / 'backends/mujoco', config)
    a, b = captured['groups']
    assert (a.force_model, a.friction_velocity) == ('net', .01)
    assert (b.force_model, b.friction_velocity) == ('scalar', None)
