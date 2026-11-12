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


def test_rigid_wrench_transform_rotates_force_and_moment():
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


def test_power_invariance_and_roundtrip_with_translation_and_rotation():
    from cutting_gpu_backend.coupling import inverse_pose
    from cutting_gpu_backend.controller.fdcc import quat_apply
    pose = torch.tensor([.2, -.4, .3, 0., 0., 2**-.5, 2**-.5], dtype=torch.float64)
    wrench = torch.tensor([[2., -3., 5., .7, 1.1, -.2]], dtype=torch.float64)
    velocity = torch.tensor([[.4, -.2, .1]], dtype=torch.float64)
    omega = torch.tensor([[.3, .7, -.9]], dtype=torch.float64)
    out = rigid_wrench_transform(wrench, pose)
    rotated_omega = quat_apply(pose[3:].expand(1, -1), omega)
    # Velocity of the target origin = velocity at source minus omega x offset.
    target_velocity = quat_apply(pose[3:].expand(1, -1), velocity) + torch.linalg.cross(
        pose[:3].expand(1, -1), rotated_omega)
    power_source = (wrench[:, :3]*velocity).sum() + (wrench[:, 3:]*omega).sum()
    power_target = (out[:, :3]*target_velocity).sum() + (out[:, 3:]*rotated_omega).sum()
    torch.testing.assert_close(power_source, power_target)
    torch.testing.assert_close(rigid_wrench_transform(out, inverse_pose(pose)), wrench)


def test_net_pure_couple_does_not_create_force_and_scalar_stays_legacy():
    from cutting_gpu_backend.coupling import SliceItCoupling, source_wrench_transform, observation_wrench
    from types import SimpleNamespace
    identity = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]], dtype=torch.float64)
    ee = identity.clone()
    ee[:, 2] = .3
    pure_couple = torch.tensor([[0., 0., 0., 0., 2., 0.]], dtype=torch.float64)
    coupling = SliceItCoupling(identity, .04)
    coupling.filter_initialized = True
    coupling.wrench_filter = SimpleNamespace(update=lambda x: x)
    contacts = torch.zeros(1, 2, 3, dtype=torch.float64)
    net = coupling.feedback(contacts, identity, ee, torch.zeros_like(pure_couple), pure_couple)
    torch.testing.assert_close(net['observation_wrench'], pure_couple, atol=0, rtol=0)
    # Scalar continues to reproduce the exact historical conversion, including its convention.
    legacy = coupling.feedback(contacts, identity, ee, pure_couple)
    torch.testing.assert_close(legacy['observation_wrench'], source_wrench_transform(pure_couple, ee))
    assert legacy['observation_wrench'][0, 0] == pytest.approx(-.6)
    torch.testing.assert_close(observation_wrench(pure_couple, ee, net=True), pure_couple)


def test_contact_moment_uses_force_evaluation_positions_and_world_signs():
    from types import SimpleNamespace
    from cutting_gpu_backend.batched_disect import BatchedDiSECt
    # One force per world, at hand-computable points. Current q is deliberately unrelated.
    contact_q = torch.tensor([[.2, 0., 0.], [.4, 0., 0.], [0., 0., .1], [0., 0., .3]])
    obj = SimpleNamespace(
        worlds=2,
        state=SimpleNamespace(knife_f=torch.tensor([[0., -10., 0.], [5., 0., 0.]]),
                              knife_contact_q=contact_q, particle_q=contact_q+100),
        model=SimpleNamespace(cut_edge_contact_coord=torch.tensor([.5, .5])),
        _contact_edge=torch.tensor([0, 1]), _contact_first=torch.tensor([0, 2]),
        _contact_second=torch.tensor([1, 3]))
    actual = BatchedDiSECt.knife_wrench(obj)
    expected = torch.tensor([[0., 10., 0., 0., 0., 3.], [-5., 0., 0., 0., -1., 0.]])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('value', [float('inf'), float('nan'), True, 0., -1.])
def test_invalid_friction_fails_before_loading_scene(value):
    from cutting_gpu_backend.batched_disect import BatchedDiSECt
    with pytest.raises(ValueError, match='positive and finite'):
        BatchedDiSECt('/does/not/exist', 1, friction_velocity=value)


def test_saw_summary_uses_simultaneous_samples_not_component_envelope_or_endpoint():
    from scripts.measure_sawing import summarize_samples
    # Two orthogonal 10 N spikes inside ONE control step, followed by a zero-force endpoint.
    series = [
        [.002, 1, 10., 0., 0., 10., 0., 0., 0., 0., 0., 10., 0],
        [.004, 1, 0., 10., 0., 0., 10., 0., 0., 2., 0., 10., 0],
        [.006, 1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 100],
        [.008, 1, 999., 0., 0., 999., 0., 0., 0., 0., 0., 999., 100]]
    result = summarize_samples(series, 100)
    assert result['phase_samples'] == 3  # post-release spike excluded from every statistic
    assert result['peak_material_force_n'] == 10
    assert result['peak_sensor_force_n'] == 10  # not sqrt(10**2 + 10**2)
    assert result['mean_material_force_n'] == pytest.approx(20/3)
    assert result['mean_net_vertical_n'] == pytest.approx(10/3)
    assert result['material_impulse_ns'] == pytest.approx(.04)
    assert result['cut_through_time_s'] == .006
