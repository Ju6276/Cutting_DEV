from types import SimpleNamespace

import pytest
import torch

from cutting_world_model.free_cut_evaluation import FreeCutEvaluationEnvironment
from cutting_world_model.traditional import FDCCReferencePolicy, reference_trajectory


def test_press_tracks_command_speed_and_holds_endpoint_without_retraction():
    path = reference_trajectory(.061, .03, .05, 180)
    t = torch.tensor([0., 1., .061/.03, 9.], dtype=torch.float64)
    p = path.displacement(t)
    torch.testing.assert_close(p[:, 1], torch.tensor([0., -.03, -.061, -.061], dtype=p.dtype),
                               atol=5e-5, rtol=0)
    assert (path.linear_velocity[:, 1] <= 0).all()
    assert not path.linear_velocity[:, (0, 2)].any()


def test_saw_has_same_feed_endpoint_and_known_amplitude_frequency():
    a = reference_trajectory(.061, .03, .05, 180, saw=True)
    b = reference_trajectory(.061, .03, .05, 180, saw=False)
    torch.testing.assert_close(a.linear_velocity[:, :2], b.linear_velocity[:, :2])
    t = torch.tensor([.25, .75, 9.], dtype=torch.float64)
    p = a.displacement(t)
    torch.testing.assert_close(p[:2, 2], p.new_tensor([.02, -.02]))
    assert float(p[-1, 1]) == pytest.approx(-.061)
    # A higher start adds travel at the same speed, preserving the physical endpoint.
    lifted = reference_trajectory(.076, .03, .05, 180)
    assert .076 + float(lifted.displacement(t[-1:])[0, 1]) == pytest.approx(0)
    for kwargs in ({'frequency': 0}, {'amplitude': float('nan')}, {'distance': -1}):
        opts = dict(distance=.06, speed=.03, dt=.05, horizon=180, **{})
        opts.update(kwargs)
        with pytest.raises(ValueError):
            reference_trajectory(**opts)


def test_free_cut_wrapper_preserves_fdcc_reference_dispatch():
    calls = []
    class Env:
        def step(self, action, reference=False):
            calls.append(reference)
            return {'frame': action}, 1, False, False, {}
    wrapper = FreeCutEvaluationEnvironment.__new__(FreeCutEvaluationEnvironment)
    wrapper.env = Env()
    wrapper._recordable = lambda frame: frame
    wrapper.step(torch.zeros(1, 6), reference=True)
    wrapper.step(torch.zeros(1, 6))
    assert calls == [True, False]


def test_fdcc_reference_never_needs_material_state_and_resets_phase():
    from cutting_gpu_backend.controller.fdcc import ControllerConfig
    class Physics:
        def __init__(self):
            self.config = ControllerConfig(compliance_ref_link='tool', sensor_ref_link='sensor')
            self.reference_start = torch.tensor([[.1, .5, .2, 0., 0., 0., 1.]], dtype=torch.float64)
            # Rotation +90 deg about x maps DiSECt +y to robot +z.
            self.coupling = SimpleNamespace(base_from_disect=torch.tensor(
                [0., 0., 0., 2**-.5, 0., 0., 2**-.5], dtype=torch.float64))
            self.sim_time = torch.zeros(1, dtype=torch.float64)
        def set_reference_trajectory(self, path):
            self.path = path
        def reference_target(self, t):
            pose = self.reference_start.clone()
            pose[:, :3] += self.path.displacement(t, self.coupling.base_from_disect[3:])
            return pose
    physics = Physics()
    env = SimpleNamespace(worlds=1, physics=physics, metadata={
        'environment': {'agent_control_dt': .05}, 'free_cut_task': {'horizon_steps': 180}})
    policy = FDCCReferencePolicy(env, saw=True)
    frame = {'previous_ee_pose': physics.reference_start.clone(),
             'target_pose': physics.reference_start.clone(),
             'target_speed': torch.tensor([[.03]]), 'observation': torch.zeros(1, 25)}
    frame['target_pose'][0, 2] = .13
    policy.reset(frame)
    first = policy.act(frame)
    assert first.shape == (1, 6) and policy.reference and policy.unchanged()
    cmd = policy.commands[-1]['target_pose_base']
    policy.reset(frame)
    policy.act(frame)
    assert policy.commands[-1]['target_pose_base'] == cmd
    assert len(policy.trajectories) == 2
