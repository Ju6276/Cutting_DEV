import os
import sys
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('CUTTING_BACKEND_ROOT'), reason='CUDA backend assets')
def test_gpu_task_reset_preserves_other_world_and_observation_state():
    root = os.environ['CUTTING_BACKEND_ROOT']
    sys.path.insert(0, root)
    from cutting_world_model.gpu_env import GPUEnvironment
    from cutting_world_model.state_spec import CuttingStateSpec
    env = GPUEnvironment(root, 2)
    initial = {k: v.clone() for k, v in env.current.items()}
    frame, reward, term, trunc, info = env.step(torch.tensor([[0.] * 6, [.2] * 6], device='cuda'))
    assert reward.shape == term.shape == trunc.shape == (2,)
    assert torch.isfinite(reward).all()
    before = {k: v.clone() for k, v in frame.items()}
    controller_before = env.physics.controller.velocity.clone()
    frame = env.reset(torch.tensor([True, False], device='cuda'))
    for k in frame:
        torch.testing.assert_close(frame[k][1], before[k][1], atol=0, rtol=0)
        torch.testing.assert_close(frame[k][0], initial[k][0], atol=0, rtol=0)
    torch.testing.assert_close(env.physics.controller.velocity[1], controller_before[1], atol=0, rtol=0)
    one = {k: v[0].cpu().numpy() for k, v in frame.items()}
    spec = CuttingStateSpec(CuttingStateSpec.metadata_from_frame(env.metadata, one)).cuda()
    state = spec.assemble(frame).float()
    torch.testing.assert_close(spec.field(state, 'observation'), frame['observation'].float())
    continued, *_ = env.step(torch.zeros(2, 6, device='cuda'))
    assert torch.isfinite(spec.assemble(continued)).all()

    before_stop = {k: v.clone() for k, v in continued.items()}
    env.physics.feedback['observation_wrench'][0, 0] = env.physics.force_limits[0, 0] + 1
    stopped, _, term, _, info = env.step(torch.zeros(2, 6, device='cuda'))
    assert term[0] and info['collision'][0] and not info['collision'][1]
    torch.testing.assert_close(stopped['particle_q'][0], before_stop['particle_q'][0], atol=0, rtol=0)
    assert not torch.equal(stopped['particle_q'][1], before_stop['particle_q'][1])

    live = env.physics.frame()
    for key in ('particle_q', 'particle_qd', 'cut_spring_ke', 'joint_q', 'joint_qd',
                'controller_velocity', 'filter_history', 'filter_zi', 'wrench_queue',
                'coupling_previous_pose', 'robot_warmstart', 'robot_time', 'sim_time'):
        torch.testing.assert_close(live[key][0], before_stop[key][0], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('CUTTING_BACKEND_ROOT'), reason='CUDA backend assets')
def test_augmented_state_observes_memory_and_restores_contact_transition():
    root = os.environ['CUTTING_BACKEND_ROOT']
    sys.path.insert(0, root)
    from cutting_world_model.gpu_env import GPUEnvironment
    from cutting_world_model.state_spec import CuttingStateSpec
    env = GPUEnvironment(root, 2)
    one = {k: v[0].cpu().numpy() for k, v in env.current.items()}
    spec = CuttingStateSpec(CuttingStateSpec.metadata_from_frame(env.metadata, one)).cuda()
    before = spec.assemble(env.current).clone()
    env.physics.controller.velocity[0, 0] += .001
    changed = dict(env.current, **env.physics.frame())
    assert not torch.equal(before[0], spec.assemble(changed)[0])
    env.reset()
    for _ in range(12):
        frame, _, term, trunc, _ = env.step(torch.zeros(2, 6, device='cuda'), reference=True)
        assert not (term | trunc).any()
    saved = {k: v.clone() for k, v in frame.items()}
    action = torch.zeros(2, 6, device='cuda')
    expected, reward, term, trunc, _ = env.step(action)
    env.restore_frame(saved)
    restored = env.physics.frame()
    for key in spec.physical_fields + spec.recurrent_constants:
        torch.testing.assert_close(restored[key], saved[key], atol=0, rtol=0)
    actual, reward2, term2, trunc2, _ = env.step(action)
    # Shared-particle FP32 atomic force sums vary even with a full scratch restore.
    assert torch.isfinite(actual['particle_q']).all()
    assert torch.isfinite(actual['particle_qd']).all()
    torch.testing.assert_close(actual['joint_q'], expected['joint_q'], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reward2, reward, atol=1e-4, rtol=1e-4)
    assert torch.equal(term, term2) and torch.equal(trunc, trunc2)
    state = spec.assemble(saved).float()
    fields = spec.integrate_physical(state, torch.zeros(2, spec.delta_dim, device='cuda'))
    new_state, _, _ = spec.derive(state, fields, action, saved['target_pose'].float())
    assert torch.isfinite(new_state).all()
    for key in spec.recurrent_constants:
        torch.testing.assert_close(spec.field(new_state, key), spec.field(state, key))

    env.restore_frame(saved)
    args = env.physics.mapper(action.double(), env.physics.target, env.physics.frames[env.physics.chain.tip])
    active = torch.ones(2, dtype=torch.bool, device='cuda')
    command1 = env.physics.controller_command(args, active).clone()
    env.restore_frame(saved)
    args = env.physics.mapper(action.double(), env.physics.target, env.physics.frames[env.physics.chain.tip])
    command2 = env.physics.controller_command(args, active).clone()
    torch.testing.assert_close(command1, command2, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get('CUTTING_BACKEND_ROOT'), reason='CUDA backend assets')
def test_net_sensor_frame_restore_metadata_and_diagnostic_sampling(monkeypatch):
    root = os.environ['CUTTING_BACKEND_ROOT']
    sys.path.insert(0, root)
    from cutting_world_model.gpu_env import GPUEnvironment
    env = GPUEnvironment(root, 2, force_model='net', friction_velocity=.01)
    assert env.metadata['material_feedback'] == 'net_contact_sensor_v1'
    assert env.metadata['contact_discretization_revision'] == 'edge_coordinate_consistent_v2'
    assert env.metadata['material_contact']['friction_velocity_override_mps'] == .01
    assert env.metadata['material_contact']['effective_friction_velocity_mps'] == pytest.approx([.01])
    initial = {k: v.clone() for k, v in env.current.items()}
    samples = []
    def sample(p, active):
        torch.testing.assert_close(p.feedback['sensor_wrench'], p.feedback['observation_wrench'])
        samples.append((p.sim_time.clone(), active.clone()))
    first, *_ = env.step(torch.zeros(2, 6, device='cuda'), sample_callback=sample)
    assert len(samples) == 25
    torch.testing.assert_close(samples[-1][0], torch.full_like(samples[-1][0], .05))
    env.restore_frame(initial)
    torch.testing.assert_close(env.physics.feedback['observation_wrench'], initial['wrench_queue'][:, -1])
    second, *_ = env.step(torch.zeros(2, 6, device='cuda'))
    torch.testing.assert_close(first['joint_q'], second['joint_q'], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(first['wrench_queue'], second['wrench_queue'], atol=1e-3, rtol=1e-3)

    env.restore_frame(initial)
    feedback = env.physics._feedback
    calls = 0
    def last_sample_overload(raw):
        nonlocal calls
        result = feedback(raw)
        calls += 1
        if calls == 25:
            # Last coupling sample: pure couple must trip torque, never create force.
            result['sensor_wrench'][0] = 0
            result['sensor_wrench'][0, 4] = env.physics.force_limits[0, 4] + 1
            result['observation_wrench'][0] = result['sensor_wrench'][0]
        return result
    monkeypatch.setattr(env.physics, '_feedback', last_sample_overload)
    frame, _, terminated, _, info = env.step(torch.zeros(2, 6, device='cuda'))
    assert info['collision'][0] and terminated[0]
    assert frame['peak_wrench'][0, 4] > env.physics.force_limits[0, 4]
