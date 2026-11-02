import numpy as np
import torch

from cutting_world_model.kinematics import SerialKinematics, pose_difference, quat_exp, observation_velocity
from cutting_world_model.state_spec import CuttingStateSpec
from cutting_world_model.world_model import CuttingWorldModel
from test_task_heads import heads


def fixture():
    xml = '<robot name="test">'
    names = ['j%d' % i for i in range(6)]
    for i, name in enumerate(names):
        xml += ('<joint name="%s" type="revolute"><parent link="l%d"/><child link="l%d"/>'
                '<origin xyz="0.01 0 0"/><axis xyz="0 0 1"/></joint>') % (name, i, i + 1)
    xml += '</robot>'
    h = heads()
    env = dict(h.env, agent_control_dt=.05, robot_control_dt=.002)
    ctl = dict(h.controller, max_twist=[1., 0.])
    frame = {
        'particle_q': np.zeros((2, 3)), 'particle_qd': np.zeros((2, 3)),
        'cut_spring_ke': np.ones(2) * 100., 'spring_reference_ke': np.ones(2) * 100.,
        'joint_q': np.zeros(6), 'joint_qd': np.zeros(6),
        'knife_pose': np.array([0., 0., 0., 0., 0., 0., 1.]), 'knife_velocity': np.zeros(6),
        'wrench_history': np.zeros((3, 6)), 'peak_wrench': np.zeros(6),
        'motion_command': np.zeros(6), 'previous_ee_pose': np.array([.06, 0., 0., 0., 0., 0., 1.]),
        'previous_velocity': np.zeros(6), 'previous_acceleration': np.zeros(6),
        'observation': np.zeros(24), 'success_counter': np.array([0.]),
    }
    meta = CuttingStateSpec.metadata_from_frame({
        'robot_xml': xml, 'robot_base': 'l0', 'robot_tip': 'l6', 'joint_names': names,
        'environment': env, 'controller': ctl, 'action_bounds': [[-1.] * 6, [1.] * 6],
    }, frame)
    return meta, frame


def test_fk_known_chain_and_gradient():
    xml = ('<robot><joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>'
           '<axis xyz="0 0 1"/></joint><joint name="tip" type="fixed"><parent link="arm"/>'
           '<child link="end"/><origin xyz="1 0 0"/></joint></robot>')
    fk = SerialKinematics(xml, 'base', 'end', ['hinge']).double()
    q = torch.tensor([[np.pi / 2]], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(fk(q)[..., :3], torch.tensor([[0., 1., 0.]], dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.autograd.gradcheck(fk, (q,))


def test_state_constant_velocity_prior_and_no_damage_projection():
    meta, frame = fixture()
    spec = CuttingStateSpec(meta).double()
    frame['particle_qd'][:] = .2
    state = spec.pack(frame).double()[None].expand(3, -1)
    delta = torch.zeros((3, spec.delta_dim), dtype=torch.float64)
    fields = spec.integrate_physical(state, delta)
    torch.testing.assert_close(fields['particle_q'], torch.full((3, 2, 3), .01, dtype=torch.float64))
    torch.testing.assert_close(fields['cut_spring_ke'], torch.full((3, 2), 100., dtype=torch.float64))
    goal = torch.tensor([[.06, 0., -.05, 0., 0., 0., 1.]], dtype=torch.float64).expand(3, -1)
    action = torch.ones(3, 6, dtype=torch.float64) * .1
    result, reward, done = spec.derive(state, fields, action, goal)
    assert result.shape == state.shape and reward.shape == (3,) and done.shape == (3,)
    assert spec.field(result, 'success_counter').shape == (3,)
    torch.testing.assert_close(spec.field(result, 'motion_command')[:, 0], torch.full((3,), .0002, dtype=torch.float64))


def test_zero_rotation_derivatives_and_source_velocity_sign():
    v = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(quat_exp, (v,))
    old = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]], dtype=torch.float64)
    new = torch.cat((old[..., :3], quat_exp(torch.tensor([[0., 0., .1]], dtype=torch.float64))), -1)
    velocity = observation_velocity(new, old, .05)
    torch.testing.assert_close(velocity[0, 5], torch.tensor(-2 * np.sin(.05) / .05, dtype=torch.float64))


def test_world_model_action_gradient_and_batch_independence():
    torch.manual_seed(2)
    meta, frame = fixture()
    model = CuttingWorldModel(meta, hidden=(16, 16)).double()
    torch.nn.init.normal_(model.dynamics.mlp.net[-1].weight, std=1e-5)
    state = model.spec.pack(frame).double()[None].repeat(2, 1)
    goal = torch.tensor([[.06, 0., -.05, 0., 0., 0., 1.]], dtype=torch.float64).repeat(2, 1)
    action = torch.tensor([[.1] * 6, [.2] * 6], dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(lambda a: model.step(state, a, goal)[0], (action,), atol=2e-5, rtol=2e-3)
    batched = model.step(state, action, goal)[0]
    single = model.step(state[:1], action[:1], goal[:1])[0]
    torch.testing.assert_close(batched[:1], single)


def test_fk_matches_kdl_quaternion_sign_for_large_rotations():
    xml = ('<robot><joint name="hinge" type="revolute"><parent link="base"/>'
           '<child link="tip"/><axis xyz="0 0 1"/></joint></robot>')
    fk = SerialKinematics(xml, 'base', 'tip', ['hinge']).double()
    angle = torch.tensor([[2 * np.pi + .2], [-np.pi]], dtype=torch.float64)
    pose = fk(angle)
    assert pose[0, 6] > 0
    assert pose[1, 5] > 0
    torch.testing.assert_close(pose[0, 3:], quat_exp(torch.tensor([[0., 0., .2]], dtype=torch.float64))[0])
