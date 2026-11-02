import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rigid_inertia import combine_fixed_inertias
from prepare_robot import adjacent_collision_pairs
from ur5e_model import action_mapping, load_model, targets
from validate_robot import validate


def test_named_torque_mapping():
    model, data = load_model()
    names = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']
    assert (model.nq, model.nv, model.nu) == (6, 6, 6)
    assert [item['joint'] for item in action_mapping(model)] == ['b_bot_' + n + '_joint' for n in names]
    assert list(model.actuator_trntype) == [mujoco.mjtTrn.mjTRN_JOINT] * 6
    for _ in range(100):
        data.ctrl[:] = targets(model, data.time, 0)
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert not np.allclose(targets(model, 0, 0), targets(model, 0, 1))


def test_source_fk_and_mass():
    result = validate('runs/source/episode_000.npz')
    assert result['position_max_error_m'] < 1e-10
    assert result['rotation_matrix_max_abs_error'] < 1e-10
    assert abs(result['source_mass_kg'] - result['imported_mass_kg']) < 1e-10


def test_fixed_body_combination_preserves_kinetic_energy():
    robot = ET.fromstring('''<robot><link name="base"><inertial>
      <mass value="2"/><inertia ixx="1" iyy="2" izz="2" ixy="0" ixz="0" iyz="0"/>
      </inertial></link><link name="child"><inertial><mass value="3"/>
      <inertia ixx="0" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
      </inertial></link><joint name="fixed" type="fixed"><parent link="base"/>
      <child link="child"/><origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>
      </joint></robot>''')
    group = combine_fixed_inertias(robot, 'base')[0]
    velocity, omega = np.array([0.3, -1, 2]), np.array([0.4, 0.5, -0.2])
    offset = np.array([1., 2., 3.])
    child_velocity = velocity + np.cross(omega, offset)
    original = (2 * velocity @ velocity + omega @ np.diag([1, 2, 2]) @ omega
                + 3 * child_velocity @ child_velocity + omega @ np.diag([0.1, 0, 0.1]) @ omega) / 2
    com_velocity = velocity + np.cross(omega, group['com'])
    combined = (group['mass'] * com_velocity @ com_velocity + omega @ np.array(group['inertia']) @ omega) / 2
    np.testing.assert_allclose(combined, original, rtol=1e-14)


def test_adjacent_collisions_follow_fixed_body_groups():
    robot = ET.fromstring('''<robot>
      <link name="base"><collision/></link>
      <link name="base_geometry"><collision/></link>
      <link name="first"><collision/></link>
      <link name="attachment"><collision/></link>
      <link name="second"><collision/></link>
      <joint type="fixed"><parent link="base"/><child link="base_geometry"/></joint>
      <joint type="revolute"><parent link="base_geometry"/><child link="first"/></joint>
      <joint type="fixed"><parent link="first"/><child link="attachment"/></joint>
      <joint type="revolute"><parent link="attachment"/><child link="second"/></joint>
      </robot>''')
    pairs = {frozenset(pair) for pair in adjacent_collision_pairs(robot, 'base')}
    expected = [('base', 'first'), ('base', 'attachment'),
                ('base_geometry', 'first'), ('base_geometry', 'attachment'),
                ('first', 'second'), ('attachment', 'second')]
    assert pairs == {frozenset(pair) for pair in expected}
    assert frozenset(('base', 'second')) not in pairs


def test_source_adjacent_collision_exclusion():
    urdf = Path('runs/model/robot.urdf')
    robot = ET.parse(urdf).getroot()
    spec = mujoco.MjSpec.from_file(str(urdf))
    original = spec.compile()
    for first, second in adjacent_collision_pairs(robot, 'b_bot_base_link'):
        spec.add_exclude(bodyname1=first, bodyname2=second)
    model = spec.compile()
    recorded = np.load('runs/source/episode_000.npz')
    contact_pairs = []
    for candidate in (original, model):
        data = mujoco.MjData(candidate)
        data.qpos[:] = recorded['joint_q'][0]
        mujoco.mj_forward(candidate, data)
        contact_pairs.append({frozenset(candidate.body(int(candidate.geom_bodyid[g])).name
                                       for g in contact.geom)
                              for contact in data.contact})
    adjacent = frozenset(('b_bot_base_link_inertia', 'b_bot_shoulder_link'))
    assert adjacent in contact_pairs[0]
    assert adjacent not in contact_pairs[1]
    np.testing.assert_array_equal(model.body_mass, original.body_mass)
    np.testing.assert_array_equal(model.body_inertia, original.body_inertia)
    np.testing.assert_array_equal(model.geom_pos, original.geom_pos)
