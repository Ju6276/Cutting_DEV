import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rigid_inertia import combine_fixed_inertias
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
