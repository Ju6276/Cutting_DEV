import mujoco
import numpy as np

from cutting_gpu_backend.robot import configure_position_hardware


def pendulum(effort):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="sphere" size="0.1" pos="1 0 0" mass="1"/>
      </body></worldbody><actuator>
      <motor joint="hinge" ctrllimited="true" ctrlrange="-%s %s"/>
      </actuator></mujoco>''' % (effort, effort))
    configure_position_hardware(model)
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    return model, data


def test_zero_speed_motor_holds_load_within_effort_limit():
    model, data = pendulum(20)
    np.testing.assert_array_equal(model.dof_frictionloss, [20])
    assert abs(data.qpos[0]) < 1e-5
    assert 9.7 < abs(data.qfrc_constraint[0]) < 9.9


def test_zero_speed_motor_yields_above_effort_limit():
    model, data = pendulum(1)
    assert abs(data.qpos[0]) > .01
    assert abs(data.qfrc_constraint[0]) <= model.dof_frictionloss[0] + 1e-8
