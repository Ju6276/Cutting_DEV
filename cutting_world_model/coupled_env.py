"""SliceIt environment adapter and read-only transition instrumentation."""
import json
from pathlib import Path

import numpy as np

from .observation_recording import ObservationRecorder
from .native_state import NativeStateRecorder, NATIVE_STATE_LAYOUT
from .feedback_wiring import verify_feedback_graph
from .task_heads import TASK_INTERFACE_REVISION


class ControlWrenchRecorder:
    def __init__(self, base):
        self.peak = np.zeros(6)
        original_act = base._set_action
        original_wrench = base.ur3e_arm.get_ee_wrench
        self.active = False

        def wrench(*args, **kwargs):
            value = original_wrench(*args, **kwargs)
            if self.active and kwargs.get('hand_frame_control', False):
                self.peak = np.maximum(self.peak, np.abs(value))
            return value

        def act(action):
            self.peak = np.zeros(6)
            self.active = True
            try:
                return original_act(action)
            finally:
                self.active = False

        base.ur3e_arm.get_ee_wrench = wrench
        base._set_action = act


def capture(env, observation, service, peak, native):
    boundary = native.clock()
    response = service()
    if not response.success:
        raise RuntimeError(response.message)
    path = Path(json.loads(response.message)['path'])
    with np.load(path) as data:
        frame = {name: data[name].copy() for name in data.files}
    path.unlink()
    base = env.unwrapped
    frame.update({
        'observation': np.asarray(observation),
        'peak_wrench': peak.peak.copy(),
        'motion_command': np.asarray(base.controller.added_motion_command).copy(),
        'previous_ee_pose': np.asarray(base.previous_ee_pose).copy(),
        'previous_velocity': np.asarray(base.previous_velocity).copy(),
        'previous_acceleration': np.asarray(base.previous_acceleration).copy(),
        'target_pose': np.asarray(base.current_target_pose),
    })
    frame.update(native.read(boundary))
    previous_pose = base.ur3e_arm.previous_pose
    frame['coupling_initialized'] = np.array([previous_pose is not None], dtype=np.float64)
    frame['coupling_previous_pose'] = (np.asarray(previous_pose).copy() if previous_pose is not None
                                       else np.array([0., 0., 0., 0., 0., 0., 1.]))
    if any(not np.isfinite(value).all() for value in frame.values()):
        raise FloatingPointError('Non-finite coupled state')
    return frame


class CoupledEnvironment:
    def __init__(self, config, seed=123456):
        import os
        import rospy
        from std_srvs.srv import Trigger
        from ur3e_openai.common import load_environment, clear_gym_params, load_ros_params
        if os.environ.get('ROS_MASTER_URI') == 'http://127.0.0.1:11321':
            raise RuntimeError('11321 belongs to the SAC baseline')
        clear_gym_params('ur3e_gym')
        clear_gym_params('ur3e_force_control')
        load_ros_params(rospackage_name='ur3e_rl', rel_path_from_package_to_file='config',
                        yaml_file_name=str(Path(config).resolve()))
        self.environment = rospy.get_param('ur3e_gym')
        self.controller = rospy.get_param('ur3e_force_control')
        self.env = load_environment(self.environment['env_id'],
                                   max_episode_steps=self.environment['steps_per_episode'])
        try:
            step_control_active = bool(self.env.unwrapped.use_step_control)
            if (step_control_active != bool(self.environment.get('use_step_control', False))
                    or step_control_active != bool(self.env.unwrapped.ur3e_arm.use_step_control)):
                raise RuntimeError('SliceIt runtime step-control mode differs from its configuration')
            self.env.max_steps = 100100
            self.env.seed(seed)
            self.env.action_space.seed(seed)
            self.env.observation_space.seed(seed)
            self.service = rospy.ServiceProxy('/world_model/snapshot', Trigger)
            self.service.wait_for_service(timeout=60)
            self.feedback_graph = verify_feedback_graph(rospy.get_master().getSystemState()[2],
                                                        self.environment['namespace'])
            if not self.environment.get('use_step_control'):
                raise ValueError('Native state snapshots require SliceIt stepped Gazebo control')
            self.native = NativeStateRecorder(self.env.unwrapped.ur3e_arm, self.environment['namespace'])
            self.peak = ControlWrenchRecorder(self.env.unwrapped)
            self.recorder = ObservationRecorder(self.env.unwrapped,
                lambda obs: capture(self.env, obs, self.service, self.peak, self.native))
            arm = self.env.unwrapped.ur3e_arm
            self.metadata = {
                'task_interface_revision': TASK_INTERFACE_REVISION,
                'state_layout': NATIVE_STATE_LAYOUT,
                'native_state_services': self.native.topics,
                'native_sampling': 'paused_after_step_control_before_terminal_shutdown',
                'reset_observation': 'post_reset',
                'step_control_active': step_control_active,
                'step_interface_revision': 'synchronous_paused_joint_snapshot_v1',
                'environment': self.environment, 'controller': self.controller,
                'robot_xml': rospy.get_param('/robot_description'),
                'robot_base': arm.base_link, 'robot_tip': arm.ee_link,
                'joint_names': list(arm.joint_names),
                'action_bounds': [self.env.action_space.low.tolist(), self.env.action_space.high.tolist()],
                'material_feedback': 'sensor_relay_filter_controller',
            }
        except BaseException:
            self.close()
            raise

    def reset(self):
        self.peak.peak[:] = 0
        obs = self.recorder.reset(self.env.reset)
        return self.recorder.read(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = bool(info.get('TimeLimit.truncated', False))
        return self.recorder.read(obs), float(reward), bool(done and not truncated), truncated, info

    def close(self):
        # The source shutdown hook switches controllers on the next Gazebo tick.
        if self.env.unwrapped.use_step_control:
            self.env.unwrapped.robot_connection.unpause()
        self.env.close()
