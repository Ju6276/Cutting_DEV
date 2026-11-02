"""Read simulator snapshots before SliceIt's terminal controller shutdown."""
import numpy as np


class ObservationRecorder:
    def __init__(self, environment, snapshot):
        self.env = environment
        self.snapshot = snapshot
        self.joint_sample = None
        self.wrench_sample = None
        self.original_obs = environment._get_obs
        self.original_wrench = environment.ur3e_arm.get_ee_wrench_hist
        self.in_observation = False
        self.wrench = None
        self.frame = None
        arm = environment.ur3e_arm
        if hasattr(arm, 'get_wrench_state'):
            self.original_wrench_state = arm.get_wrench_state
            arm.get_wrench_state = self.read_wrench_state
        if hasattr(arm, 'joint_traj_controller'):
            joints = arm.joint_traj_controller
            self.original_joint_state = joints.get_joint_state
            joints.get_joint_state = self.read_joint_state
        environment._get_obs = self.observe
        environment.ur3e_arm.get_ee_wrench_hist = self.read_wrench

    def read_joint_state(self):
        value = self.original_joint_state()
        if self.in_observation and self.joint_sample is None:
            self.joint_sample = {key: np.array(item, copy=True) for key, item in value.items()}
        return value

    def read_wrench_state(self):
        value = self.original_wrench_state()
        if self.in_observation:
            self.wrench_sample = {key: np.array(item, copy=True) for key, item in value.items()}
        return value

    def read_wrench(self, *args, **kwargs):
        value = self.original_wrench(*args, **kwargs)
        if self.in_observation:
            self.wrench = np.array(value, copy=True)
        return value

    def observe(self, *args, **kwargs):
        self.joint_sample = None
        self.wrench_sample = None
        self.in_observation = True
        self.wrench = None
        try:
            observation = self.original_obs(*args, **kwargs)
        finally:
            self.in_observation = False
        if self.wrench is None:
            raise RuntimeError("The selected SliceIt observation did not read force history")
        self.frame = self.snapshot(observation)
        self.frame["wrench_history"] = self.wrench.copy()
        if self.joint_sample is not None:
            self.frame['joint_q'] = self.joint_sample['q']
            self.frame['joint_qd'] = self.joint_sample['qd']
            self.frame['joint_ros_time'] = self.joint_sample['stamp']
            self.frame['ee_pose'] = np.array(self.env.previous_ee_pose, copy=True)
        if self.wrench_sample is not None:
            sample = self.wrench_sample
            count = len(sample['queue'])
            queue = np.zeros((int(sample['capacity']), 6))
            queue[:count] = sample['queue']
            self.frame['wrench_queue'] = queue
            self.frame['wrench_queue_count'] = np.array([count])
            self.frame['current_ft_value'] = sample['current']
            self.frame['wrench_ros_time'] = sample['stamp']
        return observation

    def reset(self, reset_environment):
        self.frame = None
        observation = reset_environment()
        self.read(observation)
        return observation

    def read(self, observation):
        if self.frame is None or not np.array_equal(self.frame["observation"], observation):
            raise RuntimeError("Recorded state does not match the returned observation")
        frame = self.frame.copy()
        frame["success_counter"] = np.asarray([self.env.success_counter], dtype=np.float32)
        frame["step_count"] = np.asarray([self.env.step_count], dtype=np.float32)
        return frame
