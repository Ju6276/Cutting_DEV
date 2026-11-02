"""Read simulator snapshots before SliceIt's terminal controller shutdown."""
import numpy as np


class ObservationRecorder:
    def __init__(self, environment, snapshot):
        self.env = environment
        self.snapshot = snapshot
        self.original_obs = environment._get_obs
        self.original_wrench = environment.ur3e_arm.get_ee_wrench_hist
        self.in_observation = False
        self.wrench = None
        self.frame = None
        environment._get_obs = self.observe
        environment.ur3e_arm.get_ee_wrench_hist = self.read_wrench

    def read_wrench(self, *args, **kwargs):
        value = self.original_wrench(*args, **kwargs)
        if self.in_observation:
            self.wrench = np.array(value, copy=True)
        return value

    def observe(self, *args, **kwargs):
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
        return observation

    def read(self, observation):
        if self.frame is None or not np.array_equal(self.frame["observation"], observation):
            raise RuntimeError("Recorded state does not match the returned observation")
        frame = self.frame.copy()
        frame["success_counter"] = np.asarray([self.env.success_counter], dtype=np.float32)
        frame["step_count"] = np.asarray([self.env.step_count], dtype=np.float32)
        return frame
