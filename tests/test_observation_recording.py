from types import SimpleNamespace

import numpy as np

from cutting_world_model.observation_recording import ObservationRecorder


def test_capture_preserves_force_sample_used_by_observation():
    sensor = [np.ones((3, 6))]
    arm = SimpleNamespace(get_ee_wrench_hist=lambda count: sensor[0])
    env = SimpleNamespace(ur3e_arm=arm, success_counter=0, step_count=0)
    env._get_obs = lambda: env.ur3e_arm.get_ee_wrench_hist(3).flatten().copy()
    recorder = ObservationRecorder(env, lambda obs: {"observation": obs.copy()})
    obs = env._get_obs()
    sensor[0][:] = 99
    env.success_counter = 1
    env.step_count = 1
    frame = recorder.read(obs)
    np.testing.assert_array_equal(frame["wrench_history"], np.ones((3, 6)))
    np.testing.assert_array_equal(frame["observation"], np.ones(18))
    assert frame["success_counter"][0] == 1


def test_reset_captures_state_and_sensor_at_observation_time():
    sensor = [np.ones((3, 6))]
    env = SimpleNamespace(ur3e_arm=SimpleNamespace(get_ee_wrench_hist=lambda count: sensor[0]))
    env._get_obs = lambda: env.ur3e_arm.get_ee_wrench_hist(3).flatten().copy()
    def reset():
        env.success_counter = 0
        env.step_count = 0
        env.physical_state = 9
        sensor[0][:] = 2
        observation = env._get_obs()
        env.physical_state = 10
        sensor[0][:] = 3
        return observation
    recorder = ObservationRecorder(env, lambda obs: {"observation": obs.copy(), "physical_state": env.physical_state})
    observation = recorder.reset(reset)
    frame = recorder.read(observation)
    assert frame["physical_state"] == 9
    np.testing.assert_array_equal(frame["wrench_history"], np.full((3, 6), 2))
    np.testing.assert_array_equal(frame["observation"], np.full(18, 2))


def test_reset_cannot_reuse_previous_frame():
    import pytest
    env = SimpleNamespace(ur3e_arm=SimpleNamespace(get_ee_wrench_hist=lambda n: np.ones((3, 6))))
    env._get_obs = lambda: env.ur3e_arm.get_ee_wrench_hist(3).flatten().copy()
    recorder = ObservationRecorder(env, lambda obs: {"observation": obs.copy()})
    old = env._get_obs()
    with pytest.raises(RuntimeError, match="Recorded state"):
        recorder.reset(lambda: old)
