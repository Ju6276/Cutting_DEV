import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from cutting_world_model.sac_gpu_bridge import (
    JsonProcess, SACGPUEnvironment, SACPolicyProcess, TerminalReplayBuffer,
    install_terminal_replay_factory, observation_size, validate_task,
)


ROOT = Path(__file__).resolve().parents[1]


def test_private_protocol_ignores_stdout_and_closes_failed_worker(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text("""
import json, socket, sys
print('native library diagnostic', flush=True)
connection = socket.socket(fileno=int(sys.argv[-1]))
with connection.makefile('rwb') as stream:
    for line in stream:
        message = json.loads(line)
        response = {'ok': message['op'] != 'fail', 'error': 'expected failure', 'value': message.get('value')}
        stream.write((json.dumps(response) + '\\n').encode())
        stream.flush()
""")
    client = JsonProcess([sys.executable, script], timeout=5)
    assert client.request("echo", value=[1, 2, 3])["value"] == [1, 2, 3]
    with pytest.raises(RuntimeError, match="expected failure"):
        client.request("fail")
    assert client.process.poll() is not None
    client.close()


def test_timeout_reaps_worker(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text("import time; time.sleep(30)\n")
    client = JsonProcess([sys.executable, script], timeout=.1)
    with pytest.raises((TimeoutError, OSError)):
        client.request("init")
    assert client.process.poll() is not None


def test_source_actor_parameter_reader_and_defaults():
    spec = importlib.util.spec_from_file_location("sac_gpu_entry", ROOT / "scripts/train_sac_gpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = sys.modules.get("rospy")
    try:
        source = module.install_source_parameter_reader({"ur3e_gym": {"wrench_hist_size": 3, "target_dims": [0, 2, 4]}})
        assert source.get_param("ur3e_gym/wrench_hist_size", 12) == 3
        assert source.get_param("/ur3e_gym/target_dims") == [0, 2, 4]
        assert source.get_param("not_present", 9) == 9
        with pytest.raises(KeyError):
            source.get_param("not_present")
    finally:
        if previous is None:
            sys.modules.pop("rospy", None)
        else:
            sys.modules["rospy"] = previous
    assert module.SOURCE_DEFAULTS["max_steps"] == 100100
    assert module.SOURCE_DEFAULTS["n_warmup"] == 1000
    assert module.SOURCE_DEFAULTS["save_model_interval"] == 5000
    assert module.SOURCE_DEFAULTS["use_prioritized_rb"]
    from types import SimpleNamespace
    parameters = {"ur3e_gym": {
        "target_dims": [0, 2, 4], "n_actions": 6, "jerkiness_as_obs": True,
        "last_action_as_obs": True, "wrench_hist_size": 3, "actor_class": "wave",
        "batch_size": 1024, "lr": .01, "auto_alpha": True, "update_interval": 1,
    }}
    received = module.make_policy(lambda **kwargs: kwargs, parameters, SimpleNamespace(**module.SOURCE_DEFAULTS))
    assert received["state_shape"] == (24,)
    assert received["batch_size"] == 1024 and received["lr"] == .01 and received["auto_alpha"]


def test_metadata_rejects_changed_force_history_order():
    import copy
    environment = {key: 1 for key in (
        "target_dims", "n_actions", "action_type", "ft_hist", "wrench_hist_size",
        "last_action_as_obs", "normalize_velocity", "jerkiness_as_obs",
        "max_distance", "goal_threshold", "workspace_limit", "successes_threshold",
        "cost", "w_dist", "w_force", "w_jerkiness", "use_dynamic_rewards",
        "termination_on_negative_reward", "agent_control_dt", "robot_control_dt",
        "steps_per_episode", "controller_type", "relative_to_ee",
    )}
    environment.update(target_dims=[0, 2, 4], n_actions=6)
    controller = {key: [1] for key in ("max_twist", "max_force_torque", "desired_force_torque")}
    source = {"ur3e_gym": environment, "ur3e_force_control": controller}
    metadata = copy.deepcopy(dict(environment=environment, controller=controller,
                                  action_bounds=[[-1.] * 6, [1.] * 6]))
    validate_task(source, metadata)
    metadata["environment"]["target_dims"] = [4, 2, 0]
    with pytest.raises(ValueError, match="target_dims"):
        validate_task(source, metadata)
    metadata["environment"]["target_dims"] = [0, 2, 4]
    metadata["controller"]["max_force_torque"] = [2]
    with pytest.raises(ValueError, match="max_force_torque"):
        validate_task(source, metadata)


@pytest.mark.parametrize("terminated,truncated,step", [(True, False, 500), (False, True, 500), (False, False, 7)])
def test_source_replay_retains_actual_terminal_at_episode_limit(terminated, truncated, step):
    from types import SimpleNamespace

    class Buffer:
        def add(self, **transition):
            self.last = transition

        def get_all_transitions(self):
            return self.last

    buffer = Buffer()
    environment = SimpleNamespace(last_response=dict(terminated=terminated, truncated=truncated, step_count=step))
    trainer_module = SimpleNamespace(get_replay_buffer=lambda *args, **kwargs: buffer)
    install_terminal_replay_factory(trainer_module)
    replay = trainer_module.get_replay_buffer(None, environment)
    replay.add(obs=np.zeros(24), act=np.zeros(6), next_obs=np.ones(24), rew=1., done=False)
    assert buffer.last["done"] == terminated
    assert replay.get_all_transitions() is buffer.last
    restored = np.array([[True], [False]])
    replay.add(obs=np.zeros((2, 24)), done=restored)
    np.testing.assert_array_equal(buffer.last["done"], restored)


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True)])
def test_terminal_observation_is_preserved_until_explicit_reset(terminated, truncated):
    gym = pytest.importorskip("gym")

    class Client:
        def request(self, operation, **arguments):
            if operation == "reset":
                return dict(observation=[0.] * 24, sim_time=0.)
            return dict(observation=[1.] * 24, sim_time=.05, reward=-100.,
                        terminated=terminated, truncated=truncated, info={"collision": terminated})

    env = SACGPUEnvironment.__new__(SACGPUEnvironment)
    env.client = Client()
    env.action_space = gym.spaces.Box(-1., 1., shape=(6,), dtype=np.float32)
    env.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float32)
    env.needs_reset = True
    env.on_step = None
    env.should_stop = None
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(np.zeros(6))
    assert not env.reset().any()
    observation, reward, done, info = env.step(np.zeros(6))
    assert observation.tolist() == [1.] * 24
    assert done and reward == -100.
    assert info["TimeLimit.truncated"] == truncated
    assert info["terminated"] == terminated
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(np.zeros(6))
    assert not env.reset().any()
    with pytest.raises(ValueError, match="bounds"):
        env.step(np.full(6, 1.1))


@pytest.mark.skipif(not os.environ.get("CUTTING_SAC_MODEL_DIR"), reason="Source TensorFlow checkpoint")
def test_source_checkpoint_batched_deterministic_inference():
    source = Path(os.environ["CUTTING_SAC_ROOT"])
    policy = SACPolicyProcess(source / ".venv/bin/python", source,
                              source / "configs/public_prism_ros.yaml", os.environ["CUTTING_SAC_MODEL_DIR"])
    try:
        observation = np.zeros((3, 24), dtype=np.float32)
        batch = policy.act(observation)
        assert batch.shape == (3, 6)
        np.testing.assert_allclose(batch, np.repeat(policy.act(observation[0])[None], 3, axis=0), rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(batch, policy.act(observation))
        assert policy.unchanged()
    finally:
        policy.close()


@pytest.mark.skipif(not os.environ.get("CUTTING_SAC_GPU_CHECK"), reason="Explicit single-world physical bridge check")
def test_gpu_worker_observation_force_and_reset_alignment():
    root = Path(os.environ["CUTTING_BACKEND_ROOT"])
    client = JsonProcess([sys.executable, ROOT / "scripts/sac_gpu_worker.py", "--backend-root", root])
    try:
        ready = client.request("init")
        initial = client.request("reset")
        metadata = ready["metadata"]
        assert observation_size(metadata["environment"]) == 24
        action = np.array([.1, -.1, .2, -.2, .3, -.3], dtype=np.float32)
        after = client.request("step", action=action.tolist())
        observation = np.asarray(after["observation"])
        assert observation.shape == (24,) and np.isfinite(observation).all()
        np.testing.assert_array_equal(observation[9:15], action)
        wrench = np.asarray(after["wrench_history"])
        normalized = wrench / metadata["controller"]["max_force_torque"]
        force = normalized[:, metadata["environment"]["target_dims"]].T.ravel()
        np.testing.assert_allclose(observation[-9:], force, rtol=1e-6, atol=1e-7)
        assert after["step_count"] == 1
        dt = metadata["environment"]["agent_control_dt"]
        assert 0 <= after["sim_time"] <= dt * (1 + np.finfo(np.float32).eps)
        reset = client.request("reset")
        np.testing.assert_array_equal(reset["observation"], initial["observation"])
        np.testing.assert_array_equal(reset["joint_q"], initial["joint_q"])
    finally:
        client.close()
