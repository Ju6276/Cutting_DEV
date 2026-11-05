"""Process interfaces between SliceIt's Python 3.8 SAC and the CUDA environment."""
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys

import numpy as np


class JsonProcess:
    """A private socket keeps native library output out of the request stream."""

    def __init__(self, command, timeout=600, environment=None, stderr=None):
        self.timeout = timeout
        self.socket, child = socket.socketpair()
        self.socket.settimeout(timeout)
        self.stream = self.socket.makefile("rwb")
        self.process = None
        try:
            self.process = subprocess.Popen(
                list(map(str, command)) + ["--protocol-fd", str(child.fileno())],
                pass_fds=(child.fileno(),), env=environment, stdout=stderr,
                stderr=stderr, start_new_session=True,
            )
        except BaseException:
            self.close()
            raise
        finally:
            child.close()

    def request(self, operation, **values):
        if self.process.poll() is not None:
            raise RuntimeError("SAC process exited with code %s" % self.process.returncode)
        message = dict(op=operation, **values)
        try:
            self.stream.write((json.dumps(message, allow_nan=False) + "\n").encode())
            self.stream.flush()
            response = self.stream.readline()
            if not response:
                raise RuntimeError("SAC process closed during " + operation)
            response = json.loads(response)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "SAC worker failed"))
            return response
        except (OSError, ValueError, RuntimeError):
            # A timed-out stream cannot be safely reused for another request.
            self.close()
            raise

    def close(self):
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=5)
        self.stream.close()
        self.socket.close()


def serve(protocol_fd, handler):
    with socket.socket(fileno=protocol_fd) as connection:
        with connection.makefile("rwb") as stream:
            for line in stream:
                message = {}
                try:
                    message = json.loads(line)
                    response = dict(ok=True, **handler(message))
                except Exception as error:
                    response = dict(ok=False, error=repr(error))
                stream.write((json.dumps(response, allow_nan=False) + "\n").encode())
                stream.flush()
                if message.get("op") == "close" or not response["ok"]:
                    return


def clean_python_environment():
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(key, None)
    return environment


def observation_size(environment):
    dims = len(environment["target_dims"])
    size = 2 * dims
    if environment["jerkiness_as_obs"]:
        size += dims
    if environment["last_action_as_obs"]:
        size += environment["n_actions"]
    return size + dims * environment["wrench_hist_size"]


def validate_task(source_parameters, metadata):
    """Require the source observation, reward, timing, and action definitions."""
    source = source_parameters["ur3e_gym"]
    current = metadata["environment"]
    keys = (
        "target_dims", "n_actions", "action_type", "ft_hist", "wrench_hist_size",
        "last_action_as_obs", "normalize_velocity", "jerkiness_as_obs",
        "max_distance", "goal_threshold", "workspace_limit", "successes_threshold",
        "cost", "w_dist", "w_force", "w_jerkiness", "use_dynamic_rewards",
        "termination_on_negative_reward", "agent_control_dt", "robot_control_dt",
        "steps_per_episode", "controller_type", "relative_to_ee",
    )
    for key in keys:
        if source[key] != current[key]:
            raise ValueError("Source and GPU task differ at " + key)
    default_velocity = [.5, .5, .5, .785398, .785398, .785398]
    if source.get("max_velocity", default_velocity) != current.get("max_velocity", default_velocity):
        raise ValueError("Source and GPU task differ at max_velocity")
    for key in ("max_twist", "max_force_torque", "desired_force_torque"):
        if source_parameters["ur3e_force_control"][key] != metadata["controller"][key]:
            raise ValueError("Source and GPU controller differ at " + key)
    expected = np.tile(np.array([[-1.], [1.]], dtype=np.float32), (1, source["n_actions"]))
    if not np.array_equal(metadata["action_bounds"], expected):
        raise ValueError("Source SAC requires action bounds [-1, 1]")


class TerminalReplayBuffer:
    """Keep physical termination when it coincides with the time limit."""

    def __init__(self, buffer, environment, after_episode=None):
        self.buffer = buffer
        self.environment = environment
        self.after_episode = after_episode

    def add(self, **transition):
        # Source restore supplies a batch; preserve its recorded done flags.
        if np.ndim(transition["done"]) == 0:
            transition = dict(transition, done=bool(self.environment.last_response["terminated"]))
        return self.buffer.add(**transition)

    def on_episode_end(self):
        # The source loop calls this after the terminal transition was stored.
        result = self.buffer.on_episode_end()
        if self.after_episode is not None:
            self.after_episode()
        return result

    def __getattr__(self, name):
        return getattr(self.buffer, name)


def install_terminal_replay_factory(trainer_module, on_create=None, after_episode=None):
    source_factory = trainer_module.get_replay_buffer

    def get_replay_buffer(policy, environment, *args, **kwargs):
        buffer = source_factory(policy, environment, *args, **kwargs)
        replay = TerminalReplayBuffer(buffer, environment, after_episode)
        if on_create is not None:
            on_create(replay)
        return replay

    trainer_module.get_replay_buffer = get_replay_buffer


class SACGPUEnvironment:
    """Gym 0.21 interface; each step returns the terminal frame before reset."""

    def __init__(self, python, backend_root, timeout=600, on_step=None, worker=None,
                 should_stop=None):
        from gym.spaces import Box

        root = Path(__file__).resolve().parents[1]
        script = worker or root / "scripts/sac_gpu_worker.py"
        self.client = JsonProcess(
            [python, script, "--backend-root", backend_root], timeout=timeout,
            environment=clean_python_environment(),
        )
        try:
            ready = self.client.request("init")
            self.metadata = ready["metadata"]
            self.n_actions = self.metadata["environment"]["n_actions"]
            self._max_episode_steps = self.metadata["environment"]["steps_per_episode"]
            low, high = np.asarray(self.metadata["action_bounds"], dtype=np.float32)
            self.action_space = Box(low, high, dtype=np.float32)
            self.observation_space = Box(
                -np.inf, np.inf, shape=(observation_size(self.metadata["environment"]),),
                dtype=np.float32,
            )
            self.on_step = on_step
            self.should_stop = should_stop
            self.sim_time = 0.
            self.last_response = ready
            self.needs_reset = True
            self._observation(ready)
        except BaseException:
            self.close()
            raise

    @property
    def unwrapped(self):
        return self

    def seed(self, seed):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    def _observation(self, response):
        observation = np.asarray(response["observation"], dtype=np.float32)
        if observation.shape != self.observation_space.shape or not np.isfinite(observation).all():
            raise ValueError("Invalid GPU observation")
        self.sim_time = response["sim_time"]
        self.last_response = response
        return observation

    def reset(self):
        response = self.client.request("reset")
        observation = self._observation(response)
        self.needs_reset = False
        return observation

    def step(self, action):
        if self.should_stop is not None and self.should_stop():
            raise InterruptedError("Requested stop before the next physical action")
        if self.needs_reset:
            raise RuntimeError("Reset before stepping a new or completed episode")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("SAC action must be a finite 6D vector")
        if not self.action_space.contains(action):
            raise ValueError("SAC action exceeds the source [-1, 1] bounds")
        response = self.client.request("step", action=action.tolist())
        observation = self._observation(response)
        reward = float(response["reward"])
        if not np.isfinite(reward):
            raise FloatingPointError("Non-finite physical reward")
        terminated, truncated = response["terminated"], response["truncated"]
        done = bool(terminated or truncated)
        info = dict(response["info"], terminated=terminated,
                    **{"TimeLimit.truncated": truncated})
        self.needs_reset = done
        if self.on_step is not None:
            self.on_step(observation, action, reward, done, info)
        return observation, reward, done, info

    def close(self):
        if hasattr(self, "client"):
            self.client.close()


class SACPolicyProcess:
    """Load a source TensorFlow checkpoint for a modern Python evaluator."""

    def __init__(self, python, sliceit_root, config, model_dir, timeout=600):
        root = Path(__file__).resolve().parents[1]
        environment = clean_python_environment()
        # Inference is small; this process never reserves the physics GPU.
        environment["CUDA_VISIBLE_DEVICES"] = ""
        self.client = JsonProcess(
            [python, root / "scripts/train_sac_gpu.py", "--sliceit-root", sliceit_root,
             "--config", config, "--model-dir", model_dir, "--actor-worker"],
            timeout=timeout, environment=environment,
        )
        try:
            self.metadata = self.client.request("init")
            self.observation_shape = tuple(self.metadata["observation_shape"])
        except BaseException:
            self.close()
            raise

    def act(self, observation, deterministic=True):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape[-1:] != self.observation_shape or observation.ndim not in (1, 2):
            raise ValueError("Unexpected SAC observation shape")
        if not np.isfinite(observation).all():
            raise ValueError("Non-finite SAC observation")
        response = self.client.request("act", observation=observation.tolist(),
                                       deterministic=bool(deterministic))
        action = np.asarray(response["action"], dtype=np.float32)
        if action.shape != observation.shape[:-1] + (6,) or not np.isfinite(action).all():
            raise ValueError("Invalid SAC actor response")
        return action

    def close(self):
        self.client.close()

    def unchanged(self):
        return bool(self.client.request("unchanged")["unchanged"])

    def validate_metadata(self, metadata):
        validate_task(self.metadata["source_parameters"], metadata)
