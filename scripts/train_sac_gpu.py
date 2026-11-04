"""Run SliceIt's source SAC trainer against the ROS-free, single-world backend."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cutting_world_model.sac_gpu_bridge import (
    SACGPUEnvironment, install_terminal_replay_factory, observation_size, serve, validate_task,
)
from cutting_world_model.provenance import git_revision


SOURCE_DEFAULTS = dict(
    batch_size=256, n_warmup=1000, max_steps=100100, save_model_interval=5000,
    test_interval=1e10, test_episodes=1, normalize_obs=False, auto_alpha=False,
    use_prioritized_rb=True, lr=1e-3, update_interval=1,
)


def source_parameters(path):
    import yaml
    with Path(path).open() as stream:
        parameters = yaml.safe_load(stream)
    if parameters["ur3e_gym"]["use_dynamic_rewards"]:
        raise ValueError("The fixed GPU task uses static SliceIt reward weights")
    return parameters


def install_source_parameter_reader(parameters):
    """The source actor reads two ROS parameters; the trainer also reads time."""
    module = types.ModuleType("rospy")
    missing = object()

    def get_param(name, default=missing):
        value = parameters
        for key in name.strip("/").split("/"):
            if not isinstance(value, dict) or key not in value:
                if default is missing:
                    raise KeyError(name)
                return default
            value = value[key]
        return value

    module.get_param = get_param
    module.get_time = time.monotonic
    sys.modules["rospy"] = module
    return module


def load_source(sliceit_root, parameters):
    source = Path(sliceit_root) / "underlay_ws/src/ur_python_utilities/dependencies/tf2rl"
    pins = json.loads((ROOT / "integrations/sliceit/patches/sources.json").read_text())
    pin = next(item for item in pins if item["path"].endswith("/tf2rl"))
    revision = git_revision(source)
    if revision["commit"] != pin["source_commit"] or revision["dirty"]:
        raise ValueError("tf2rl checkout does not match the recorded SliceIt source")
    sys.path.insert(0, str(source))
    os.environ.setdefault("MPLBACKEND", "Agg")
    ros_parameters = install_source_parameter_reader(parameters)
    import tensorflow as tf
    # The old TensorFlow wheel runs the small learner on CPU, beside CUDA physics.
    tf.config.set_visible_devices([], "GPU")
    from tf2rl.algos.sac import SAC
    from tf2rl.experiments.cb_trainer import Trainer
    return tf, SAC, Trainer, ros_parameters, revision


def make_policy(SAC, parameters, args):
    environment = parameters["ur3e_gym"]
    return SAC(
        state_shape=(observation_size(environment),), action_dim=environment["n_actions"],
        max_action=1., actor_class=environment.get("actor_class", "default"),
        batch_size=environment.get("batch_size", args.batch_size),
        n_warmup=args.n_warmup, auto_alpha=environment.get("auto_alpha", args.auto_alpha),
        lr=environment.get("lr", args.lr),
        update_interval=environment.get("update_interval", args.update_interval),
    )


def restore_policy(tf, policy, model_dir):
    checkpoint = tf.train.latest_checkpoint(str(model_dir))
    if checkpoint is None:
        raise FileNotFoundError("No TensorFlow checkpoint in " + str(model_dir))
    restored = tf.train.Checkpoint(policy=policy).restore(checkpoint)
    restored.assert_existing_objects_matched()
    # Training checkpoints may also contain lazily created optimizer slots.
    restored.expect_partial()
    return checkpoint


def actor_worker(args, tf, SAC, parameters, revision):
    policy = make_policy(SAC, parameters, args)
    checkpoint = restore_policy(tf, policy, args.model_dir)
    weights = [value.numpy().copy() for value in policy.weights]
    optimizers = [getattr(policy, name + "_optimizer") for name in ("actor", "qf1", "qf2")]
    if policy.auto_alpha:
        optimizers.append(policy.alpha_optimizer)
    counters = [int(value.iterations.numpy()) for value in optimizers]

    def handle(message):
        if message["op"] == "init":
            return dict(
                checkpoint=checkpoint, source=revision,
                observation_shape=[observation_size(parameters["ur3e_gym"])], action_shape=[6],
                source_parameters=parameters,
                optimizer_iterations=dict(zip(("actor", "qf1", "qf2", "alpha"), counters)),
            )
        if message["op"] == "act":
            observation = np.asarray(message["observation"], dtype=np.float32)
            action = policy.get_action(observation, test=message["deterministic"])
            return dict(action=np.asarray(action).tolist())
        if message["op"] == "unchanged":
            unchanged = len(weights) == len(policy.weights)
            unchanged = unchanged and all(np.array_equal(old, new.numpy())
                                          for old, new in zip(weights, policy.weights))
            unchanged = unchanged and counters == [int(value.iterations.numpy()) for value in optimizers]
            return dict(unchanged=unchanged)
        if message["op"] == "close":
            return {}
        raise ValueError("Unknown actor operation: " + message["op"])

    serve(args.protocol_fd, handle)


def run(args, tf, SAC, Trainer, ros_parameters, parameters, revision):
    from tf2rl.misc.get_replay_buffer import save_replay_buffer
    from tf2rl.experiments import cb_trainer
    install_terminal_replay_factory(cb_trainer)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.model_dir and not args.evaluate:
        raise ValueError("Start unified-environment training fresh; use --evaluate for old checkpoints")
    config = dict(
        code=git_revision(ROOT), tf2rl=revision, arguments=vars(args),
        source_parameters=parameters, physical_backend="mujoco_warp_disect", worlds=1,
    )
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    started = time.monotonic()
    state = dict(state="running", steps=0, updates=0, episodes=0)
    episode = dict(steps=0, reward=0.)
    episode_results = []
    resources = {}
    tracker = None
    events = (output / "metrics.jsonl").open("a", buffering=1)
    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def emit(values):
        values = dict(values, iteration=state["steps"], wall_seconds=time.monotonic() - started)
        events.write(json.dumps(values, allow_nan=False) + "\n")
        (output / "progress.json").write_text(json.dumps(dict(state, wall_seconds=values["wall_seconds"]), indent=2) + "\n")
        if tracker is not None:
            tracker.log(values)

    def should_stop():
        return stopped or bool(args.wall_seconds and time.monotonic() - started >= args.wall_seconds)

    def on_step(observation, action, reward, done, info):
        state["steps"] += 1
        episode["steps"] += 1
        episode["reward"] += reward
        emit(dict(kind="transition", reward=reward, **{
            "reward/step": reward, "physics/peak_force": info["peak_force"],
            "physics/released_springs": info["released_springs"],
        }))
        if done:
            result = dict(episode, success=info["success"], collision=info["collision"],
                          outside=info["outside"], truncated=info["TimeLimit.truncated"])
            episode_results.append(result)
            state["episodes"] += 1
            emit(dict(kind="episode", **result, **{"reward/episode": result["reward"]}))
            episode.update(steps=0, reward=0.)

    class TrackedTrainer(Trainer):
        def update_policy(self, replay_buffer, *values, **kwargs):
            resources["replay"] = replay_buffer
            result = super().update_policy(replay_buffer, *values, **kwargs)
            if any(not np.isfinite(value.numpy()).all() for value in self._policy.weights):
                raise FloatingPointError("Non-finite SAC parameter after optimizer update")
            state["updates"] += 1
            if state["updates"] == 1 or state["updates"] % 100 == 0:
                emit(dict(kind="optimizer", **{
                    "optimizer/" + name: int(getattr(self._policy, name + "_optimizer").iterations.numpy())
                    for name in ("actor", "qf1", "qf2", "alpha")
                    if hasattr(self._policy, name + "_optimizer")
                }))
            if should_stop():
                raise InterruptedError("Requested training stop at an optimizer boundary")
            return result

    try:
        if not args.no_wandb:
            from cutting_world_model.sidecar_tracking import SidecarTracker
            tracker = SidecarTracker(output, config, job_type="sac-gpu-evaluation" if args.evaluate else "sac-gpu-training")
        environment = SACGPUEnvironment(args.gpu_python, args.backend_root, on_step=on_step,
                                        timeout=args.worker_timeout, should_stop=should_stop)
        resources["environment"] = environment
        validate_task(parameters, environment.metadata)
        seed = parameters["ur3e_gym"].get("rand_seed", 0)
        environment.seed(seed)
        ros_parameters.get_time = lambda: environment.sim_time
        args.episode_max_steps = environment._max_episode_steps
        args.logdir = str(output / "learner")
        policy = make_policy(SAC, parameters, args)
        trainer = TrackedTrainer(policy, environment, args, test_env=None, seed=seed)
        resources["trainer"] = trainer
        (output / "learner_path.txt").write_text(trainer._output_dir + "\n")
        (output / "task.json").write_text(json.dumps(environment.metadata, indent=2) + "\n")
        if args.evaluate:
            if not args.model_dir:
                raise ValueError("Frozen evaluation requires --model-dir")
            restore_policy(tf, policy, args.model_dir)
            weights = [value.numpy().copy() for value in policy.weights]
            counters = [int(getattr(policy, name + "_optimizer").iterations.numpy())
                        for name in ("actor", "qf1", "qf2", "alpha")
                        if hasattr(policy, name + "_optimizer")]
            result = trainer.evaluate_policy(0)
            if any(not np.array_equal(old, new.numpy()) for old, new in zip(weights, policy.weights)):
                raise RuntimeError("SAC weights changed during frozen evaluation")
            after = [int(getattr(policy, name + "_optimizer").iterations.numpy())
                     for name in ("actor", "qf1", "qf2", "alpha")
                     if hasattr(policy, name + "_optimizer")]
            if counters != after:
                raise RuntimeError("SAC optimizer ran during evaluation")
            state["source_evaluation"] = list(map(float, result))
        else:
            trainer()
        state["state"] = "completed"
    except InterruptedError:
        state["state"] = "stopped"
    except BaseException as error:
        state.update(state="failed", error=repr(error))
        raise
    finally:
        from cutting_world_model.lifecycle import finalize_training
        trainer = resources.get("trainer")
        state["wall_seconds"] = time.monotonic() - started
        state["episode_results"] = episode_results

        def save():
            if trainer is not None and not args.evaluate:
                state["checkpoint"] = trainer.checkpoint_manager.save()
                if "replay" in resources:
                    save_replay_buffer(resources["replay"], trainer.replay_buffer_path)

        def record():
            (output / "result.json").write_text(json.dumps(state, indent=2) + "\n")
            (output / "progress.json").write_text(json.dumps(state, indent=2) + "\n")

        def finish():
            if tracker is not None:
                try:
                    tracker.log(dict(kind="finish", **state))
                finally:
                    tracker.finish(status=state)

        def close():
            if "environment" in resources:
                resources["environment"].close()

        errors = finalize_training(save, record, finish, close, state)
        events.write(json.dumps(dict(kind="finish", **state)) + "\n")
        events.close()
        if errors and sys.exc_info()[0] is None:
            raise RuntimeError("SAC finalization failed: " + repr(errors))


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--sliceit-root", required=True)
    bootstrap.add_argument("--config", required=True)
    preliminary, _ = bootstrap.parse_known_args()
    parameters = source_parameters(preliminary.config)
    tf, SAC, Trainer, ros_parameters, revision = load_source(preliminary.sliceit_root, parameters)
    parser = Trainer.get_argument()
    parser.add_argument("--sliceit-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend-root")
    parser.add_argument("--gpu-python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--output")
    parser.add_argument("--actor-worker", action="store_true")
    parser.add_argument("--protocol-fd", type=int)
    parser.add_argument("--worker-timeout", type=float, default=600)
    parser.add_argument("--wall-seconds", type=float, default=0)
    parser.add_argument("--no-wandb", action="store_true")
    parser.set_defaults(**SOURCE_DEFAULTS)
    args = parser.parse_args()
    if args.actor_worker:
        if not args.model_dir or args.protocol_fd is None:
            parser.error("Actor worker requires --model-dir and --protocol-fd")
        actor_worker(args, tf, SAC, parameters, revision)
    else:
        if not args.backend_root or not args.output:
            parser.error("Training and evaluation require --backend-root and --output")
        run(args, tf, SAC, Trainer, ros_parameters, parameters, revision)


if __name__ == "__main__":
    main()
