"""Run SliceIt's source SAC trainer against the ROS-free, single-world backend."""
import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
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
RESUME_STATE = "resume_state.json"
OPTIMIZERS = ("actor", "qf1", "qf2", "alpha")
# Learner settings that must match between a stopped run and its continuation.
RESUME_ARGUMENTS = (
    "batch_size", "n_warmup", "save_model_interval", "normalize_obs", "auto_alpha",
    "use_prioritized_rb", "use_nstep_rb", "n_step", "lr", "update_interval",
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


def optimizer_iterations(policy):
    return {name: int(getattr(policy, name + "_optimizer").iterations.numpy())
            for name in OPTIMIZERS if hasattr(policy, name + "_optimizer")}


def checkpoint_iterations(tf, checkpoint):
    reader = tf.train.load_checkpoint(str(checkpoint))
    key = "policy/{}_optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE"
    return {name: int(reader.get_tensor(key.format(name)))
            for name in OPTIMIZERS if reader.has_tensor(key.format(name))}


def replay_rows(path):
    with Path(path).open("rb") as stream:
        transitions = pickle.load(stream)
    rows = {len(value) for value in transitions.values()}
    if len(rows) != 1:
        raise ValueError("Replay fields have different lengths in " + str(path))
    return rows.pop()


def load_resume(parent, args, parameters):
    """Return the last save whose checkpoint, replay and counters agree."""
    parent = Path(parent).resolve()
    config = json.loads((parent / "run_config.json").read_text())
    if config["source_parameters"] != parameters:
        raise ValueError("Resume requires the parent run's source parameters")
    previous = config["arguments"]
    if previous.get("evaluate"):
        raise ValueError("Cannot resume training from a frozen evaluation output")
    for key in RESUME_ARGUMENTS:
        if previous.get(key) != getattr(args, key):
            raise ValueError("Resume changes learner setting " + key)
    path = parent / RESUME_STATE
    if path.exists():
        record = json.loads(path.read_text())
    else:
        # Outputs written before resume support record their final save in result.json.
        result = json.loads((parent / "result.json").read_text())
        if result.get("state") != "stopped" or not result.get("checkpoint"):
            raise ValueError("Parent has no resumable save: " + str(parent))
        learner = Path((parent / "learner_path.txt").read_text().strip())
        record = {key: result[key] for key in ("state", "steps", "updates", "episodes",
                                               "episode_results", "checkpoint")}
        record["replay"] = str(learner / "replay_buffer.pkl")
    completed = sum(item["steps"] for item in record["episode_results"])
    if len(record["episode_results"]) != record["episodes"] or completed > record["steps"]:
        raise ValueError("Parent episode records do not match its step count")
    if not 0 < record["steps"] < args.max_steps:
        raise ValueError("Parent steps %d leave nothing to train before --max-steps %d"
                         % (record["steps"], args.max_steps))
    rows = replay_rows(record["replay"])
    if rows != record["steps"]:
        raise ValueError("Parent replay has %d rows for %d steps" % (rows, record["steps"]))
    # Transitions of an unfinished episode stay in replay; the episode restarts from reset.
    return dict(record, parent=str(parent), replay_rows=rows,
                discarded_episode_steps=record["steps"] - completed)


def actor_worker(args, tf, SAC, parameters, revision):
    policy = make_policy(SAC, parameters, args)
    checkpoint = restore_policy(tf, policy, args.model_dir)
    weights = [value.numpy().copy() for value in policy.weights]
    counters = optimizer_iterations(policy)

    def handle(message):
        if message["op"] == "init":
            return dict(
                checkpoint=checkpoint, source=revision,
                observation_shape=[observation_size(parameters["ur3e_gym"])], action_shape=[6],
                source_parameters=parameters,
                optimizer_iterations=counters,
            )
        if message["op"] == "act":
            observation = np.asarray(message["observation"], dtype=np.float32)
            action = policy.get_action(observation, test=message["deterministic"])
            return dict(action=np.asarray(action).tolist())
        if message["op"] == "unchanged":
            unchanged = len(weights) == len(policy.weights)
            unchanged = unchanged and all(np.array_equal(old, new.numpy())
                                          for old, new in zip(weights, policy.weights))
            unchanged = unchanged and counters == optimizer_iterations(policy)
            return dict(unchanged=unchanged)
        if message["op"] == "close":
            return {}
        raise ValueError("Unknown actor operation: " + message["op"])

    serve(args.protocol_fd, handle)


def run(args, tf, SAC, Trainer, ros_parameters, parameters, revision):
    from tf2rl.misc.get_replay_buffer import save_replay_buffer
    from tf2rl.experiments import cb_trainer
    if args.model_dir and not args.evaluate:
        raise ValueError("Continue unified-environment training with --resume; use --evaluate for old checkpoints")
    if args.resume and args.evaluate:
        raise ValueError("--resume continues training; frozen evaluation uses --model-dir")
    resume = load_resume(args.resume, args, parameters) if args.resume else None
    offset = resume["steps"] if resume else 0
    if resume:
        iterations = checkpoint_iterations(tf, resume["checkpoint"])
        if not iterations or set(iterations.values()) != {resume["updates"]}:
            raise ValueError("Parent checkpoint optimizer iterations %s differ from %d updates"
                             % (iterations, resume["updates"]))
        resume["optimizer_iterations"] = iterations
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = dict(
        code=git_revision(ROOT), tf2rl=revision, arguments=vars(args),
        source_parameters=parameters, physical_backend="mujoco_warp_disect", worlds=1,
    )
    if resume:
        config["resume"] = {key: value for key, value in resume.items() if key != "episode_results"}
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    started = time.monotonic()
    state = dict(state="running", steps=offset, updates=resume["updates"] if resume else 0,
                 episodes=resume["episodes"] if resume else 0)
    if resume:
        state["resumed_from"] = resume["parent"]
    episode = dict(steps=0, reward=0.)
    episode_results = list(resume["episode_results"]) if resume else []
    resources = {}
    saved = dict(steps=None)
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

    def save_resumable(final=False):
        """Save checkpoint and replay; record them for --resume when counters agree."""
        trainer, replay = resources["trainer"], resources["replay"]
        iterations = optimizer_iterations(trainer._policy)
        rows = int(replay.get_stored_size())
        consistent = rows == state["steps"] and set(iterations.values()) <= {state["updates"]}
        if not consistent and not final:
            raise RuntimeError("Replay rows %d, optimizer %s, steps %d, updates %d disagree"
                               % (rows, iterations, state["steps"], state["updates"]))
        checkpoint = trainer.checkpoint_manager.save()
        if consistent:
            save_replay_buffer(replay, trainer.replay_buffer_path)
            record = dict(
                state=state["state"], steps=state["steps"], updates=state["updates"],
                episodes=state["episodes"], episode_results=list(episode_results),
                episode_in_progress=dict(episode), checkpoint=checkpoint,
                replay=trainer.replay_buffer_path, replay_rows=rows,
                optimizer_iterations=iterations, wall_seconds=time.monotonic() - started,
            )
            temporary = output / (RESUME_STATE + ".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(output / RESUME_STATE)
            saved["steps"] = state["steps"]
        return checkpoint

    def step_boundary():
        # Source saves every save_model_interval steps; count them across resumed runs.
        if state["steps"] % args.save_model_interval == 0 and saved["steps"] != state["steps"]:
            emit(dict(kind="checkpoint", checkpoint=save_resumable()))

    install_terminal_replay_factory(
        cb_trainer, on_create=lambda replay: resources.__setitem__("replay", replay),
        after_episode=step_boundary,
    )

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
        def __call__(self):
            try:
                return super().__call__()
            except ZeroDivisionError:
                # Source averages over episodes of this process; a short resumed tail may have none.
                if state["steps"] < args.max_steps:
                    raise
                return None

        def update_policy(self, replay_buffer, *values, **kwargs):
            result = super().update_policy(replay_buffer, *values, **kwargs)
            if "restore" in resources:
                # Adam slots are created, and filled from the checkpoint, by the first update.
                resources.pop("restore").assert_consumed()
            if any(not np.isfinite(value.numpy()).all() for value in self._policy.weights):
                raise FloatingPointError("Non-finite SAC parameter after optimizer update")
            state["updates"] += 1
            if state["updates"] in (1, first_update) or state["updates"] % 100 == 0:
                emit(dict(kind="optimizer", **{
                    "optimizer/" + name: value
                    for name, value in optimizer_iterations(self._policy).items()
                }))
            step_boundary()
            if should_stop():
                raise InterruptedError("Requested training stop at an optimizer boundary")
            return result

        def _detailed_log(self, n_episode, total_steps, episode_steps, episode_return):
            return super()._detailed_log(n_episode + episode_offset, total_steps + offset,
                                         episode_steps, episode_return)

    first_update = state["updates"] + 1
    episode_offset = state["episodes"]

    try:
        if not args.no_wandb:
            from cutting_world_model.sidecar_tracking import SidecarTracker
            tracker = SidecarTracker(output, config, job_type="sac-gpu-evaluation" if args.evaluate else "sac-gpu-training")
        environment = SACGPUEnvironment(args.gpu_python, args.backend_root, on_step=on_step,
                                        timeout=args.worker_timeout, should_stop=should_stop,
                                        worker=args.gpu_worker)
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
        # Source Trainer counts from zero in each process; continue the total count.
        trainer._max_steps = args.max_steps - offset
        policy.n_warmup = max(0, args.n_warmup - offset)
        state["warmup_steps_remaining"] = policy.n_warmup
        trainer._save_model_interval = sys.maxsize  # step_boundary saves on total steps
        if resume:
            restore = trainer._checkpoint.restore(resume["checkpoint"])
            restore.assert_existing_objects_matched()
            if set(optimizer_iterations(policy).values()) != {resume["updates"]}:
                raise RuntimeError("Restored optimizer iterations differ from parent updates")
            resources["restore"] = restore
            # Source __call__ loads this file into the new replay buffer before its first reset.
            shutil.copyfile(resume["replay"], trainer.replay_buffer_path)
            log = Path(resume["checkpoint"]).parent / "detailed_log.npy"
            if log.exists():
                rows = [row for row in np.load(log, allow_pickle=True).tolist() if row[1] <= offset]
                if rows:
                    np.save(Path(trainer._output_dir) / "detailed_log.npy", rows)
            set_step = tf.summary.experimental.set_step
            tf.summary.experimental.set_step = lambda step: set_step(step + offset)
        (output / "task.json").write_text(json.dumps(environment.metadata, indent=2) + "\n")
        if args.evaluate:
            if not args.model_dir:
                raise ValueError("Frozen evaluation requires --model-dir")
            restore_policy(tf, policy, args.model_dir)
            weights = [value.numpy().copy() for value in policy.weights]
            counters = optimizer_iterations(policy)
            result = trainer.evaluate_policy(0)
            if any(not np.array_equal(old, new.numpy()) for old, new in zip(weights, policy.weights)):
                raise RuntimeError("SAC weights changed during frozen evaluation")
            if counters != optimizer_iterations(policy):
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
                if "restore" in resources:
                    # Adam slots are still pending restoration; the parent save remains the resume point.
                    state.update(checkpoint=None, resumable=False)
                elif "replay" in resources:
                    state["checkpoint"] = save_resumable(final=True)
                    state["resumable"] = saved["steps"] == state["steps"]
                else:
                    state["checkpoint"] = trainer.checkpoint_manager.save()

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
    parser.add_argument("--resume", help="Output directory of a stopped training run to continue")
    parser.add_argument("--gpu-worker", help="Physical worker script; tests substitute a CPU stub")
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
