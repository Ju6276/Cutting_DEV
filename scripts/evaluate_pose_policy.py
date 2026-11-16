#!/usr/bin/env python3
"""Frozen, noise-free first-episode evaluation for either multi-object pose backend.

One scored episode per world, fixed before execution. Other worlds may reset while
unfinished cases continue; their later episodes are excluded. No optimization.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(os.environ.get("CUTTING_EVAL_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT), str(ROOT / "backends/mujoco")]
from cutting_world_model.episode_log import episode_row
from cutting_world_model.policy import SliceItPolicy
from cutting_world_model.provenance import git_revision


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summarize(rows, expected):
    result = {"expected_cases": expected, "completed_cases": len(rows),
              "complete": len(rows) == expected}
    if not rows:
        return result
    for field in ("success", "collision", "outside", "timeout"):
        result[field + "_rate"] = sum(bool(r.get(field, False)) for r in rows) / len(rows)
    for field in ("return", "steps", "peak_material_force_n", "peak_limit_ratio",
                  "material_impulse_ns", "speed_error_mps"):
        result["mean_" + field] = sum(r[field] for r in rows) / len(rows)
    if all("object_slip_failure" in r for r in rows):
        result["object_slip_failure_rate"] = sum(r["object_slip_failure"] for r in rows) / len(rows)
    if all("tool_motion_failure" in r for r in rows):
        result["tool_motion_failure_rate"] = sum(r["tool_motion_failure"] for r in rows) / len(rows)
    if all("completion_xy_error_m" in r for r in rows):
        result["mean_completion_xy_error_m"] = sum(r["completion_xy_error_m"] for r in rows) / len(rows)
    if all("object_slip_m" in r for r in rows):
        result["mean_terminal_object_slip_m"] = sum(r["object_slip_m"] for r in rows) / len(rows)
    if all("separated" in r for r in rows):
        result["separated_rate"] = sum(r["separated"] for r in rows) / len(rows)
    if all("material_components" in r for r in rows):
        # Multiple components may include fragments: not proof of the intended slice.
        result["multiple_components_rate"] = sum(r["material_components"] > 1 for r in rows) / len(rows)
    return result


def validate_interfaces(saved, actual, config):
    if config["task"] != saved["free_cut_task"]:
        raise ValueError("Evaluation task/reward differs from the checkpoint")
    for key in ("environment", "controller", "policy_observation_revision",
                "physical_backend", "task_interface_revision", "tool_revision",
                "progress_semantics", "board_pose_geometry", "contact_discretization_revision"):
        if saved.get(key) != actual.get(key):
            raise ValueError("Checkpoint/environment mismatch: " + key)
    if saved["state_shapes"]["observation"] != actual["state_shapes"]["observation"]:
        raise ValueError("Observation dimensions differ")


def evaluation_config(source, worlds_per_object, seed, blademaster=False):
    """Regenerate paired evaluation cases, including explicit clearances and episode RNG."""
    from cutting_world_model.shared_scene import sample_clearances
    from cutting_world_model.scene_alignment import validate_alignment
    if type(worlds_per_object) is not int or worlds_per_object < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Positive world count and nonnegative evaluation seed required")
    config = copy.deepcopy(source)
    config["seed"] = seed
    if config.get("episode_conditions") is not None:
        config["episode_conditions"]["seed"] = seed
    aligned = config["task"].get("scene_alignment")
    clearances = []
    for group in config["groups"]:
        if not aligned and group.get("start_clearance_m") is not None:
            raise ValueError("Explicit clearances require a shared scene alignment for case regeneration")
        group["worlds"] = worlds_per_object
        if aligned:
            bounds = validate_alignment(aligned)["clearance_range_m"]
            values = sample_clearances(seed, group["name"], worlds_per_object, bounds)
            if not blademaster:
                group["start_clearance_m"] = values
            clearances.extend(values)
    if blademaster:
        config["worlds"] = sum(g["worlds"] for g in config["groups"])
        if aligned:
            config["case_seed"] = seed
            config["start_clearance_m"] = clearances
        elif config.get("start_clearance_m") is not None:
            raise ValueError("Explicit clearances require shared scene alignment")
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    c = p.add_mutually_exclusive_group(required=True)
    c.add_argument("--conditions", type=Path)
    c.add_argument("--blademaster-config", type=Path)
    p.add_argument("--worlds-per-object", type=int, default=16)
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--max-hours", type=float, default=1)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--record-geometry", action="store_true", help="Capture physical states for video; no change to policy or task")
    args = p.parse_args()
    if args.worlds_per_object < 1 or args.max_hours <= 0 or args.seed < 0:
        p.error("Positive world count and wall budget required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"state": "initializing", "pid": os.getpid(), "cases_done": 0}
    def write(name, value):
        dest = args.output / name
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        tmp.replace(dest)
    def progress(**values):
        status.update(values, wall_seconds=time.monotonic() - started)
        write("progress.json", status)
    progress()
    env = None
    rows, traces = [], []
    expected = 0
    group_names = []
    try:
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        metadata = saved["metadata"]
        config_path = args.conditions or args.blademaster_config
        config = json.loads(config_path.read_text())
        source_config = copy.deepcopy(config)
        config = evaluation_config(source_config, args.worlds_per_object, args.seed,
                                   blademaster=bool(args.blademaster_config))
        groups = config["groups"]
        expected = sum(g["worlds"] for g in groups)
        group_names = [g["name"] for g in groups for _ in range(g["worlds"])]
        policy = SliceItPolicy(int(metadata["goal_dim"]), int(metadata["state_shapes"]["observation"][0])).cuda().eval()
        policy.load_state_dict(saved["learner"]["policy"])
        learner = saved["learner"]
        identity = {k: learner[k] for k in ("updates", "wm_updates", "policy_updates", "training_stage") if k in learner}
        manifest = {"git": git_revision(ROOT), "script_sha256": file_hash(__file__),
                    "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_hash(args.checkpoint),
                    "checkpoint_identity": identity, "config_source": str(config_path),
                    "config_source_sha256": file_hash(config_path), "source_config": source_config,
                    "config": config, "seed": args.seed, "expected_cases": expected,
                    "policy": "frozen, no exploration noise, one scored episode per world",
                    "scope": "development snapshot; task and physics identity recorded explicitly",
                    "command": sys.argv}
        write("manifest.json", manifest)
        del saved, learner
        progress(phase="environment_initialization")
        if args.blademaster_config:
            from cutting_world_model.blademaster_backend import BladeMasterEnvironment
            env = BladeMasterEnvironment(ROOT / "backends/mujoco", config)
        else:
            from cutting_world_model.multi_env import MultiConditionEnvironment
            env = MultiConditionEnvironment.from_config(ROOT / "backends/mujoco", config)
        from cutting_world_model.control_state import ControlStateSpec
        frame = env.current
        actual = ControlStateSpec.metadata_from_frame(env.metadata, {k: v[0].cpu().numpy() for k, v in frame.items()})
        validate_interfaces(metadata, actual, config)
        conditions = env.conditions(range(env.worlds)) if hasattr(env, "conditions") else [
            {"object": name} for name in group_names]
        write("cases.json", [{"case": w, "group": group_names[w], "condition": c} for w, c in enumerate(conditions)])
        history = frame["observation"].float()[None].repeat(policy.history, 1, 1)
        n, device = env.worlds, env.device
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        acc = {k: torch.zeros(n, device=device) for k in ("return", "steps", "force", "ratio", "impulse", "speed")}
        horizon = int(config["task"]["horizon_steps"])
        def capture(step):
            if args.record_geometry:
                from cutting_world_model.training_snapshot import capture_training_environment
                capture_training_environment(env, ROOT / "backends/mujoco",
                    args.output / "geometry" / ("%04d" % step),
                    {"iteration": step, "training_stage": "frozen_evaluation",
                     "policy_updates": identity.get("updates"), "seed": args.seed,
                     "ee_pose_base": frame["ee_pose"].detach().cpu().tolist()})
        capture(0)
        with (args.output / "episodes.jsonl").open("w") as log, torch.no_grad():
            for step in range(1, horizon + 2):
                if time.monotonic() - started > args.max_hours * 3600:
                    raise TimeoutError("Evaluation wall budget reached")
                active = ~finished
                goal = env.goal(frame)
                clean = policy(history, goal)
                action = clean.clamp(-1, 1)
                if not bool(torch.isfinite(action).all()):
                    raise FloatingPointError("Nonfinite policy action")
                before = {k: frame[k].detach().cpu().clone() for k in (
                    "observation", "joint_q", "ee_pose", "step_count", "cut_progress", "peak_wrench")}
                frame, reward, terminated, truncated, info = env.step(action)
                if not bool(torch.isfinite(frame["observation"]).all() & torch.isfinite(reward).all()):
                    raise FloatingPointError("Nonfinite physics observation/reward")
                capture(step)
                acc["return"] += torch.where(active, reward, 0)
                acc["steps"] += active
                acc["force"] = torch.maximum(acc["force"], torch.where(active, info["peak_force"], 0))
                acc["ratio"] = torch.maximum(acc["ratio"], torch.where(active, info["safety_margin"], 0))
                acc["impulse"] += torch.where(active, info["material_impulse"], 0)
                error = (env.heads.cutting_speed(frame["observation"]) - frame["target_speed"].reshape(n)).abs()
                acc["speed"] += torch.where(active, error, 0)
                traces.append({"before": before, "active": active.cpu().clone(), "action": action.cpu(),
                               "clean_action": clean.cpu(), "goal": goal.cpu(), "reward": reward.cpu(),
                               "terminated": terminated.cpu(), "truncated": truncated.cpu(),
                               "next_observation": frame["observation"].cpu().clone(),
                               "next_pose": frame["ee_pose"].cpu().clone(),
                               "next_step_count": frame["step_count"].cpu().clone(),
                               "info": {k: v.cpu().clone() for k, v in info.items() if isinstance(v, torch.Tensor)}})
                ended = terminated | truncated
                for w in (ended & active).nonzero().flatten().tolist():
                    row = episode_row(w, 0, step, acc["return"], acc["steps"], info, truncated,
                                      acc["force"], acc["ratio"], conditions[w])
                    row.update(case=w, group=group_names[w],
                               material_impulse_ns=float(acc["impulse"][w]),
                               speed_error_mps=float(acc["speed"][w] / acc["steps"][w]))
                    rows.append(row)
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                finished |= ended
                if step % 10 == 0 or bool(finished.all()):
                    progress(state="running", phase="physics", step=step, cases_done=int(finished.sum()), expected_cases=expected)
                if bool(finished.all()):
                    break
                history = torch.cat((history[1:], frame["observation"].float()[None]), 0)
                if bool(ended.any()):
                    frame = env.reset(ended)
                    history[:, ended] = frame["observation"][ended].float()[None]
            if not bool(finished.all()):
                raise RuntimeError("Task did not terminate all first episodes within its horizon")
        progress(state="completed")
    except BaseException as error:
        progress(state="failed", error=repr(error))
        raise
    finally:
        try:
            torch.save(traces, args.output / "trajectories.pt")
            summary = {"state": status["state"], "overall": summarize(rows, expected),
                       "groups": {g: summarize([r for r in rows if r["group"] == g], group_names.count(g))
                                  for g in sorted(set(group_names))}}
            write("summary.json", summary)
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
