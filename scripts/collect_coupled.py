#!/usr/bin/env python3
"""Record original SliceIt transitions at env.step boundaries, without changing its task."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import torch

import numpy as np
import rospy
from std_srvs.srv import Trigger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutting_world_model.provenance import git_revision
from cutting_world_model.observation_recording import ObservationRecorder
from cutting_world_model.policy import SliceItPolicy



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sliceit-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--policy-seed", type=int, default=0)
    args = parser.parse_args(rospy.myargv()[1:])
    from cutting_world_model.coupled_env import CoupledEnvironment
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rospy.init_node("world_model_collector", anonymous=True)
    coupled = CoupledEnvironment(args.config, args.seed)
    env = coupled.env
    max_steps = coupled.environment["steps_per_episode"]
    (output / "task.json").write_text(json.dumps(coupled.metadata, indent=2) + "\n")
    torch.set_num_threads(2)
    torch.manual_seed(args.policy_seed)
    policy = SliceItPolicy().eval()
    root = Path(args.sliceit_root)
    manifest = {"git": git_revision(Path(__file__).resolve().parents[1]),
                "sliceit_git": git_revision(root), "disect_git": git_revision(root / "disect"),
                "robot_git": git_revision(root / "underlay_ws/src/ur_python_utilities"),
                "arguments": vars(args), "command": sys.argv,
                "ros_master": os.environ["ROS_MASTER_URI"],
                "environment": rospy.get_param("ur3e_gym"),
                "controller": rospy.get_param("ur3e_force_control"),
                "behavior_policy": "initial SuperMimic encoder/decoder, Gaussian action noise 0.1",
                "policy_seed": args.policy_seed, "policy_updates": 0,
                "action_bounds": [env.action_space.low.tolist(), env.action_space.high.tolist()]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    for episode in range(args.episodes):
        initial = coupled.reset()
        observation = initial["observation"]
        history = [observation.copy() for _ in range(policy.history)]
        frames = [initial]
        actions, clean_actions, rewards, terminated, truncated, infos = [], [], [], [], [], []
        for step in range(max_steps):
            with torch.no_grad():
                clean = policy(torch.tensor(np.asarray(history), dtype=torch.float32)[:, None],
                               torch.tensor(env.unwrapped.current_target_pose, dtype=torch.float32)[None])[0]
                executed = clean + .1 * torch.randn_like(clean)
            action = np.clip(executed.numpy(), env.action_space.low, env.action_space.high)
            frame, reward, term, trunc, info = coupled.step(action)
            observation = frame["observation"]
            done = term or trunc
            history = history[1:] + [observation.copy()]
            frames.append(frame)
            actions.append(action)
            clean_actions.append(clean.numpy())
            rewards.append(reward)
            is_truncated = bool(info.get("TimeLimit.truncated", False))
            terminated.append(bool(done and not is_truncated))
            truncated.append(is_truncated)
            infos.append({key: value.tolist() if isinstance(value, np.ndarray) else value
                          for key, value in info.items()})
            progress = {"episode": episode, "step": step + 1, "reward": float(reward),
                        "done": bool(done), "wall_seconds": time.monotonic() - started}
            (output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            print(json.dumps(progress), flush=True)
            if done:
                break
        data = {key: np.asarray([frame[key] for frame in frames]) for key in frames[0]}
        data.update(actions=np.asarray(actions), clean_actions=np.asarray(clean_actions), rewards=np.asarray(rewards),
                    terminated=np.asarray(terminated), truncated=np.asarray(truncated))
        np.savez_compressed(output / ("episode_%03d.npz" % episode), **data)
        (output / ("episode_%03d.json" % episode)).write_text(json.dumps(infos, indent=2) + "\n")
    (output / "report.json").write_text(json.dumps({"episodes": args.episodes,
        "wall_seconds": time.monotonic() - started}, indent=2) + "\n")


if __name__ == "__main__":
    main()
