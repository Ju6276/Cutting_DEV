#!/usr/bin/env python3
"""Compare analytic SliceIt heads against recorded environment outputs and sample upstream replay."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutting_world_model.provenance import git_revision
from cutting_world_model.replay import replay_from_episodes
from cutting_world_model.task_heads import SliceItHeads, pose_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    root = Path(args.dataset).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    heads = SliceItHeads(manifest["environment"], manifest["controller"])
    reports = []
    paths = sorted(root.glob("episode_*.npz"))
    for path in paths:
        with np.load(path) as data:
            arrays = {key: torch.as_tensor(value) for key, value in data.items()}
        info = json.loads(path.with_suffix(".json").read_text())
        obs = arrays["observation"][1:]
        collision = torch.tensor([item["collision"] for item in info])
        success = torch.tensor([item["success"] for item in info])
        done, goal, outside, count = heads.termination(obs, arrays["success_counter"][:-1, 0], collision)
        reward = heads.reward(obs, collision, success, outside)
        error = pose_error(arrays["target_pose"][1:], arrays["previous_ee_pose"][1:])
        jerk = torch.diff(arrays["previous_acceleration"], dim=0) / manifest["environment"]["agent_control_dt"]
        rebuilt = heads.observation(error, arrays["previous_velocity"][1:], jerk,
                                    arrays["actions"], arrays["wrench_history"][1:])
        difference = (rebuilt - obs).abs()
        reports.append({"episode": path.name, "transitions": len(obs),
            "reward_max_abs_error": float((reward - arrays["rewards"]).abs().max()),
            "observation_max_abs_error": float(difference.max()),
            "observation_component_max_error": {name: float(difference[:, begin:end].max())
                for name, begin, end in [("pose", 0, 3), ("velocity", 3, 6), ("jerk", 6, 9),
                                         ("action", 9, 15), ("force", 15, 24)]},
            "termination_mismatches": int((done != arrays["terminated"]).sum()),
            "success_mismatches": int((goal != success).sum()),
            "counter_max_abs_error": float((count - arrays["success_counter"][1:, 0]).abs().max())})
    buffer, layout = replay_from_episodes(paths)
    batch = buffer.sample_windows(2048, 8)
    result = {"git": git_revision(Path(__file__).resolve().parents[1]),
              "episodes": reports, "replay_rows": buffer.size,
              "window_batch_shape": list(batch.states.shape)}
    (root / "interface_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
