#!/usr/bin/env python3
"""Compare one recorded control interval with a restart from the saved physical state."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cutting_world_model.provenance import git_revision
from disect.cutting import CuttingSim, load_settings


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--frame", type=int, required=True)
    args = parser.parse_args()
    directory = Path(args.dataset).resolve()
    record = json.loads((directory / "manifest.json").read_text())
    data = np.load(directory / "trajectory.npz")
    settings = load_settings(record["arguments"]["config"])
    settings.sim_dt = record["arguments"]["sim_dt"]
    sim = CuttingSim(settings, experiment_name="state_restart", adapter="cuda",
                     requires_grad=False, allow_nans=False)
    sim.cut()
    sim.state = sim.model.state()
    sim.disable_gradients()
    frame = args.frame
    for name in ("particle_q", "particle_qd", "cut_spring_ke"):
        getattr(sim.state, name).copy_(torch.as_tensor(data[name][frame], device="cuda"))
    pose = data["knife_pose"][frame]
    action = data["actions"][frame]
    sim.motion.set_position(pose[:3])
    sim.motion.set_rotation(pose[3:])
    sim.motion.set_linear_velocity(action[:3])
    sim.motion.set_angular_velocity(action[3:])
    sim.sim_time = float(data["time"][frame])
    sim.sim_step = round(sim.sim_time / sim.sim_dt)
    substeps = round(record["arguments"]["control_dt"] / sim.sim_dt)
    for _ in range(substeps):
        sim.simulation_step()
    result = {"frame": frame, "git": git_revision(Path(__file__).resolve().parents[1]),
              "released_at_restart": int(np.count_nonzero(data["cut_spring_ke"][frame] == 0))}
    for name in ("particle_q", "particle_qd", "cut_spring_ke"):
        actual = getattr(sim.state, name).cpu().numpy()
        difference = actual - data[name][frame + 1]
        result[name] = {"rms": float(np.sqrt(np.mean(difference.astype(np.float64)**2))),
                        "max_abs": float(np.max(np.abs(difference)))}
    (directory / "restart_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
