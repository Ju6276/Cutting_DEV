#!/usr/bin/env python3
"""Record full cutting states at fixed control intervals; SI units, Y up."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cutting_world_model.provenance import git_revision
from disect.cutting import CuttingSim, load_settings


def snapshot(sim):
    state = sim.state
    force = getattr(state, "knife_f", None)
    return {
        "particle_q": state.particle_q.detach().cpu().numpy().copy(),
        "particle_qd": state.particle_qd.detach().cpu().numpy().copy(),
        "cut_spring_ke": state.cut_spring_ke.detach().cpu().numpy().copy(),
        "knife_pose": np.concatenate((
            sim.motion.current_pos.detach().cpu().numpy(),
            sim.motion.current_rot.detach().cpu().numpy(),
        )),
        "knife_velocity": np.concatenate((
            sim.motion.lin_vel.detach().cpu().numpy(),
            sim.motion.ang_vel.detach().cpu().numpy(),
        )),
        "force": np.array([0.0 if force is None else
                           torch.linalg.vector_norm(force, dim=1).sum().item()]),
        "time": np.array(sim.sim_time),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--disect-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--sim-dt", type=float, default=1e-5)
    parser.add_argument("--actions", help="NPY array of six-dimensional knife twists, m/s and rad/s")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = load_settings(str(Path(args.config).resolve()))
    settings.sim_dt = args.sim_dt
    ratio = args.control_dt / args.sim_dt
    substeps = round(ratio)
    if not np.isclose(substeps, ratio):
        raise ValueError("control_dt must contain an integer number of physics steps")
    if args.actions:
        actions = np.load(args.actions).astype(np.float32)
    else:
        count = round(settings.sim_duration / args.control_dt)
        actions = np.zeros((count, 6), dtype=np.float32)
        actions[:, 1] = settings.velocity_y
    if actions.ndim != 2 or actions.shape[1] != 6 or not np.isfinite(actions).all():
        raise ValueError("Expected finite actions with shape (steps, 6)")
    manifest = {
        "git": git_revision(ROOT),
        "disect_git": git_revision(args.disect_root),
        "arguments": vars(args),
        "settings": dict(settings),
        "command": sys.argv,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    np.save(output / "actions.npy", actions)
    started = time.perf_counter()
    sim = CuttingSim(settings, experiment_name="world_model_data", adapter="cuda",
                     requires_grad=False, allow_nans=False)
    sim.cut()
    sim.state = sim.model.state()
    sim.disable_gradients()
    topology = {name: getattr(sim.model, name).detach().cpu().numpy()
                for name in ("cut_edge_indices", "cut_edge_coords", "cut_tri_indices",
                             "cut_virtual_tri_indices", "cut_spring_indices")}
    topology["triangles"] = np.asarray(sim.builder.tri_indices)
    topology["cut_spring_kd"] = sim.state.cut_spring_kd.detach().cpu().numpy()
    topology["knife_vertices"], topology["knife_triangles"] = sim.knife.create_mesh()
    np.savez_compressed(output / "topology.npz", **topology)
    frames = [snapshot(sim)]
    for index, action in enumerate(actions):
        sim.motion.set_linear_velocity(action[:3])
        sim.motion.set_angular_velocity(action[3:])
        for _ in range(substeps):
            sim.simulation_step()
        frame = snapshot(sim)
        if any(not np.isfinite(value).all() for value in frame.values()):
            raise FloatingPointError("Non-finite simulation state at control step %d" % index)
        frames.append(frame)
        progress = {"completed_steps": index + 1, "total_steps": len(actions),
                    "simulation_seconds": sim.sim_time,
                    "wall_seconds": time.perf_counter() - started}
        (output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        if (index + 1) % 5 == 0:
            print(json.dumps(progress), flush=True)
    arrays = {key: np.asarray([frame[key] for frame in frames]) for key in frames[0]}
    arrays["actions"] = actions
    np.savez_compressed(output / "trajectory.npz", **arrays)
    progress.update({"vertices": arrays["particle_q"].shape[1],
                     "springs": arrays["cut_spring_ke"].shape[1],
                     "released_initial": int(np.count_nonzero(arrays["cut_spring_ke"][0] == 0)),
                     "released_final": int(np.count_nonzero(arrays["cut_spring_ke"][-1] == 0)),
                     "peak_force": float(arrays["force"].max()),
                     "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20})
    (output / "report.json").write_text(json.dumps(progress, indent=2) + "\n")
    print(json.dumps(progress), flush=True)


if __name__ == "__main__":
    main()
