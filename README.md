# Cutting GPU Backend

The current SliceIt **UR5e** `b_bot` robot on **MuJoCo Warp 3.16.0**, the physical backend pinned by SuperMimic-Reproduced `ba6105349069bf040f1fdad60873d13138d505a2` (mjlab `8bb2bc9e187797689ec82c78c35019afd2fd0376`, Warp 1.18.0).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/prepare_robot.py \
  --task runs/source/task.json \
  --source-root /dataHDD/dong/Projects/SliceIt/underlay_ws/src \
  --source-root /dataHDD/dong/Projects/SliceIt/catkin_ws/src \
  --source-root /opt/ros/noetic/share --output runs/model
PYTHONPATH=. .venv/bin/python -m pytest tests -q
.venv/bin/python scripts/validate_robot.py --episode runs/source/episode_000.npz --output runs/fk.json
timeout 180 .venv/bin/python -u scripts/benchmark_ur5e.py --worlds 1 --steps 100 --output runs/robot_1/report.json
```

Copy `task.json` and `episode_000.npz` from the current SliceIt interface capture into `runs/source/`. The importer resolves original ROS assets in place, preserves collision meshes and inertias, and combines fixed-link inertias by the parallel-axis theorem. Source assets retain their original licenses and are not redistributed.

The benchmark applies named motor torques to the six current robot joints. One captured CUDA graph updates controls and integrates one 2 ms step. It measures synchronized world-steps/s, JIT/setup time, process GPU memory, finite states, joint motion and CPU MuJoCo agreement. Start with 1 and 32 worlds and inspect resources before expanding. Runs remain outside Git.

[Method and coupling work](docs/METHOD.md) · [Measured results](docs/RESULTS.md).

RTX 4090: **4096 UR5e robot worlds** measured at **908,648 world-steps/s**, **3558 MiB** process memory after stepping, with existing training active. The original controller and DiSECt coupling are the next integration stage.
