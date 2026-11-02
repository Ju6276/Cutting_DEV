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

RTX 4090: **4096 UR5e robot worlds** measured at **908,648 world-steps/s**, **3558 MiB** process memory after stepping, with existing training active. The single-world controller/material loop below measures complete cutting separately.

## Single-world cutting loop

The tensor FDCC controller, headless DiSECt core and MuJoCo Warp robot connect
in `scripts/run_coupled.py`. It replays recorded SliceIt actions with 1 ms robot
updates, 2 ms force/motion exchange and the original force/goal checks.

Prepare the DiSECt model once in the existing SliceIt environment, from its
`disect/` directory (Xvfb serves the source package's import-time Qt dependency):

```bash
xvfb-run -a python /dataHDD/dong/Projects/CuttingGpuBackend/scripts/export_disect_scene.py \
  --disect-root . --config ../configs/public_prism.json \
  --output /dataHDD/dong/Projects/CuttingGpuBackend/runs/disect_source --steps 200
```

Then use the new backend environment, without ROS or a display:

```bash
export TORCH_EXTENSIONS_DIR=/dataHDD/dong/.cache/torch_extensions_cutting_gpu
export CUDA_HOME=/usr/local/cuda-12.9 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export XDG_CACHE_HOME=/dataHDD/dong/.cache TMPDIR=/dataHDD/dong/tmp
.venv/bin/python scripts/validate_disect.py --scene runs/disect_source --output runs/disect_check.json
timeout 660 .venv/bin/python -u scripts/run_coupled.py \
  --scene runs/disect_source/scene.pt --wall-seconds 600 --output runs/coupled_cut
```

The controller parameter snapshot is `runs/controller_source/controller_params.json`.
Import the coupled robot with `prepare_robot.py --output runs/model_coupled`;
this includes the source engine's adjacent-body collision exclusions.
The runner writes per-action progress, particle trajectories and a report with
the termination reason, released springs, force, runtime and force-feedback
response. `export_disect_scene.py --warmup-steps 65000` prepares a reference from
inside a cut for checking material/contact continuation across environments.
`run_coupled.py --hold-last-action` continues with the final recorded action
until the original task terminates or its 500-action budget expires.
`--actions-file PATH.npz` reads a separate recorded `actions` array, retaining
the same initial robot/material state and controller formulas.
`--scene-motion` follows the exported DiSECt constant-velocity reference for
its configured duration, using the captured controller settings. It exercises
robot/material coupling independently of a learned policy.

Single-world reference result: original task goal reached in **2.0 s simulation /
162.70 s wall time**, peak force **9.97 N**, **288/304** springs released.
The complete backend test suite has **18 passing tests**.

## A6000 host

`tamsgpu5` (`134.100.13.216`) is reachable through `tamsgpu9`. From a Mac
with the existing 4090 SSH master:

```bash
ssh -o 'ProxyCommand=ssh -S /tmp/sliceit-live.sock dong@tamsgpu9 -W %h:%p' \
  -o HostKeyAlias=tamsgpu5 dong@134.100.13.216
```

The backend is at `/dataHDD/dong/Projects/CuttingGpuBackend`, with its own
`.venv`. Use the environment variables above with `CUDA_VISIBLE_DEVICES=1`
and `TORCH_CUDA_ARCH_LIST=8.6`. Imported mesh assets live in
`runs/mesh_assets`; the exported URDF paths point there. Run the same reference:

```bash
timeout 660 .venv/bin/python -u scripts/run_coupled.py \
  --scene-motion --wall-seconds 600 --output runs/a6000_reference
```

The world-model source is at `/dataHDD/dong/Projects/CuttingWorldModel`.
`CuttingBatch` batches independent material, controller and robot states.
Its task/reward adapter and online training entry are `cutting_world_model/gpu_env.py`
and `scripts/train_gpu.py` in that repository. `scripts/benchmark_gpu_env.py`
measures complete reference cuts at the requested environment count.
