# Running Cutting DEV

Commands below run from the repository root. The GPU backend is bundled at
`backends/mujoco`; `--backend-root` can select a separate prepared checkout.

## GPU environment

Use Linux, Python 3.12 and NVIDIA CUDA. On the existing workstations, place the
checkout, environment, caches and outputs under `/dataHDD/dong`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-gpu.txt
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export XDG_CACHE_HOME=/dataHDD/dong/.cache
export TMPDIR=/dataHDD/dong/tmp
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions_cutting_gpu"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor_cutting"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton_cutting"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp/cutting_gpu_backend"
export TORCH_CUDA_ARCH_LIST=8.6  # A6000; use 8.9 on RTX 4090
export MAX_JOBS=2
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
```

Choose the available card with `CUDA_VISIBLE_DEVICES`. W&B uses the current
login or standard environment variables. Credentials stay outside this checkout.
The ROS logger uses `CUTTING_LOGGER_PYTHON`, defaulting to this root's
`.venv/bin/python`.

## Prepare robot and material assets

The existing exported task contains the URDF, reset state and task settings.
The DiSECt export contains the material model, cutting springs and initial
state. The controller snapshot in `configs/public_prism/controller.json`
is copied from the configuration used in the recorded runs.

On the 4090 workstation, reconstruct the bundled backend's assets with:

```bash
.venv/bin/python scripts/prepare_backend.py \
  --capture /dataHDD/dong/Projects/CuttingGpuBackend/runs/source \
  --controller configs/public_prism/controller.json \
  --scene /dataHDD/dong/Projects/CuttingGpuBackend/runs/disect_source_v4 \
  --source-root /dataHDD/dong/Projects/SliceIt/underlay_ws/src \
  --source-root /dataHDD/dong/Projects/SliceIt/catkin_ws/src \
  --source-root /opt/ros/noetic/share
```

It copies the input capture and exported scene, and builds the MuJoCo robot
from the resolved ROS meshes. It refuses to replace existing assets.

For a new machine, restore and build the pinned SliceIt sources first
([source checkout](SOURCES.md)). Place the official
[DiSECt dataset](https://github.com/NVlabs/DiSECt/tree/main/dataset) under
`external/SliceIt/disect/dataset`, following its download instructions.

1. Collect a reset and physical episode using the Gazebo collection command
   below. Its `data/` directory supplies `task.json` and `episode_000.npz`.
2. Export the material in the built SliceIt Python environment. Set
   `CUTTING_DEV_ROOT` to this repository's absolute path before sourcing:
   `source external/SliceIt/scripts/public_env.sh`. Then run:

```bash
xvfb-run -a python "$CUTTING_DEV_ROOT/backends/mujoco/scripts/export_disect_scene.py" \
  --disect-root "$SLICEIT_ROOT/disect" \
  --config "$SLICEIT_ROOT/configs/public_prism.json" \
  --output "$CUTTING_DEV_ROOT/runs/material_export"
```

3. In the Python 3.12 environment, run `prepare_backend.py` with the new
   capture, material export, controller JSON and restored ROS package roots.

Assets are stored under `backends/mujoco/runs/`:
`source/`, `model_coupled/`, `controller_source/` and `disect_source_v4/`.
They remain outside Git.

## GPU training and evaluation — run 3

The commands for the current working baseline (resumed from policy62/WM562 with
critic reset and 100 pre-fit rounds) and for the `source` configuration (script
defaults, from scratch) are in [README](../README.md#run); the two must not be
confused. Without variant flags, `train_gpu.py` runs `source`: imagined critic
targets, 32-step policy rollout, no BC (`--critic-target imagined`, `--actor-horizon`
unset = 32, `--actor-bc-alpha` unset = off). Variant flags are documented in
[METHOD](METHOD.md).

The maximum eight-step WM fit, effective batches 2048/4096 and original task
rewards are defined in the source configuration.
Training collects new physical transitions, updates WM, then policy/critic.
The first 500 WM rounds leave policy/critic unchanged.

`--resume RUN/last.pt` restores networks, optimizers and training counters.
Physics and replay restart, then the original replay warmup runs.
SIGUSR1 saves after the current iteration. Completion/SIGTERM writes
`last.pt` and `progress.json`. Frozen evaluation performs zero optimizer updates.

The shared fixed-environment evaluation commands are in [BENCHMARK](BENCHMARK.md).
The current task list is in [PLAN](PLAN.md).
Checkpoints from before the absorbing-terminal repair are usable for frozen
evaluation, while training requires the current terminal-rule metadata.

## Gazebo collection and training — runs 1 and 2

These commands require a built SliceIt ROS Noetic checkout and its Python 3.8
environment. Export `CUTTING_LOGGER_PYTHON` to the modern environment above.
Use an absolute project path when sourcing `public_env.sh`, which changes cwd.

```bash
export CUTTING_DEV_ROOT="$PWD"
export CUTTING_LOGGER_PYTHON="$PWD/.venv/bin/python"
export SLICEIT_ROOT=/dataHDD/dong/Projects/SliceIt
source "$SLICEIT_ROOT/scripts/public_env.sh"
cd "$CUTTING_DEV_ROOT"
python scripts/run_coupled_collection.py --sliceit-root "$SLICEIT_ROOT" \
  --mode collect --episodes 1 --wall-seconds 600 --output runs/capture
python scripts/run_coupled_collection.py --sliceit-root "$SLICEIT_ROOT" \
  --mode train --model-state control-history --wm-only-updates 500 \
  --max-hours 6 --wall-seconds 22200 --output runs/gazebo_wm
```

SAC uses its source settings. Create `runs/sac/run_config.json` as `{}`
for a fresh run, or include a `resume` path for existing weights/replay.
Its original endpoint is 100100 environment steps.

```bash
mkdir -p runs/sac
printf '{}\n' > runs/sac/run_config.json
python scripts/run_sac_training.py --sliceit-root "$SLICEIT_ROOT" \
  --output runs/sac
```

SAC uses ROS/Gazebo ports 11321/11355; Gazebo WM uses 11322/11356.
Keep current running SliceIt processes on their original checkouts.

## Checks

The robot tests also use a standalone import in `runs/model/`. Generate it
from the same task capture and package roots as the coupled robot:

```bash
.venv/bin/python backends/mujoco/scripts/prepare_robot.py \
  --task backends/mujoco/runs/source/task.json \
  --source-root "$SLICEIT_ROOT/underlay_ws/src" \
  --source-root "$SLICEIT_ROOT/catkin_ws/src" \
  --source-root /opt/ros/noetic/share \
  --output backends/mujoco/runs/model
```

The independent-versus-batched material test uses a scene already in contact.
In the SliceIt Python environment, from `external/SliceIt/disect`, export it
with the existing 65000-step warmup:

```bash
xvfb-run -a python "$CUTTING_DEV_ROOT/backends/mujoco/scripts/export_disect_scene.py" \
  --disect-root . --config ../configs/public_prism.json \
  --warmup-steps 65000 \
  --output "$CUTTING_DEV_ROOT/backends/mujoco/runs/disect_contact"
```

The existing 4090 export at
`/dataHDD/dong/Projects/CuttingGpuBackend/runs/disect_contact` can be copied
to that destination instead. Its manifest records the same warmup and source
revision. Return to the root Python 3.12 environment for the tests.

Run both suites separately to preserve their module resolution:

```bash
CUTTING_BACKEND_ROOT="$PWD/backends/mujoco" PYTHONPATH=. \
  .venv/bin/python -m pytest tests -q
(cd backends/mujoco && PYTHONPATH=. ../../.venv/bin/python -m pytest tests -q)
```

Complete reference cutting uses `scripts/benchmark_gpu_env.py --worlds 1
--wall-seconds 600 --output runs/reference`. Frozen policy evaluation records
task success, force, released springs and timing. Value accuracy additionally
uses complete physical discounted returns; recorded results are in RESULTS.

## Record and render a physical episode

`record_episode.py` uses the benchmark controller factory, contract and evaluation
loop. It saves raw states/actions in `episode_000.npz`, measured world-frame
geometry in `episode.npz`, and the benchmark report. WM–MPPI also saves every
planning call in `planning.json`. The video run uses the selected H16 planner:

```bash
/dataHDD/dong/Envs/cutting-mppi-20261108/bin/python scripts/record_episode.py --method wm-mppi \
  --checkpoint runs/mpc_assets/policy62_wm563_real_prefit100.pt --label wm_mppi \
  --seed 0 --wall-seconds 600 --mpc-horizon 16 --mpc-samples 512 \
  --mpc-refinements 4 --mpc-temperature 0.1 --mpc-noise-std 0.5 \
  --mpc-discount 0.99 --mpc-terminal-value none \
  --output runs/videos_20261108/wm_mppi
```

`--method supermimic` records a frozen actor; `--case N` retains the predefined
perturbation case interface for that method. Rendering runs separately with
VTK, numpy, Pillow, pycollada and ffmpeg (`requirements-render.txt`); the GPU7 environment is
`/dataHDD/dong/Envs/render-20261108`. It needs no physics packages:

```bash
/dataHDD/dong/Envs/render-20261108/bin/python scripts/render_episode.py \
  --recording runs/videos_20261108/wm_mppi --title 'Learned WM + MPPI' \
  --source-root /opt/ros/noetic/share \
  --source-root /dataHDD/dong/Projects/SliceIt/catkin_ws/src \
  --speed 0.1 --fps 30
```

Add `--preview` for three PNGs before encoding. URDF visual meshes and their
material colors attach to the recorded robot body poses; `--robot-geometry
collision` selects the simplified physics meshes. The video shows a whole-arm view,
a cutting close-up and measured force traces. Playback speed is derived from
recorded simulation timestamps. Control-step force peaks and spring releases
are held at their last measured value; geometry alone is interpolated. The full
gray URDF tool includes its handle and mounting adapter; the blue overlay shows
the recorded DiSECt contact blade. Both views preserve their recorded poses,
and the close-up includes the mounting adapter. `--wireframe` shows
material triangles, including DiSECt's interpolated virtual cut vertices and
cut triangles. The board support is display geometry.

## Remote-only development and storage

Open GPU7 in VS Code Remote SSH and use /dataHDD/dong/Projects/Cutting_DEV
(branch main). The Mac holds no project checkout or experiment files.
GPU9 is no longer a run target.

GPU7 storage:

- Code: /dataHDD/dong/Projects/Cutting_DEV.
- Runs: /dataHDD/dong/Experiments/Cutting_DEV/runs; the checkout's runs links here.
- Prepared robot/material assets: /dataHDD/dong/Assets/Cutting_DEV/prepared;
  backends/mujoco/runs links here.
- Planner Python: /dataHDD/dong/Envs/cutting-mppi-20261108/bin/python.
  This environment imports the base GPU packages through its existing
  cutting-base.pth; keep the main checkout's base .venv link.
- Rendering Python: /dataHDD/dong/Envs/render-20261108/bin/python.
- Archived experiment checkouts:
  /dataHDD/dong/Archive/cutting_cleanup_20261109/code.
- Migrated Mac files and uncommitted BladeMaster work:
  /dataHDD/dong/Archive/mac_cutting_20261109/Code.
- GPU9-only code (bundles with all branches, uncommitted patches, untracked files of
  Cutting_DEV_critic, Cutting_DEV_sac, DiSECt, DiSECt-Optimization; the GPU9 tmp scripts,
  including the 2026-11-04 video renderers): /dataHDD/dong/Archive/gpu9_20261110.
  All of that code is superseded by main; no merge needed. GPU9 run data stays on GPU9.

Source the existing runs/sac_env.sh before GPU commands. For MPC commands
above, use the planner Python explicitly. Machine addresses, protected active
paths and the migration inventory are in the server's runs/operations/MACHINES.md.
A6000's running A comparison keeps its frozen checkout, assets and base
environment until its process exits.
