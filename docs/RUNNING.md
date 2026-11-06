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

### New DiSECt material scene

`export_disect_scene.py` preserves the supplied configuration's material step
and simulation duration. For the official apple example, select
`examples/config/ansys_sphere_apple.json` and provide its `dataset/meshes/sphere.rst`.
This asset is a spherical mesh with apple material parameters. Export it to
`backends/mujoco/runs/apple_native` in the SliceIt environment above, then use
the GPU environment to prepare the robot reset and run the reference controller:

```bash
.venv/bin/python backends/mujoco/scripts/prepare_scene_reset.py \
  --capture backends/mujoco/runs/source \
  --scene backends/mujoco/runs/apple_native/scene.pt \
  --output backends/mujoco/runs/apple_capture
.venv/bin/python scripts/benchmark_gpu_env.py --worlds 1 \
  --scene runs/apple_native/scene.pt --capture runs/apple_capture \
  --record-episode --wall-seconds 660 --output runs/apple_reference
```

The reset command derives initial joints and the goal from the exported knife
motion, checks the joint limits and initial blade clearance, and writes
`reset.json`. `--record-episode` saves world 0 physical states (T+1 frames),
rewards, terminal flags and commanded reference poses (T transitions) in
`episode_world0.npz`. The manifest identifies the prescribed reference controller.

## GPU training and evaluation — run 3

```bash
.venv/bin/python scripts/train_gpu.py --worlds 1024 --wm-only-updates 500 \
  --max-hours 24 --output runs/training
.venv/bin/python scripts/evaluate_benchmark.py --method supermimic \
  --checkpoint runs/training/last.pt --output runs/evaluation
```

The maximum eight-step WM fit, 32-step policy rollout, effective batches
2048/4096 and original task rewards are defined in the source configuration.
Training collects new physical transitions, updates WM, then policy/critic.
The first 500 WM rounds leave policy/critic unchanged.

`--resume RUN/last.pt` restores networks, optimizers and training counters.
Physics and replay restart, then the original replay warmup runs.
SIGUSR1 saves after the current iteration. Completion/SIGTERM writes
`last.pt` and `progress.json`. Frozen evaluation performs zero optimizer updates.

The shared fixed-environment evaluation commands are in [BENCHMARK](BENCHMARK.md).
The currently open critic issue is described in [PLAN](PLAN.md).
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

## Manipulation SuperMimic — UR5e apple

Prepare the apple scene and robot reset with the earlier scene commands. Use
an independent upstream checkout and its locked environment:

```bash
export MANIPULATION_ROOT=/dataHDD/dong/Projects/supermimic_manipulation
git clone https://github.com/emoPointer/supermimic_manipulation "$MANIPULATION_ROOT"
git -C "$MANIPULATION_ROOT" checkout eda7da480c6d92a68cc41c9628b7efdae08cae23
(cd "$MANIPULATION_ROOT" && uv sync --locked --extra dev)
export CUDA_VISIBLE_DEVICES=1
export XDG_CACHE_HOME=/dataHDD/dong/.cache
export TMPDIR=/dataHDD/dong/tmp
"$MANIPULATION_ROOT/.venv/bin/python" scripts/train_manipulation.py \
  --source-root "$MANIPULATION_ROOT" --physics-python "$PWD/.venv/bin/python" \
  --worlds 16 --scene runs/apple_native/scene.pt --capture runs/apple_capture \
  --smoke --max-hours 0.2 --output runs/manipulation_smoke
```

Paths for scene/capture are relative to `backends/mujoco`. W&B project is
`cutting-manipulation`; use the account's existing login. Full training omits
`--smoke`: source defaults use 1,000 WM-only iterations followed by joint
updates, ending at iteration 10,000; WM/policy batches are 2048/4096 and the
imagined policy horizon is 16. Set the world count and wall budget before
launching. SIGTERM/SIGINT requests a save at the next completed iteration.
`--resume RUN/model_XXXXXX.pt` restores learner/optimizer/counters/RNG;
physics and replay restart with the source warmup.
`--iterations N` follows the source runner: execute N additional iterations
in this invocation, including after resume.

Frozen evaluation uses the common physical benchmark and saved task contract:

```bash
.venv/bin/python scripts/evaluate_benchmark.py \
  --method manipulation-supermimic \
  --manipulation-python "$MANIPULATION_ROOT/.venv/bin/python" \
  --manipulation-source-root "$MANIPULATION_ROOT" \
  --checkpoint runs/manipulation_smoke/model_000003.pt \
  --contract runs/manipulation_smoke/params/benchmark.json \
  --scene runs/apple_native/scene.pt --capture runs/apple_capture \
  --wall-seconds 1500 --output runs/manipulation_evaluation
```

Results include physical return, force, released springs, termination and
wall time. The full-task reset is enabled; the OmniReset apple bank is pending.

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
