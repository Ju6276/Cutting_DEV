# Running Cutting DEV

Historical fixed-prism/source commands below are reproduction entries, not
instructions to restart S3. Current long-run settings and frozen launch records
are in [TRAINING_SHARED_20261116](TRAINING_SHARED_20261116.md). The versioned opt-in free-cut contract is in [TASK_FREE_CUT](TASK_FREE_CUT.md); follow [PLAN](PLAN.md).
Commands run from the repository root. The GPU backend is bundled at
`backends/mujoco`; `--backend-root` can select a separate prepared checkout.


**Physics version:** main now uses `edge_coordinate_consistent_v2` (contact-coordinate
bug correction; see METHOD). Historical checkpoint reproduction requires the code/configuration recorded
in that run's manifest and an isolated kernel cache. `38b85a5` is the immediate
pre-coordinate-fix reference, not the exact revision of every historical run. Do not remove metadata
checks to reuse old checkpoints as identical-physics results. No training restart
is authorized by this correction.

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

## Unified GPU training and evaluation

The commands for the historical working baseline (resumed from policy62/WM562 with
critic reset and 100 pre-fit rounds) and for the `source` configuration (script
defaults, from scratch) are in
[README](../README.md#run-the-existing-fixed-prism-task); the two must not be confused. Without variant flags, `train_gpu.py` runs `source`: imagined critic
targets, 32-step policy rollout, no BC (`--critic-target imagined`, `--actor-horizon`
unset = 32, `--actor-bc-alpha` unset = off). Variant flags are documented in
[METHOD](METHOD.md).

The maximum eight-step WM fit, effective batches 2048/4096 and original task
rewards are defined in the source configuration.
Training collects new physical transitions, updates WM, then policy/critic.
The first 500 WM rounds leave policy/critic unchanged.

`--resume RUN/last.pt` restores networks, optimizers and training counters.
Physics restarts. By default replay loads from colocated `replay.pt` only when
checkpoint name/iteration match; open replay episodes are closed at restart.
Named checkpoints from the 2026-11-14 restart retain a matching <stem>.replay.pt; last.pt uses replay.pt. Do not separate weights from their paired replay when resuming.

With a restored replay, version-2 schedule state preserves the warmup origin and fractional update phase; no second warmup is inserted. Deliberately empty replay starts a refill warmup. Legacy schedule snapshots emit a migration warning. Use `--resume-without-replay` for old checkpoints
without matching snapshots (the historical README command). One `replay.pt`
is overwritten at every save: older checkpoints can be evaluated but cannot all
resume with the latest replay. Empty-replay resume is not exact continuation.
SIGUSR1 saves after the current iteration. Completion/SIGTERM writes
`last.pt` and `progress.json`. Frozen evaluation performs zero optimizer updates.

The shared fixed-environment evaluation commands are in [BENCHMARK](BENCHMARK.md).
The current task list is in [PLAN](PLAN.md).
The absorbing-terminal repair permits older checkpoints in frozen evaluation only
when the environment/physics metadata also match. Training additionally requires
compatible terminal-rule metadata. This does not override the contact-revision
requirement above.

## Independent Gazebo reproduction

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

Run both suites separately to preserve their module resolution. On the prepared
GPU7 checkout, first source runs/sac_env.sh; this sets CUDA_HOME and
TORCH_EXTENSIONS_DIR as well as the source/asset paths. An empty shell does not
have a working DiSECt kernel environment.

For full optional-MPPI coverage on GPU7, use
`/dataHDD/dong/Envs/cutting-mppi-20261108/bin/python` for the root suite;
the backend suite uses the base environment as below.

```bash
source runs/sac_env.sh
CUTTING_BACKEND_ROOT="$PWD/backends/mujoco" PYTHONPATH=.:backends/mujoco \
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
planning call in `planning.json`. The example records the existing working
checkpoint with the selected H16 planner.
The policy and planner comparison must load that same checkpoint; it is not a
new multi-condition result:

```bash
POLICY_CHECKPOINT=runs/sm_ckpts/abl_a250_last.pt
/dataHDD/dong/Envs/cutting-mppi-20261108/bin/python scripts/record_episode.py --method wm-mppi \
  --checkpoint "$POLICY_CHECKPOINT" --paired-checkpoint "$POLICY_CHECKPOINT" --label wm_mppi \
  --seed 0 --wall-seconds 600 --mpc-horizon 16 --mpc-samples 512 \
  --mpc-refinements 4 --mpc-temperature 0.1 --mpc-noise-std 0.5 \
  --mpc-discount 0.99 --mpc-terminal-value none \
  --output runs/videos/wm_mppi
```

`--method supermimic` records a frozen actor. Recording now follows the nominal
benchmark only; the retired perturbation option is removed. Choose a new output
directory for each recording. Rendering runs separately with
VTK, numpy, Pillow, pycollada and ffmpeg (`requirements-render.txt`); the GPU7 environment is
`/dataHDD/dong/Envs/render-20261108`. It needs no physics packages:

```bash
/dataHDD/dong/Envs/render-20261108/bin/python scripts/render_episode.py \
  --recording runs/videos/wm_mppi --title 'Learned WM + MPPI' \
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

Open GPU7 in VS Code Remote SSH and use /dataHDD/dong/Projects/Cutting_DEV_board
(branch codex/board-cut-v4). The isolated A6000 training checkout and launch status are in [TRAINING_20261114](archive/TRAINING_20261114.md). The Mac holds no project checkout or experiment files.
GPU9 was reauthorized on2026-11-15 for BladeMaster; access it through tams61.
Historical checkouts remain untouched; see PLAN for its isolated deployment.

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
Machine notes are timestamped observations, not proof that a job is still active.
A6000 keeps its existing checkouts and dependencies; inspect current process
ownership before changing them. GPU7 cleanup details and retained assets are in
[DATA](DATA.md).


Bounded review only, on GPU7 after sourcing runs/sac_env.sh:
~~~bash
PYTHONPATH=. .venv/bin/python scripts/check_free_cut_task.py --config configs/free_cut/review_v2.json --output runs/NEW_REFERENCE_CHECK --wall-seconds 2700
PYTHONPATH=. .venv/bin/python scripts/check_free_cut_reset.py --output runs/NEW_RESET_CHECK --wall-seconds 1200
PYTHONPATH=. .venv/bin/python scripts/check_free_cut_learning.py --output runs/NEW_UPDATE_CHECK --wall-seconds 600
~~~
Use fresh output directories. The final command is one gradient-update execution check with
a short test deadline and small batches, not a recommended training setup. No policy efficacy
can be inferred from it. Review details: [FREE_CUT_REPAIR](archive/FREE_CUT_REPAIR_20261113.md).


Second-review checks (fresh output directories; GPU7, source runs/sac_env.sh):
~~~bash
PYTHONPATH=. .venv/bin/python scripts/check_free_cut_task.py --config configs/free_cut/review_v3.json --output runs/NEW_V3_REFERENCES --wall-seconds 2100
PYTHONPATH=. .venv/bin/python scripts/check_free_cut_throughput.py --worlds 256 --iterations 32 --staggered-start --full-update-profile --output runs/NEW_V3_TIMING --wall-seconds 1200
~~~
The throughput script executes real controller transitions and gradient updates, with an
explicit artificial 16-step deadline and optionally staggered initial remaining deadlines
for reset stress. Initial updates use small batches; the optional final profile uses source
batch counts (WM 10x2048, actor 4096, real critic 5 epochs x4 minibatches), horizon 8,
microbatch 64, full networks and eager execution. It is not a convergence experiment, does
not produce a trained policy, and does not authorize long training. CUDA compilation memory
and different-object worker processes are not bounded by this one-object eager profile.
The source training logger now separates time/reset from time/physics.

## Online W&B for long training (required since 2026-11-15)

Every long training run uses --wandb-mode online. Both training entry points
already default to online; do not override that default with offline/disabled.
ExperimentTracker propagates initialization errors rather than falling back.
Before launch, set NETRC=/dataHDD/dong/.config/wandb/netrc on GPU7 or A6000.
Check authentication without printing credentials. Verify the resulting run URL
and cloud history before leaving the run unattended. Fix login/connectivity
failure before training; local-only logging is not an acceptable fallback.

Local server checkpoints and logs remain recovery copies. Historical commands
with --wandb-mode offline describe previous runs and must not be copied into a
new long-training launch.


For mixed Torch/Warp workloads, --cuda-cache-reserve-gib 3 can release unused
Torch cache before physics when less than3GiB is free. It is opt-in and is
recorded in the training manifest. It does not reduce the number of worlds or
effective learning batches. Confirm its free-memory and physics/reset behavior
in a full local preflight before relying on it.

## Frozen multi-object pose evaluation (2026-11-15)

Use scripts/evaluate_pose_policy.py for pose_depth_v2 and historical free_cut_pose_v1 / blademaster_pose_v1.
It executes one predeclared first episode per world without exploration or learning,
writes cases, checkpoint/config hashes, raw per-step trajectories and per-object
summaries, and closes the environment on success/error. Wall-budget/incomplete
runs are not complete evaluations. Configured task, limits and observation interface
must match the checkpoint. Native BM component count is not labeled intended-slice
separation. The historical evaluate_multi.py rejects these pose tasks.

Example (development conditions, not a sealed final test set):

```bash
.venv/bin/python scripts/evaluate_pose_policy.py   --checkpoint RUN/training/last.pt   --conditions configs/free_cut/pose_three_object_wrench6_3072.json   --worlds-per-object 16 --seed 20261115 --max-hours 1 --output RUN/evaluation
```

For BM use --blademaster-config configs/blademaster/three_intact_batched256.json
instead of --conditions. This config is nominal, so repeated worlds are
replicas rather than distinct randomized material/geometry cases.
CUTTING_EVAL_ROOT can point to a frozen source checkout when running the audited
evaluation script from an external experiment directory; the manifest records
both the physical-source Git revision and evaluation-script SHA256.

Frozen evaluation videos use the same studio appearance as the November 14
stills, with actual physical blade geometry:

```bash
/dataHDD/dong/Envs/render-20261108/bin/python scripts/export_evaluation_scene.py   --evaluation RUN --group 0 --backend DiSECt   --source-root /opt/ros/noetic/share   --source-root /dataHDD/dong/Projects/SliceIt/catkin_ws/src   --source-root /dataHDD/dong/Projects/SliceIt/underlay_ws/src   --output RUN/studio_scene.npz
blender -b -t 12 --python-exit-code 1 -P scripts/render_episode_still.py --   --scene RUN/studio_scene.npz --output RUN/video.mp4 --video --preview --samples 16 --fps 10 --speed 1
# Remove --preview to encode the complete clip; --view wide gives the robot view.
```

Use `--appearance-reference DIR` only for a verified identical FEM topology
and matching reference geometry. Use `--color-reference DIR` for approximate
scan color transfer to another representation; this changes no physical geometry.
The reference directory contains episode.npz and its hash-verified appearance.npz.
Both flags are optional: geometric blocks use the neutral material palette.
BladeMaster uses `--backend BladeMaster` and recorded physical blade poses.
Only the recorded original handle and physical blade are displayed; no fabricated
connector is added. A measured adapter mesh remains unavailable.
All outputs describe a frozen-policy rerun, not footage of the original training.

The removed `render_pose_evaluation.py` is not a supported video entry. Preserve
recordings and evaluation results; superseded visual previews can be deleted.

### Current geometric-depth and material-motion verification

Use the shared preparation and check commands in [SHARED_SCENE](SHARED_SCENE.md).
Add `--tool-reward-audit` to check_pose_depth.py to compare both robot-position
modes on measured transitions by varying goal XY offsets. It verifies reward and
termination equations; it is not a natural large-motion rollout. The historical
--motion-reward-audit exercises the retired particle-reference metric only.
Superseded separate scene-check inputs are under configs/archive/scene_checks_20261116.

## Shared randomized scene inputs (2026-11-16)

Generate both input files with prepare_shared_scenes.py from configs/shared/scene_v1.json.
The generated DiSECt conditions and BladeMaster config use one episode sampler.
Changing world count requires regenerating paired clearance cases rather than
editing only the world count in generated JSON. Training resumes require the
saved episode_conditions sampler state when this protocol is selected.
Full learner checks accept check_training_capacity.py --backend disect or
--backend blademaster with the matching generated --config.

Frozen shared-scene evaluations regenerate clearance arrays and episode-condition
seeds from --seed and --worlds-per-object for both backends; the resolved cases
are written into the manifest. Reports include object-motion failures and terminal
material displacement. The shared scene screenshot entry requires --shared-contract
and --prepared and does not override material, gap or target speed.

### Optional tool-position modes: physical rule demonstration

Use record_tool_motion_demo.py --shared-contract configs/shared/scene_v1.json
--prepared PREPARED/manifest.json --output NEW_RUN/recording --seconds 900.
It selects the first paired cucumber case, runs a scripted lateral excursion and
return with the final-only task, and computes the optional path-rule settlement
on the same measured states. This is not a policy comparison or a braking model.
Export NEW_RUN/recording with export_evaluation_scene.py (group 0 / DiSECt), render
with render_episode_still.py --video --fps 10 --speed .5, then call
compose_tool_motion_demo.py --recording NEW_RUN/recording --render NEW_RUN/studio.mp4
--output NEW_RUN/tool_position_modes.mp4. The latter uses the rendered frame
directory and actual task JSON for thresholds; no physical coordinates change.


## Current controller and a future UR10e port (2026-11-16)

The active simulation uses `FDCCController` in
`backends/mujoco/cutting_gpu_backend/controller/fdcc.py`: a ROS-free tensor port
of the SliceIt FZI position controller, pinned to the cambel fork at
`f4359b966bb3b8d8a07d829df463da814c2bee33`. Its input combines pose error,
sensor wrench and configured compliance/gains; it returns joint-position
commands, followed by source-style position/velocity saturation. The robot
substep is 1 ms, material/force exchange is 2 ms and policy action interval is
50 ms. Its virtual solver integration step is a separate algorithm parameter.
This is not UR's internal torque controller or an untouched ROS2 node.

Open-source upstream:
[FZI Cartesian Controllers](https://github.com/fzi-forschungszentrum-informatik/cartesian_controllers)
and its [FDCC compliance controller](https://github.com/fzi-forschungszentrum-informatik/cartesian_controllers/blob/ros2/cartesian_compliance_controller/README.md).
The upstream describes an admittance-based combination of motion and force
tracking and includes UR10e applications. Real UR integration can use the
[official UR ROS2 driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver),
but installing a driver does not complete this project's port.

The current project is still UR5e-specific: the compiled MuJoCo robot,
source task/initial-joint/controller snapshots, `b_bot_knife_sim` chain,
canonical shared-knife mount, workspace/goal reachability and some visual
mesh/link selectors depend on that asset. There is no verified `ur_type=ur10e`
launch switch. A UR10e port must regenerate these assets and transforms and
validate joint ordering, limits, force-frame/payload compensation and control
response. Same six joint dimensions do not make the UR5e-trained WM or policy
robot-independent; use new UR10e data and validate/retrain before transfer claims.
No UR10e deployment or hardware test was performed in this audit.

Current overview recordings include `knife_world` for every world plus physical
triangles and explicit robot-base/world transforms. The overview and studio
export share `physical_robot_actors`; neither draws the obsolete full URDF knife
or a fabricated connector. Old snapshots without per-world blade data are
rejected by the overview renderer rather than silently substituted.


### Pending learning rounds and shutdown status (2026-11-16)

`train_gpu.py` checkpoints include `update_queue` in addition to collection
credits. A vector step can earn several learning rounds; completed-round
milestones save the remaining unstarted rounds. Resume with replay drains those
rounds before collecting another physical step, retaining their LR clocks.
Intentionally resuming without replay discards them with a recorded reason and
uses the existing warmup/refill rule. Legacy files without the queue warn that
unrecorded pending work cannot be reconstructed.

If interruption occurred inside an optimizer round, its saved parameter and
optimizer steps are kept; the whole round is not silently replayed. Its unfinished
remainder is recorded as discarded, with a warning. This recovers unstarted
rounds, not exact mid-minibatch execution or physical simulation state.
Environment shutdown is attempted before W&B finalization, including failure
status if shutdown raises; the outer setup guard avoids a duplicate close attempt.
Native SIGILL can still bypass Python cleanup and must be judged by the external
process exit status. This change does not fix GPU9's unresolved native fault.

Root test setup adds the bundled backend import path independently of collection
order. The explicit PYTHONPATH above also works with the optional MPPI Python
without depending on an editable backend install.


## Sensor force/torque panel below studio video

Keep the shared studio scene unchanged, then append synchronized charts:

~~~bash
PYTHONPATH=. /dataHDD/dong/Envs/cutting-mppi-20261108/bin/python scripts/compose_wrench_video.py   --render RUN/studio/cucumber.mp4 --output RUN/studio/cucumber_wrench.mp4
~~~

Requires the video's JSON sidecar and original evaluate_pose_policy recording,
including trajectories.pt and geometry metadata. This offline Python has Torch,
Pillow and Matplotlib; the lean training environment need not install graphics
packages. The source video is never overwritten or rerendered. The compositor
verifies case/episode identity, continuous first-episode observations, simulation
time, sidecar and encoded frame counts. It uses the same playback sampling and
final-frame hold as the studio renderer.

Signals are signed filtered sensor-frame Fx/Fy/Fz in N and Mx/My/Mz in N m,
recovered from the versioned full-wrench observation with its recorded limits.
These are20 Hz control-step endpoint samples, not the500 Hz force-limit peaks
or the material reaction-force norm. Plots have separate units and a moving time
cursor. Force-limit event rates must still come from the physical evaluation.
The first end-to-end composition used a historical frozen-policy movie only;
it does not validate a newly trained policy or replace that movie's old physics.


## Actual all-world training snapshot

Use the dedicated VTK environment and resolve both SliceIt and ROS mesh packages.
This reads the geometry recorded by the active trainer; display grid offsets do
not change the physical environment.

~~~bash
PYTHONPATH=.:backends/mujoco /dataHDD/dong/Envs/render-20261108/bin/python scripts/render_training_snapshot.py \
  --recording /dataHDD/dong/Projects/Cutting_DEV_shared_long_20261116/runs/shared_long_20261116/disect_training/snapshot_00000001 \
  --source-root /dataHDD/dong/Projects/SliceIt \
  --source-root /opt/ros/noetic/share \
  --output runs/shared_long_20261116/actual_disect_3072.png
~~~

The 3072-world first-step output is an initialization/recording check, not an
example of learned cutting. The optional MPPI Python lacks VTK and should not
be substituted for this command.
