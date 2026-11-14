# Script entry points

Run commands from the repository root; installation is in [RUNNING](../docs/RUNNING.md).
The current task is in [PLAN](../docs/PLAN.md). Training entries remain for
historical reproduction; their availability does not schedule a new run. `train_gpu.py` runs the fixed
prism by default and the multi-object task with `--conditions <json>`; the
force-risk constraint is `--risk-constraint`.

## Retained execution entries (historical task definitions)

| Purpose | Entry |
|---|---|
| WM and policy training (fixed prism or `--conditions`, optional `--risk-constraint`) | `train_gpu.py` |
| Multi-object environment check with reference control | `check_multi_condition.py` |
| Parallel vs sequential object groups | `check_parallel_groups.py` |
| Frozen multi-object evaluation (WM-method checkpoints) | `evaluate_multi.py` |
| Unified-environment SAC | `train_sac_gpu.py`; `sac_gpu_worker.py` is its process bridge |
| Frozen policy, SAC, MPPI and reference evaluation | `evaluate_benchmark.py` |
| Reference cuts / world scaling | `benchmark_gpu_env.py` |
| Record a nominal physical episode | `record_episode.py` |
| Render saved geometry | `render_episode.py` |

The separately adapted official implementation is `train_official_supermimic.py`;
see [its mapping and limitations](../docs/OFFICIAL_SUPERMIMIC.md). Its first cutting
run is not a successful reproduction result. Do not confuse it with the
SuperMimic-Reproduced `source` configuration in `train_gpu.py`.

## Assets and independent source reproduction

Scanned objects: `prepare_food_mesh.py` (scan surface -> tetrahedral mesh),
`make_food_configs.py` (DiSECt scene configs with official materials), then
`backends/mujoco/scripts/export_disect_scene.py`. `measure_cut_force.py` measures the
force-proxy response under the specified hold/trajectory, not minimum real cutting force.

`prepare_backend.py`, `checkout_sliceit.py` and `export_disect_trajectory.py`
prepare assets or reference trajectories. Backend-specific exporters and robot
preparation remain in `backends/mujoco/scripts`.

Gazebo/ROS entry points are retained for the independent SliceIt reproduction
and data interfaces: `run_coupled_collection.py`, `run_sac_training.py` and their
workers (`collect_coupled.py`, `coupled_bridge.py`, `train_coupled*.py`,
`train_sac.py`). They are not the current multi-condition training track.
`guard_training_exit.py` and `finish_online_run.py` support these bounded jobs.
`collect_disect.py` records material-only dynamics.

## Targeted analysis, not a queue to rerun

`check_*.py` validates recorded interfaces, restart consistency, materials or
reference scenes. The retired diagnostics (`scripts/legacy/`: reliability, branch check, fixed fit, loss gradients, value evaluation, B1)
remain available for named analysis. Their existence does not schedule a new
diagnostic or reopen a completed experiment; PLAN selects their use.

## Historical tools

`legacy/evaluate_perturbed.py` and `legacy/perturbed_eval.py` preserve the retired
action-noise protocol and paired analysis. `legacy/build_evidence.py` rebuilds
the archived 2026-11-10 package using `configs/archive/evidence_20261110.json`.
These are historical reproduction tools, excluded from the current paper/run
plan. The nominal recording entry no longer imports their perturbation wrapper.

## Task diagnosis

`diagnose_rollout.py` runs the reference trajectory or a checkpoint's deterministic policy on a
multi-condition config and records per-step pose errors, released cut springs, limit ratio and
knife speed, the ending and the workspace dimension crossed. `check_material_stability.py` scans
material scales per object for nonfinite DiSECt states (optionally with another substep). `measure_sawing.py` compares a pure press with
press plus lateral sawing on exported scenes. Current net-force norms use
simultaneous 2 ms samples (500 Hz), not per-axis temporal maxima. Early reports
used component-peak envelopes and are historical. Results depend on the feedback
mode, friction setting and contact revision recorded in each report. The 98%
spring-release marker is a diagnostic, not approved task completion; these samples
do not resolve every material-substep peak.

`record_reference_cut.py` records the geometry of a prescribed press or press-plus-sawing cut
(no policy, force limit disabled) in the format `render_episode.py` reads; illustration only.

`check_free_object_scene.py` is the platform check of the free-object single-cut task on one
scene: mesh connectivity across the cut plane, rest on the board, press/hold/retract, severed state.


- check_free_cut_task.py: bounded physical reference/reward checks for the opt-in
  versioned free-cut task; requires an explicit config and writes complete episode rows.
  No policy training. See docs/archive/FREE_CUT_REVIEW_20261112.md.


- check_free_cut_reset.py: bounded eager vs packed/graph stationary reset timing and state comparison.
- check_free_cut_learning.py: actual six-action controller collection followed by one WM and
  actor/critic update; deliberately small verification batches, not a training experiment.

### Same-condition free-cut reference comparison

`compare_free_cut_references.py` compares the first episode of each world with the
saved policy diagnostic. It checks **each** reference's material, target speed and
initial lift by world before collection, and rejects a mismatch rather than
reporting an unpaired comparison. This script decomposes the v3 reward only.

Each reference now saves `<name>_transitions.npz` (state, pre-reset next state,
goal, reward, ending flags, world/time indices and the reference target at the
last coupling sample) and `<name>_conditions.json`. Reference targets bypass
the action mapper; the zero action placeholder must not be treated as a policy
action or fed to the WM as if it generated this trajectory. Reports hash their
input files. Older policy diagnostics lack historical checkpoint hashes; matching
iteration/update counters alone does not prove tensor identity.

`--wall-seconds` defaults to 7200 for the total collection budget, checked between
control steps (not a subprocess watchdog). Failure or budget expiry closes the
environment and does not produce a complete comparison report. Earlier references
already saved before an interruption can remain on disk. Reward/physics/training
are unchanged. Critic differences on reference trajectories are
`V_policy(state) - G_reference`, not on-policy value errors.

The original 4b08d49 summary remains historical: its condition check covered only
the final (sawing) reference, and it did not retain reference raw transitions.
This repair does not retrospectively claim those missing checks or data.


### Free-object geometry measurements

`check_free_object_scene.py` now writes measurement revision
`knife_pose_geometry_v2`. It transforms the actual knife mesh with the full pose,
uses the asset's lowest local vertices as the edge, and keeps initial-bottom,
current-bottom and world-height references separate. The edge-height summary
uses 50 ms step endpoints; `geometry.npz` retains synchronized 2 ms knife poses,
spring endpoints/stiffness and object bottoms for offline review. This asset
selection is a geometric diagnostic, never a success criterion.

Existing output directories are rejected. `--wall-seconds` (default 1800) is
checked between control steps, not a hard watchdog; use an outer timeout for a
strict process budget. The geometry report does not certify swept cut coverage
or diagnose the damage law from minimum height alone.


### Residual material-connection diagnosis

`check_free_object_scene.py --trace-damage` optionally observes every
`eval_cut_springs` launch after settling. `cut_damage_trace.py` temporarily wraps
the launch dispatcher in this diagnostic process, restores it on exit, reads the
pre-integration spring extension/contact forces, and compares an independent
expected stiffness with the kernel's actual output. It does not write physics
tensors, modify the integrator or enter a training path.

`damage_trace.npz` stores initial/final stiffness, per-phase contact and damage
counts, actual/expected accumulated decrement, maximum per-substep formula
difference, and last-contact/last-damage substep indices. Indices multiplied by the
recorded material dt give time since the post-settle diagnostic began. A contact
count uses the kernel's nonzero knife-force criterion; it is not geometric
intersection. The option adds overhead; use the same outer process timeout as the
platform check. `geometry.npz` and first-separation events remain available.

## Board-boundary diagnostics (not a training task)

evaluate_board_cut.py transfers a frozen v3 policy to the historical v4
coverage/board audit. It now saves initial particle/COM velocities, coverage
masks and minimum particle heights as well as per-step knife gap and outcomes.
Its success is v4 diagnostic path success, not the selected target-pose task
or proof of separation. summarize_board_cut_audit.py pairs the original 64-case
audit. These scripts do not train a WM.

The old check_free_object_scene.py press/hold/retract entry has force-limit
termination disabled and no board stop. Its output is a historical geometry /
damage measurement, not executable safe-task data or a new-policy rollout.
Use TASK_FREE_CUT and PLAN for the selected task, not this script's existence.

## Current pose-task engineering checks

- `check_pose_cut.py`: physical pose/board settlement, real-controller data, one WM/actor/critic update and checkpoint round-trip.
- `prepare_stable_food_pose.py`: gravity-supported pose of an uncut tetrahedral mesh; no remeshing or material tuning. Re-export DiSECt after applying the pose. Requires the food-mesh Python environment.

## Potato preparation comparison

- `prepare_half_food_mesh.py`: cap the upper half of an existing uncut y-up food
  mesh and remesh it with a flat support face; run in the food-mesh environment.
- `compare_potato_preparation.py`: bounded six-condition settling comparison and
  two nominal pose-task reference cuts. No training, material/contact tuning or
  change to success rules. Asset preparation changes mass, height and mesh.
  Results and the current candidate decision are in [RESULTS](../docs/RESULTS.md)
  and [PLAN](../docs/PLAN.md).


## Recorded appearance and stills

- `render_episode.py --material cucumber|potato|prism` selects fallback
  surface colors; default `generic` is neutral. Prepared cut-face highlighting is
  off by default; `--highlight-cut-faces` restores the contrasting diagnostic band. Robot visual meshes are
  preserved. `--surface-map appearance.npz` adds projected scan texture.
- `prepare_scan_appearance.py --recording episode.npz --usd Object.usd
  --texture textures/Scan.jpg --output appearance.npz` runs in the food-mesh
  environment. It records rigid-registration error and file identities. Dense
  display triangles stay on the recorded FEM surface; there is no geometric
  smoothing or force change.
- `render_episode_still.py --export --recording DIR --material cucumber
  --surface-map DIR/appearance.npz --source-root ROS_ROOT --scene DIR/still_scene.npz`
  exports the actual recorded geometry using the rendering Python environment.
  Repeat `--source-root` for needed URDF packages.
- Then `blender -b -t 8 --python-exit-code 1 -P scripts/render_episode_still.py --
  --scene DIR/still_scene.npz --output DIR/textured_still.png` renders a Cycles CPU
  still (plus reproducible .blend and provenance JSON). Tested with GPU7 Blender 2.82
  (legacy vertex-color encoding); use a wall limit when rendering batches.

Visual seams are pre-generated DiSECt cut surfaces, not separation measurements.
Source textures are real scans. With highlighting off, prepared faces inherit
nearby exterior appearance; existing geometric openings are not filled. Board
shading and the optional contrasting cut-face colors are illustrative. These
commands render existing data and never start policy training.


### Single-tool views and actual batch initialization

The mounted URDF knife/adapter is the default visible tool. The blue DiSECt
contact proxy is hidden unless `--show-contact-blade` is explicitly supplied
to either renderer. Hiding the proxy does not change its contact computation
or assert that its geometry matches the appearance mesh. Blender stills accept
`--view close` (default) or `--view wide`.

To inspect a full batch without starting training, run in the training environment
after sourcing `runs/sac_env.sh`:

```bash
PYTHONPATH=. .venv/bin/python scripts/capture_environment_batch.py \
  --config configs/free_cut/pose_cucumber.json --worlds 256 \
  --output runs/batch_preview --wall-seconds 300
```

This loads one object group, samples its material/clearance/target-speed settings
and settles actual worlds. It stores per-world particles, joint states, robot
transforms and material parameters. No policy or optimizer is loaded. The
`world_states.npz` leading dimension is world, not time. Output directories must
be new, and a completed `report.json` is required for rendering. Packed topology
is reduced to its first per-world block for the world-0 display template.

Then, in the VTK rendering environment:

```bash
python scripts/render_environment_batch.py --recording runs/batch_preview \
  --source-root /dataHDD/dong/Projects/SliceIt/catkin_ws/src \
  --source-root /dataHDD/dong/Projects/SliceIt/underlay_ws/src \
  --source-root /opt/ros/noetic/share --output runs/batch_preview/all_worlds.png
```

The grid uses display offsets only. All worlds retain individual simulated
geometry; robot visual mappers are shared to reduce rendering cost. Material
uses a distant fallback color. The image explicitly says NO TRAINING and its
sidecar records all world IDs and the captured-state checksum.
