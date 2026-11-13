# Script entry points

Run commands from the repository root; installation is in [RUNNING](../docs/RUNNING.md).
The current task is [free_cut_v3](../docs/TASK_FREE_CUT.md);
[PLAN](../docs/PLAN.md) records the authorized continuation and branch ownership.

## Current entries

| Purpose | Entry |
|---|---|
| Physical collection, WM/actor/critic updates | `train_gpu.py --conditions <json>` |
| Frozen per-world physical evaluation | `evaluate_multi.py` |
| Same-trajectory prediction/value diagnostics | `diagnose_free_cut_predictions.py` |
| Reference reward and termination checks | `check_free_cut_task.py` |
| Reset and end-to-end cost checks | `check_free_cut_reset.py`, `check_free_cut_throughput.py` |
| One-update execution check | `check_free_cut_learning.py` |

Current configs are listed in [configs/free_cut](../configs/free_cut/README.md).
The training and diagnostic scripts run from separately pinned checkouts; read
PLAN before using a similarly named worktree.

## Historical and comparison entries

| Purpose | Entry |
|---|---|
| Fixed-prism / old grouped WM variants | `train_gpu.py` (defaults are the old task) |
| Group/material checks | `check_multi_condition.py`, `check_parallel_groups.py` |
| Unified fixed-prism SAC | `train_sac_gpu.py`, `sac_gpu_worker.py` |
| Frozen policy/SAC/MPPI/reference benchmark | `evaluate_benchmark.py` |
| Reference cuts / world scaling | `benchmark_gpu_env.py` |
| Record/render physical geometry | `record_episode.py`, `render_episode.py` |

The separately adapted official learner is `train_official_supermimic.py`;
[SOURCES](../docs/SOURCES.md#official-supermimic-adaptation) records its scope and
limitations. It is distinct from the community-source learner above.
Runnable entries do not authorize additional experiments.

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
