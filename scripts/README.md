# Script entry points

Run commands from the repository root; installation is in [RUNNING](../docs/RUNNING.md).
The current task is in [PLAN](../docs/PLAN.md). `train_gpu.py` runs the fixed
prism by default and the multi-object task with `--conditions <json>`; the
force-risk constraint is `--risk-constraint`.

## Current execution

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

## Assets and independent source reproduction

Scanned objects: `prepare_food_mesh.py` (scan surface -> tetrahedral mesh),
`make_food_configs.py` (DiSECt scene configs with official materials), then
`backends/mujoco/scripts/export_disect_scene.py`. `measure_cut_force.py` measures the
force needed to cut each exported object (PLAN step 1).

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
press plus lateral sawing on exported scenes.
