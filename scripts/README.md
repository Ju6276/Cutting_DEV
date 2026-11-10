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
force needed to cut each exported object (PLAN step 1). `b1_force_risk.py` is the
completed B1 risk screen.

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
reference scenes. `collect_reliability_data.py`, `wm_reliability_eval.py`,
`wm_branch_check.py`, `wm_fixed_fit.py`, `wm_loss_gradients.py` and `value_eval.py`
remain available for named analysis. Their existence does not schedule a new
diagnostic or reopen a completed experiment; PLAN selects their use.

## Historical tools

`legacy/evaluate_perturbed.py` and `legacy/perturbed_eval.py` preserve the retired
action-noise protocol and paired analysis. `legacy/build_evidence.py` rebuilds
the archived 2026-11-10 package using `configs/archive/evidence_20261110.json`.
These are historical reproduction tools, excluded from the current paper/run
plan. The nominal recording entry no longer imports their perturbation wrapper.
