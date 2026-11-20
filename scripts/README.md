# Script entry points

Run from the repository root. [PLAN](../docs/PLAN.md) selects current work;
[RUNNING](../docs/RUNNING.md) gives environments and commands.
An available script does not authorize another experiment.

All four controllers already coexist on main. Select the controller with
--method and the material backend with --conditions / --blademaster-config;
do not switch to an old controller worktree. The [controller ledger](../docs/methods/CURRENT_BASELINES.md)
distinguishes interface support from tested task-version/backend combinations.

## Current task and training

| Purpose | Entry / protocol |
|---|---|
| Server runtime environment (base / SAC) | `source scripts/env/cutting_env.sh`, `CUTTING_ENV=sac`; create the SAC env with `env/make_sac_env.sh` |
| Shared scene inputs for both material backends | `prepare_shared_scenes.py`; [SHARED_SCENE](../docs/SHARED_SCENE.md) |
| WM+BPTT training | `train_gpu.py`; explicit generated config and the pose_depth_v3 command in [RUNNING](../docs/RUNNING.md#current-wmbptt-training-command-pose_depth_v3); 2026-11-16 [launch record](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/TRAINING_SHARED_20261116.md) |
| Batched SB3 SAC baseline | `train_sac_multi.py` in the SAC env; [CURRENT_BASELINES](../docs/methods/CURRENT_BASELINES.md#commands); 2026-11-16 [SAC protocol](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/TRAINING_SAC_20261116.md) (archived) |
| Frozen RL, SAC, WM+MPPI, analytic MPC | `evaluate_pose_policy.py --method wm_policy|sac|wm_mppi|analytic_mpc` |
| Recorded entry / short-run resume evidence | python -m scripts.verify_entry and python -m scripts.verify_resume; missing stages fail |
| DiSECt reference reward audit | audit_pose_reward.py with explicit current config; all groups, ten trajectories |
| Physical latch/reset, separately normalized WM loss and save/restore | python -m scripts.check_wm_latch_learning --config GENERATED_CONFIG --backend BACKEND (disect or blademaster) --output NEW_DIR; optional --saw; bounded probe, no policy updates |
| Actual collection/update/reset/replay capacity | `check_training_capacity.py` |
| Task/reward equations and optional tool-position rule | `check_pose_depth.py`; [SHARED_SCENE](../docs/SHARED_SCENE.md) |
| pose_depth_v3 WM/terminal-head/critic prediction diagnostic on frozen-policy trajectories | `diagnose_pose_predictions.py`; [RUNNING](../docs/RUNNING.md#pose_depth_v3-wm-prediction-diagnostic) |

Do not launch `train_gpu.py` with bare defaults as the current task: defaults
retain the historical fixed-prism source variant. Current task is pose_depth_v3 (v2 training refused);
material separation is diagnostic. The optional risk variant is not selected.
[CURRENT_BASELINES](../docs/methods/CURRENT_BASELINES.md) records tested backend coverage,
input contracts, SAC provenance and analytic-model limitations.

## Approved visualization

1. `evaluate_pose_policy.py --record-geometry` records actual frozen execution.
2. `export_evaluation_scene.py` exports the physical blade and original robot/
   handle; use the correct backend and source mesh roots.
3. `render_episode_still.py --video` renders the shared studio through Blender.
4. `compose_wrench_video.py` appends synchronized sensor force/torque curves.

No blue blade overlay, fabricated mounting rod or highlighted artificial cracks.
Color transfer is approximate when representation differs.
`render_training_snapshot.py` reads actual recorded all-world training geometry;
`capture_shared_scenes.py` makes shared-config previews.
`record_tool_motion_demo.py` and `compose_tool_motion_demo.py` illustrate the
optional process constraint; current long runs select final-position-only.
[RUNNING](../docs/RUNNING.md) contains exact video commands and signal definitions.

## Pre-cut vision

| Purpose | Entry |
|---|---|
| RGB/depth from shared scene or settled snapshot | `capture_visual_cut_camera.py`, `capture_visual_studio.py` |
| Geometric or learned midpoint to calibrated goal | `plan_visual_cut.py` |
| Ideal-depth training data | `generate_visual_center_data.py` |
| DP3 XYZ encoder + midpoint head (default); local comparator | `train_visual_center.py --architecture dp3_xyz_v1|local_v1` |
| Matched-case encoder comparison | `compare_visual_centers.py` |
| Native BladeMaster physical visual-goal pilot | `evaluate_visual_cut.py --method wm_policy|wm_mppi|analytic_mpc` |

WM+MPPI requires a full model checkpoint paired by exact actor tensors with
the goal-convention policy checkpoint. Cross-backend inference is explicit
and still checks robot/task/tool contracts. Analytic MPC loads no learned
control weights. SAC visual execution is not implemented.
[Vision protocol](../docs/methods/VISION_ROUTE1.md) specifies scope, flags and budgets.

## Assets and diagnostic tools

`prepare_food_mesh.py`, `make_food_configs.py` and backend exporters prepare
meshes/material scenes. Parameters are candidate simulation inputs, not
automatically calibrated food properties.
`check_*.py`, `measure_*.py`, `diagnose_*.py`, `analyze_policy_mechanics.py`
and `audit_pose_reward.py` serve specific recorded questions, not a queue
that must be rerun before every training launch. Consult their explicit task
versions and [RESULTS](../docs/RESULTS.md).

## Historical reproduction — not current launch defaults

- `train_sac_gpu.py` / `sac_gpu_worker.py`: original TensorFlow SAC bridge.
  These are not the current batched SB3 learner.
- `evaluate_benchmark.py`, `record_episode.py`: fixed-prism benchmark.
  `render_episode.py` reads the historical recording format and is also
  imported by shared export utilities; it is not the default studio video entry.
- `evaluate_multi.py`: old free-cut task; rejects current pose tasks.
- `compare_free_cut_references.py`: free_cut_v3 reward decomposition, not the pose tasks.
- `diagnose_free_cut_predictions.py`: free_cut_v3 separation diagnostic; refuses pose tasks.
- `train_coupled*.py`, `train_sac.py`, `run_sac_training.py` and Gazebo
  collection bridges retain independent upstream reproduction.
- `train_official_supermimic.py`: separate official adaptation; see
  [OFFICIAL_SUPERMIMIC](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/OFFICIAL_SUPERMIMIC.md).
- `legacy/`: retired perturbation, B1 and model-fit diagnostics.

Historical commands and detailed diagnostic provenance remain in the
[previous script map](https://github.com/Ju6276/Cutting_DEV/blob/e7b9e69/scripts/README.md)
and dated results. Importable historical helpers are retained for compatibility;
no old checkpoint or experiment data is deleted by this cleanup.

## Estimated UR10e/ATI robot assets (2026-11-20)

- prepare_ur10e_tool.py: supplied CAD/USDZ conversion, rigid registration and estimated inertials (requirements/cad.txt, separate environment).
- compile_robot_platform.py: nominal UR10e MuJoCo model and FDCC frame/compensation configuration.
- check_robot_platform.py: same robot-only checker for the legacy UR5e backend root or a prepared UR10e bundle.
- export_robot_platform.py: verified robot poses and original CAD through the common studio renderer.

See [UR10e setup](../docs/hardware/UR10E_FDCC_SETUP_ZH.md) for exact commands and limitations.
Robot checks are not a material-cutting integration or hardware acceptance test.

UR10e adapter verification: check_baseline_adapters.py --evidence <learning-run>
--output <new-dir> checks actual recorded states on CPU; add
--live disect or --live blademaster for bounded physical planner/SAC execution.
check_reference_episodes.py checks natural termination, masked reset and a
subsequent physical step with the configured task horizon.
