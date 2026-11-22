# Script index

Commands and environments: [Operations](../docs/OPERATIONS.md).
Task and method contracts: [Method](../docs/METHOD.md).
All current methods share main; selecting a backend does not select another branch.

## Current entries

- prepare_shared_scenes.py: generate both backend configs from the current five-food contract.
- train_gpu.py: WM+BPTT; pass the selected variant flags in Operations.
- train_sac_multi.py: batched SB3 SAC.
- evaluate_pose_policy.py: wm_policy, sac, wm_mppi, analytic_mpc.
- check_pose_depth.py / check_board_continue.py: task equations/reset and board continuation.
- diagnose_pose_predictions.py: current V3 WM/critic diagnostic.
- audit_pose_reward.py: DiSECt reference-trajectory reward audit.
- check_training_capacity.py: bounded collection/update/reset/replay capacity.
- verify_entry.py / verify_resume.py: validate saved evidence; do not launch training.
- render_scene.py / compose_wrench_video.py: canonical studio output and sensor curves.
- capture_visual_studio.py / plan_visual_cut.py / generate_visual_center_data.py /
  train_visual_center.py / compare_visual_centers.py / evaluate_visual_cut.py:
  [visual pipeline](../docs/methods/VISION.md), with the documented hardware/backend limits.
- prepare_ur10e_tool.py / compile_robot_platform.py / check_robot_platform.py /
  export_robot_platform.py: CAD and nominal platform preparation/checks.
- check_baseline_adapters.py: recorded CPU-only check, or --live disect|blademaster
  for bounded physical execution; recorded mode is not required for GPU training.
- check_reference_episodes.py: natural termination, selected reset and a subsequent step.

Renderer appearance comes from configs/render/studio_v1.json. No per-video substitute
food textures, fabricated connectors or different physical knives.

## Retained compatibility entries

Old TensorFlow/Gazebo SAC bridges, train_coupled*, train_official_supermimic,
fixed-prism evaluate_benchmark/record_episode, free-cut evaluate_multi and
diagnose_free_cut_predictions, and latch-era check_wm_latch_learning are historical.
They are not launch alternatives for the current task. Keep importable helpers
needed by tests/old checkpoint evaluation; current commands live only in Operations.
Third-party physics differentiation tests are not this project's training prerequisite.
