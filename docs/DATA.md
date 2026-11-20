# Server data and checkpoint index

Location/recovery index, not an authorization to launch. All paths are on the
servers; the Mac keeps only explicitly requested previews/videos. Decisions and
budgets are in [PLAN](PLAN.md). Do not delete checkpoints, paired replay, scans
or frozen sources based on age.

## Code and run entry

- /dataHDD/dong/Projects/Cutting_DEV on every server (GPU7 = tamsGPU7,
  A6000 = tamsGPU5, GPU9 = tamsGPU9 via tams61); confirm `git rev-parse HEAD`.
  SSH aliases: cutting-gpu7, cutting-a6000. Server names do not imply exclusive
  GPU access.
- Frozen checkouts (evidence only; never advance them to synchronize code):
  `Cutting_DEV_shared_long_20261116` on GPU7 and A6000 (5ec9f85, pose_depth_v2
  runs; native BladeMaster /dataHDD/dong/Projects/BladeMaster-Warp-shared-knife-20261116
  at c51ef73); A6000 `Cutting_DEV_pose_train_20261114` (runtime assets/data);
  GPU9 `Cutting_DEV_bm_long_20261115` (931e2b5, SIGILL-interrupted BladeMaster run).
- GitHub archive tags preserve unmerged historical branches
  (`git checkout -b NAME archive/...`).

## Shared dependencies and assets

| Contents | GPU7 location |
|---|---|
| Experiments (repository runs symlink) | /dataHDD/dong/Experiments/Cutting_DEV/runs |
| Prepared robot/material data (backend runs symlink) | /dataHDD/dong/Assets/Cutting_DEV/prepared |
| Planner/SAC environment (all servers) | checkout `.venv-sac` -> /dataHDD/dong/Envs/cutting-sac (scripts/env/make_sac_env.sh) |
| BladeMaster runtime extras (all servers) | /dataHDD/dong/Envs/cutting-bm-extra (scripts/env/make_bm_extra.sh, requirements/blademaster.txt) |
| Renderer environment (all servers) | checkout `.venv-render` -> /dataHDD/dong/Envs/cutting-render (scripts/env/make_render_env.sh, requirements/render.txt) |
| UR10e/ATI/IKEA knife robot bundle | /dataHDD/dong/Assets/Cutting_DEV/robots/ur10e_ati_ikea_midblade_v1 |
| UR10e paired scene configuration | /dataHDD/dong/Assets/Cutting_DEV/ur10e_shared_midblade_20261120/ |

Older environments (cutting-mppi-20261108, render-20261108, A6000
blademaster-extra-20261114) are kept only for the runs that used them.

Current DiSECt scenes under prepared/sweep are free_cucumber_003, free_prism
and free_potato_half_20261114 (1 s settling for all three).

- Source scans: /dataHDD/dong/Research/food_meshes/omniobject3d/<object>/.
  [OmniObject3D](https://omniobject3d.github.io/), CC BY 4.0, downloaded from
  [HF mirror](https://huggingface.co/datasets/AlbertS666/OmniObject3D-objects).
- Tetrahedral meshes and preparation metadata: food_meshes/tet/.
  Half potato: potato_003_half_20261114.msh; config
  configs/free_scenes/potato_003_half.json. This is our derived preparation,
  not an official half-potato scan; the whole mesh is retained.
- Appearance: food_meshes/omniobject3d_textured/ for cucumber_002,
  cucumber_003 and potato_003. Texture/scan geometry is not material calibration.
- Half-potato stability evidence: runs/potato_preparation_20261114 and
  runs/half_potato_randomized_20261114 (144 production reset cases).
- BladeMaster intact assets: /dataHDD/dong/Assets/Cutting_DEV/blademaster_intact_20261114.
- BladeMaster-derived DiSECt candidates (not used in training):
  /dataHDD/dong/Assets/Cutting_DEV/blademaster_disect_20261114;
  prepared/sweep/bm_{banana,cake,tofu,luncheon}_20261114.

## Current verification evidence

Paths relative to /dataHDD/dong/Experiments/Cutting_DEV/ on the named host.

- Board continuation: GPU7 runs/board_normal_20261119/audited/, runs/review_reset_contact_20261120/, runs/review_9ea91b2_20261120/.
- Entry/resume: GPU7 and A6000 runs/entry_check_dbdddf6_20261119/; validators runs/review_a1852dc_20261119/.
- DiSECt reference reward audit: A6000 runs/audit_pose_reward_362d15e_20261119/.
- GPU memory headroom: GPU7 runs/mem_headroom_20261117/.
- SliceIt rule audit: GPU7 runs/sliceit_task_rule_audit_20261119/evidence.json.
- UR10e integration: runs/ur10e_platform_20261120/ (GPU7 and A6000; see the integration handoff).

## Historical runs and checkpoints

These are evaluate-only under their recorded task versions; they are not
initializations for current runs.

- pose_depth_v2 shared runs (2026-11-16/17): frozen checkouts above, under
  runs/shared_long_20261116 (GPU7 disect_training, W&B scr1v5np; A6000 blademaster_training, W&B pvord4sx,
  bm510_preflight). GPU9 SAC: checkout /dataHDD/dong/Projects/Cutting_DEV_sac_long_20261116, output runs/sac_long_20261116 (W&B i00yr0fg).
  Frozen evaluations: GPU7 runs/final_eval_20261117, runs/policy_video_20261117.
- 2026-11-14/15 pose-v1 runs: GPU7 runs/pose_wrench6_3072_20261114/training
  ([W&B k48zc1lh](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/k48zc1lh));
  GPU9 runs/bm_gpu9_20261115/training
  ([W&B aqoyihk2](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/aqoyihk2);
  SIGILL record exit_investigation_20261115.json). Frozen evaluations under
  runs/frozen_pose_eval_20261115 on GPU7, A6000 and GPU9.
- Fixed-prism checkpoints (GPU7 experiment root): mpc_assets/policy62_wm562_original.pt
  (imagined critic, H32); sm_ckpts/abl_a250_last.pt (policy390/WM890, real critic,
  H8); mpc_assets/policy62_wm563_real_prefit100.pt (not the WM562 pair; do not
  rename); free_cut_joint_20261113/training/last.pt (old free_cut_v3). Hashes:
  runs/operations/checkpoint_index_20261110.json. A6000 keeps the canonical B1
  replay critic_abl_a250_20261106/training/replay.pt and WM reliability runs.
- GPU9 keeps the original SliceIt Gazebo SAC runs and early CuttingWorldModel runs.

## Vision artifacts (2026-11-17)

- GPU7 runs/visual_center_20261117 (dataset, learning; W&B abn7qktr),
  runs/dp3_center_20261117 (six DP3/local checkpoints, comparison),
  runs/vision_route1_20261117, runs/vision_variants_20261117.
- GPU9 runs/vision_closedloop_20261117, runs/visual_center_closedloop_20261117,
  runs/dp3_center_closedloop_20261117, runs/dp3_controllers_20261117 (physical
  handoffs; studio deliveries).

## Cleanup receipts

Cleanups preserved unique data and deleted only rebuildable caches or disposable
preflight tensors. Receipts resolve old paths:
/dataHDD/dong/Experiments/Cutting_DEV/operations/cleanup_20261115/ (each host),
/dataHDD/dong/Experiments/Cutting_DEV/maintenance/ (2026-11-19, including the
A6000 MPPI-v3 worktree bundle), /dataHDD/dong/Archive/ (older code and Mac
exports).
