# Server data and checkpoint index — 2026-11-19

This is the current location/recovery index, not an authorization to launch.
All paths are on servers. No project source, weights or datasets were copied
to the Mac during this cleanup. See [PLAN](PLAN.md) for decisions and budgets.

## Current workspace and retention status — 2026-11-19

Use /dataHDD/dong/Projects/Cutting_DEV on main for current code and run
commands on GPU7 and A6000. GPU9 is unreachable via tams61 as of this check;
its synchronization and cleanup are not verified.

The old 11-18 inventory is preserved at
[the recorded revision](https://github.com/Ju6276/Cutting_DEV/blob/a1852dc/docs/DATA.md).
It is not a list of worktrees that still exist.

- GPU7: superseded _board, _pose_long_20261114, _ur10e_docs and temporary
  review/audit checkouts were removed after preserving unique files. The
  _shared_long_20261116 checkout remains frozen with its v2 run data.
- A6000: _board, _pose_scale_20261114 and _mppi_v3 were removed. MPPI's
  two historical commits were not merged into the current controller. Its Git
  bundle, 44 hashed files (152414594 bytes), asset/run records and symlink
  inventory are in
  /dataHDD/dong/Experiments/Cutting_DEV/maintenance/mppi_v3_retirement_20261119T004826Z/.
  _pose_train_20261114 and _shared_long_20261116 remain frozen evidence.
- GitHub retains main; archive tags preserve unmerged historical work.
  A local branch attached to a frozen historical checkout is not a current
  training entry. Never advance those checkouts to synchronize current code.
- GPU7's 20 modified literature notes and new untracked literature archive are
  ongoing user work. They were not committed, reverted or deleted by this review.

Cleanup receipts:
 /dataHDD/dong/Experiments/Cutting_DEV/maintenance/cleanup_20261119_tamsGPU7/manifest.json
and cleanup_20261119_tamsGPU5/manifest.json on their respective servers.
Only rebuildable Python caches were deleted. Superseded/aborted 11-19 audit
outputs moved out of GPU7's active runs directory into that maintenance archive;
A6000's old backends/mujoco/runs.pre_unify_20261118 asset backup moved there too.
Moves preserve unique data. Receipts resolve the current location of old paths.

Current verification evidence:
- GPU7: runs/entry_check_dbdddf6_20261119/ (DiSECt learning/resume and planners).
- A6000: the same directory name (BladeMaster checks, plus DiSECt policy/SAC
  evaluation and diagnostics moved from GPU7 after resource contention).
- A6000: runs/audit_pose_reward_362d15e_20261119/ (complete DiSECt 30-case audit).
- Both hosts: runs/review_a1852dc_20261119/ (independent content rechecks;
  GPU7 also holds before/after negative reproductions and full suite logs).

Historical free_cut, pose-v1/v2, fixed-prism and visual records are retained
under their original task/version. They are not current v3 learning evidence.
Do not delete checkpoints, paired replay, scans or frozen sources based on age.
The full earlier worktree cleanup manifests remain under maintenance/.

## Current shared-task deployments

Both 2026-11-16 runs (pose_depth_v2, now read/evaluate-only) ended on 2026-11-17
(PLAN); these checkouts are frozen evidence, not running jobs.
GPU7 and A6000 each use an isolated checkout at
/dataHDD/dong/Projects/Cutting_DEV_shared_long_20261116.
Their checked source is5ec9f85; native material source is
/dataHDD/dong/Projects/BladeMaster-Warp-shared-knife-20261116 atc51ef73.
Robot model, source task/reset, FDCC parameters and three intact geometry assets
match byte-for-byte. Authoritative status/URLs are in [TRAINING_SHARED_20261116](archive/TRAINING_SHARED_20261116.md);
do not infer a launched run merely from the checkout's existence.

Run files under each checkout: runs/shared_long_20261116.
GPU7:disect_training, launch_plan.json, supervisor.json, training.log.
A6000:bm510_preflight/report.json (passed), launch_gate.json,
blademaster_training, supervisor.json and training.log. The gate launched the
long run only after the complete learning/reset/restore probe passed.
All runs use online W&B; physical snapshots and recovery data stay on servers.

## Historical checkouts and training runs

| Host | Checkout | Role / frozen code |
|---|---|---|
| GPU7 (tamsGPU7), GPU9 (tamsGPU9), A6000 (tamsGPU5) | /dataHDD/dong/Projects/Cutting_DEV | **Current development and run entry on every server, branch `main`** (unified 2026-11-18). Confirm `git rev-parse HEAD` before a run |
| GPU7 | Former Cutting_DEV_board | Removed; source preserved in Git, never a current run entry |
| GPU7 | Former Cutting_DEV_pose_long_20261114 | Checkout removed; 02203b1 and shared Experiments run data retained |
| GPU9 (tamsGPU9) | /dataHDD/dong/Projects/Cutting_DEV_bm_long_20261115 | SIGILL-interrupted BladeMaster run, 931e2b5 |
| A6000 (tamsGPU5) | Former Cutting_DEV_pose_scale_20261114 | Removed development mirror; its linked training data was not deleted |

GPU9 is reached through tams61. GPU7 and A6000 local SSH aliases are
cutting-gpu7 and cutting-a6000. Server names do not imply exclusive GPU access.
Active runs stay on their frozen checkouts; syncing development docs does not
replace their loaded code.

The two historical 2026-11-14/15 training outputs are under
/dataHDD/dong/Experiments/Cutting_DEV/runs (not the current shared runs):

- GPU7: pose_wrench6_3072_20261114/training, 3072 worlds (1024/object).
  Online history forwarding resumes the existing
  [W&B run](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/k48zc1lh);
  forward_cloud.py and cloud_sync.json are in the parent directory.
- GPU9: bm_gpu9_20261115/training, 256 worlds (86/85/85).
  Native online [W&B run](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/aqoyihk2).
- Each is bounded by 24 hours or 1,280,000 new physical transitions.
  Their checkpoints and paired replay files are protected.
- Earlier A6000 192-world and GPU7 three-world runs are stopped historical
  launches; their final recovery files are retained. They are not active jobs.

## Shared dependencies and assets

| Contents | GPU7 location |
|---|---|
| Experiments (repository runs symlink) | /dataHDD/dong/Experiments/Cutting_DEV/runs |
| Prepared robot/material data (backend runs symlink) | /dataHDD/dong/Assets/Cutting_DEV/prepared |
| Main Python environment | /dataHDD/dong/Projects/CuttingGpuBackend/.venv |
| Planner/SAC environment (all servers) | checkout `.venv-sac` -> /dataHDD/dong/Envs/cutting-sac (scripts/env/make_sac_env.sh); GPU7 cutting-mppi-20261108 kept for historical runs |
| BladeMaster runtime extras (all servers) | /dataHDD/dong/Envs/cutting-bm-extra (scripts/env/make_bm_extra.sh, requirements-blademaster-runtime.txt); A6000 blademaster-extra-20261114 kept for the 2026-11-16 run |
| Renderer environment (all servers) | checkout `.venv-render` -> /dataHDD/dong/Envs/cutting-render (scripts/env/make_render_env.sh, requirements-render-lock.txt); GPU7 render-20261108 kept for historical runs |
| Earlier MPPI development | /dataHDD/dong/Experiments/Cutting_DEV/mpc_development_20261108 |
| Earlier metrics | /dataHDD/dong/Experiments/Cutting_DEV/legacy_metrics |

Base Cutting_DEV directories hold Git worktree metadata and/or runtime links.
A6000 Cutting_DEV_pose_train_20261114 retains actual runtime assets/data.
The old MPPI assets/data have moved to the maintenance archive listed above.
SliceIt/ROS sources and shared environments remain dependencies.

Current DiSECt scenes under prepared/sweep are free_cucumber_003, free_prism
and **free_potato_half_20261114**. The completed pose-v1 run used 1 s of per-material
settling for half potato and 0.75 s for cucumber/prism. The current shared task
uses 1 s for all three. Whole-potato exports
free_potato_003 and free_potato_stable_20261114_v3 are historical preparation
comparisons, not the active potato asset.

- Source scans: /dataHDD/dong/Research/food_meshes/omniobject3d/<object>/.
  [OmniObject3D](https://omniobject3d.github.io/), CC BY 4.0, downloaded from
  [HF mirror](https://huggingface.co/datasets/AlbertS666/OmniObject3D-objects).
- Tetrahedral meshes and preparation metadata: food_meshes/tet/.
  Half potato: potato_003_half_20261114.msh; config
  configs/free_scenes/potato_003_half.json. This is our derived preparation,
  not an official half-potato scan; original whole mesh is retained.
- Appearance: food_meshes/omniobject3d_textured/ for cucumber_002,
  cucumber_003 and potato_003. Texture/scan geometry is not material calibration.
- Stability evidence: runs/potato_preparation_20261114 and
  runs/half_potato_randomized_20261114 (144 production reset cases).
- Reward evidence: runs/pose_reward_audit_20261114, including raw trajectories.
- BladeMaster intact assets: /dataHDD/dong/Assets/Cutting_DEV/blademaster_intact_20261114.
  Historical external solver: /dataHDD/dong/Projects/BladeMaster-Warp-batched-20261114,
  e2e52216c23b7f71af5bc46566818c48b975c398; copied with hashes to GPU9.
- BladeMaster-derived DiSECt candidates: /dataHDD/dong/Assets/Cutting_DEV/blademaster_disect_20261114;
  prepared/sweep/bm_{banana,cake,tofu,luncheon}_20261114.
  Evidence: runs/bm_disect_assets_20261114 and runs/ground_contact_20261114.
  These are not added to main training. See [asset provenance](archive/BLADEMASTER_DISECT_ASSETS.md).

## Historical checkpoints and research evidence

GPU7 paths below are relative to its shared experiment root. They are not
initializations for the fresh shared-task runs.

| Path | Identity |
|---|---|
| mpc_assets/policy62_wm562_original.pt | policy62 / WM562, imagined critic, H32 |
| sm_ckpts/abl_a250_last.pt | policy390 / WM890, real critic, H8 |
| mpc_assets/policy62_wm563_real_prefit100.pt | policy62 / WM563, real-critic prefit; not the WM562 pair |
| free_cut_joint_20261113/training/last.pt | iteration2691 / policy543, old v3 separation task |

The misleading policy62_wm562.pt filename was previously renamed to explicit
WM563. Do not recreate it or bypass exact-tensor pairing checks.
Older checkpoint hashes: runs/operations/checkpoint_index_20261110.json.

Preserved on A6000 under /dataHDD/dong/Projects/Cutting_DEV/runs:
critic_abl_a250_20261106/training/replay.pt (canonical B1 fit replay),
wm_reliability_supplement_20261108, wm_reliability_20261108,
wm_route_compare_20261108, a_pair_base_20261108, a_pair_head_20261108,
a_pair_reference_20261108 and a_pair_compare_20261109.
The B1 "final" split was already used on 2026-11-10; it is not a fresh test set.
GPU7 runs/route_b1_20261110 retains fitted models and reports.

Fixed-task MPPI/SAC evidence, main_compare_20261109, retired perturbation
reports and task_diagnosis_20261112 remain for audit. Perturbation results
remain excluded from the paper. Videos retain raw geometry NPZ, reports and
MP4; old regenerated frame caches were previously removed.

GPU9 retains historical SliceIt/runs (Gazebo SAC, last recorded 57722 steps,
ckpt-13) and CuttingWorldModel/runs (early 2026-11-02/03 experiments).
GPU7 /dataHDD/dong/Archive/gpu9_20261110 retains earlier code bundles/patches.
For dated historical locations/deletions, consult Git history and RESULTS;
old folder names do not make their task definitions current.

## Cleanup completed 2026-11-15

Each host has an exact operation ledger at:
 /dataHDD/dong/Experiments/Cutting_DEV/operations/cleanup_20261115/

- inventory_before.json and dependency_audit.json record the pre-cleanup
  worktrees, local changes, runtime links and process references.
- cleanup_result.json records removed files, byte counts, SHA256 and retained
  duplicate locations.
- worktrees/<name>/ contains source.tgz, staged.patch, unstaged.patch and metadata.
  Restore instructions are in RESTORE.md. Git archive refs preserve old HEADs;
  source-history.bundle on GPU7/A6000 preserves refs. Unmerged branches were not
  deleted. No remaining unsaved code was discarded.
- GPU7: removed 15 inactive worktrees, eight disposable preflight tensors and
  one exact duplicate historical replay; 6,269,467,188 tensor bytes reclaimed.
  The replay original remains on A6000 at the canonical B1 path above; hash
  83fd556980159c5c694a27f2d80f4d9bcd451d73ab4969bf60de6f0296bc3bec.
- A6000: removed two inactive worktrees and 12 disposable preflight tensors;
  1,511,716,284 tensor bytes reclaimed.
- GPU9: removed eight preflight tensors and replaced two byte-identical rolling
  diagnostic copies by relative links to retained numbered checkpoints;
  2,295,040,722 tensor bytes reclaimed.

Removed test tensors came from completed capacity/smoke checks, not current
training. Their reports, configurations, logs and checksum identities remain.
CLEANUP_20261115.json beside affected outputs explains absent tensors. They
are intentionally not backed up; rerun the documented preflight to recreate
test tensors. Unique research data, active paired checkpoints/replay,
live assets/environments and other users' jobs were preserved.

Completed review handoffs and stopped launch notes are in docs/archive.
Six redundant plan/paper drafts were removed from the working tree; immutable
Git links in [the index](README.md) retain their content. Measured results,
failed experiments and unresolved limitations remain in RESULTS.


## Frozen evaluation and exit records — 2026-11-15

GPU7 runs/frozen_pose_eval_20261115/disect_last evaluates final policy1109.
GPU9 runs/frozen_pose_eval_20261115/blademaster_3456 evaluates the latest complete
saved policy866, not the998 updates reached before process exit.
A6000 runs/frozen_pose_eval_20261115/disect_192_ready evaluates earlier policy373
under the same declared cases. The first A6000 attempt failed initialization
because the runtime PATH lacked ninja; that failure is retained separately.
Each completed folder retains manifest/config/checkpoint hashes, cases,
episodes, raw trajectories and summary. A copied193 MB checkpoint on A6000 is
evaluation input, not a new training run; its hash matches the GPU7 source.

GPU9 runs/bm_gpu9_20261115/exit_investigation_20261115.json records the SIGILL.
The original progress.json remains unchanged and stale; launch_status.json
records exit_code=-4. Neither file may be read as evidence of ongoing training.

Frozen evaluations are now complete: GPU7 disect_last (0/48, timeout), A6000 disect_192_ready (0/48, workspace), GPU9 blademaster_3456 (48/48 pose success). GPU7 comparison.json stores all three identities/results; see RESULTS for protocol and limitations. No long run restarted.


## Documentation and branch cleanup — 2026-11-17

Development is fast-forward consolidated onto origin/main from the verified
visual-route branch; no force push or commit rewrite. Active/frozen checkouts
on GPU7, GPU9 and A6000 keep their recorded commits and physical inputs.
Only clean development checkouts are synchronized.

The following remote heads are retired after ancestry checks: every commit is
reachable from the consolidated main. Existing experiment links to these names
must use the commit below when the historical revision matters.

| Retired branch | Preserved commit |
|---|---|
| codex/blademaster-multiobject | e4c9f413811e8085a6eeda256d21fd16fe33dfa1 |
| codex/board-cut-v4 | cfbac161d19b9e77799c1a70851e51bd4ac201a8 |
| codex/current-baselines | cfbac161d19b9e77799c1a70851e51bd4ac201a8 |
| codex/knife-geometry-check | c0d5a76d67860b78f007eb9eae3f8ceedfcf9841 |
| codex/reference-audit | cb699722c81531c4792b7884d8cd109e01e30802 |
| codex/residual-spring-debug | 1c4a217d12609b33fd5cede5e093a08ad456a281 |
| codex/transition-schedule-20261114 | 3738271127e232363e628e1a72579c19c72b6a28 |

2026-11-18 cleanup (user-confirmed): GitHub keeps only `main`. Deleted after
merging into main: codex/free-cut-v1, codex/visual-cut-route1,
codex/bm-gpu9-online-20261115, codex/blademaster-batched (its two commits are
in main as equivalent patches) and claude/docs-reorg-20261118. Branches with
unmerged commits were tagged before deletion; restore with `git checkout -b NAME TAG`:

| Deleted branch | Archive tag | Commit |
|---|---|---|
| codex/docs-cleanup-20261113 | archive/codex-docs-cleanup-20261113 | a254def |
| codex/free-cut-infra | archive/codex-free-cut-infra | b1bcb92 |
| codex/free-cut-mppi | archive/codex-free-cut-mppi | 3635dab |
| feat/apple-cutting-adapter | archive/feat-apple-cutting-adapter | 6a453fa |

Local branches still attached to a worktree are kept until those worktrees are
removed.

Historical batching, scale/launch and framework audits moved into docs/archive
with rebased links. The old README recipes and PAPER narrative remain at
4653d10 in Git instead of being duplicated in the current task entry points.
RESULTS and scientific provenance are retained. No checkpoints, buffers,
physical assets, active jobs or user files are deleted by this cleanup.

Validation note: the first cleanup test invocation omitted runs/sac_env.sh
and failed on missing TORCH_EXTENSIONS_DIR. RUNNING now includes that prerequisite.
Final validation uses the complete prepared CUDA environment; the failed
invocation is not counted as a passing check.

Merged local heads not attached to a worktree are also pruned after ancestry checks. Each server retains its active/development and frozen-worktree branches; their checkout versions are not advanced by pruning.


## Learned visual midpoint pilot — 2026-11-17

GPU7 development checkout runs/visual_center_20261117:
- dataset/:900 RGB/depth cases, dataset.npz (visible clouds and offline targets),
  manifest.json (source hashes, camera, configuration, split/placement identities).
- learning/: center.pt, epochs.jsonl, report.json, test_predictions.npz, wandb.json.
- Online run abn7qktr:120 epochs, validation-selected epoch117, finished.

GPU9 development checkout runs/visual_center_closedloop_20261117:
- fixed-midpoint/ and learned-midpoint/: manifests, perception inputs/plans,
  complete180-step transitions, geometry, episode metrics and studio video export.
- comparison.json: offline full-surface midpoint error and final height/force
  audit; queue.json records normal subprocess completion.
- A copy of comparison.json/queue.json is retained on GPU7.

Both trials use the old frozen policy66 intentionally, for the same handoff
comparison as the earlier route1 trial; they are not results of the latest
long-trained policy. No source/training dataset/weight is copied to the Mac.

## DP3 and policy527 artifacts — 2026-11-17

GPU7 development checkout runs/dp3_center_20261117: six versioned checkpoints,
epoch histories, reports, paired predictions/comparison, W&B verification and
regression logs. Reuses runs/visual_center_20261117/dataset unchanged.
GPU9 development checkout runs/dp3_center_closedloop_20261117: DP3 nominal
frozen-policy66 handoff, original observations/geometry, outcomes and analysis.
GPU7 runs/policy_video_20261117: failed first initialization preserved, successful
evaluation_current and shared studio clips/wrench panels for three objects.
Mac contains only requested MP4/PNG previews in
/Users/judong/Downloads/cutting_policy527_20261117; no source/dataset/checkpoints.

## DP3 controller videos — 2026-11-17

GPU9 development checkout runs/dp3_controllers_20261117:
assets/paired_policy66.pt, wm_mppi/, analytic_mpc/, rl/ studio export of the
existing DP3+RL episode, comparison.json, planning histories and failed-attempt
logs. source_at_trial preserves the modified sources before Git synchronization.
delivery/ contains approved-studio movies with pre-cut visual-selection intro
and synchronized force/torque panels. All physics trajectories/weights stay
on servers; requested videos/previews go to Mac Downloads/dp3_controllers_20261117.

## 2026-11-17 entry-point cleanup

Current PLAN/script guide were condensed; historical instructions remain at
e7b9e69. No datasets, replay, weights, training checkouts or unmerged historical
branches were deleted. Imported historical rendering helpers and task fixtures
are retained with explicit scope. GPU7 validation artifacts:
runs/repository_cleanup_20261117/{root_tests.log,grouped_retry.log,static_checks.json}.
The first two preserve the GPU7 co-resident-training OOM, not passing coverage.
GPU9 final suite logs use runs/repository_cleanup_20261117/{root_tests.log,backend_tests.log}
in Cutting_DEV_baselines_20261116.
Development mirrors are synchronized separately from frozen training checkouts.

## 2026-11-17 Mac historical-output cleanup

Verified backup on GPU7:
 /dataHDD/dong/Archive/mac_cutting_cleanup_20261117/
The manifest lists 1512 original files (526795759 bytes), checked individually
by SHA-256 before any local deletion. verified.json and cleanup_receipt.json
record verification and the exact removed paths.

Removed from the Mac: October cutting_gpu_20261016 experiment outputs,
five superseded November12–15 preview folders, and two historical CRESSim
helper scripts. Total1491 files /475165167 bytes. All originals remain in
the archive payload with their home-relative paths.
No other project, personal document, literature PDF or reference media was
deleted. November16–17 shared-scene, force-panel, policy-analysis, policy527
and visual/DP3 exports remain at their existing Downloads paths.

The local Literature/Robotic-Cutting README now indexes current server/GitHub
sources and retained previews. REMOTE_GPU_ZH and RESEARCH_PLAN_ZH no longer
provide obsolete local-code launch instructions; original documents were
backed up first. Historical NOTES remains labeled as October evidence.
Local receipt: Literature/Robotic-Cutting/mac_cleanup_20261117.json.
