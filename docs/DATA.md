# Server data and checkpoint index

Inventory reviewed 2026-11-10. Source lives on GPU7 at
`/dataHDD/dong/Projects/Cutting_DEV`. All paths below are server paths; no project
data or source is stored on the Mac. This is a location/status index, not a run plan.

## Storage

| Contents | GPU7 location |
|---|---|
| Experiment records | `/dataHDD/dong/Experiments/Cutting_DEV/runs` (repository `runs` symlink) |
| Prepared robots/materials | `/dataHDD/dong/Assets/Cutting_DEV/prepared` (backend `runs` symlink) |
| Earlier MPPI development | `/dataHDD/dong/Experiments/Cutting_DEV/mpc_development_20261108` |
| Earlier metrics records | `/dataHDD/dong/Experiments/Cutting_DEV/legacy_metrics` |
| GPU Python dependency | `/dataHDD/dong/Projects/CuttingGpuBackend/.venv` |
| Planner / renderer environments | `/dataHDD/dong/Envs/cutting-mppi-20261108` / `render-20261108` |
| Machine routes and cleanup manifests | `runs/operations/` |

The base environment, SliceIt/ROS sources and prepared assets are live
dependencies, even though their directory names predate this repository.

## Existing checkpoint identities

Paths are relative to the repository. Identities below were read from checkpoint
metadata, not inferred from filenames. SHA256/size records are in
`runs/operations/checkpoint_index_20261110.json`.

| Path | Policy / WM updates | Meaning |
|---|---:|---|
| `runs/mpc_assets/policy62_wm562_original.pt` | 62 / 562 | Original source-chain checkpoint; imagined critic, H32 |
| `runs/sm_ckpts/abl_a250_last.pt` | 390 / 890 | Selected working continuation; real critic, H8, critic prefit 100 |
| `runs/mpc_assets/policy62_wm563_real_prefit100.pt` | 62 / 563 | Historical MPPI development model after real-critic prefit; not the WM562 pair |

The misleading disk filename `policy62_wm562.pt` actually contained WM563.
It was renamed to the explicit WM563 name; bytes and SHA256 are unchanged.
The previous explicit WM563 symlink was replaced by that same physical file.
Do not recreate the ambiguous filename. The empty historical
`runs/critic_mask_validation_20261104/training` path was removed; use the original
WM562 path above for the old resume command. Same-WM comparisons still require
the evaluator's exact tensor pairing check.

These checkpoints are historical/dev resources. New multi-condition learning
in PLAN starts fresh; do not silently initialize it from these weights.

## Retained data and their roles

- Fixed-task references: `mpc_pair_repair_20261109`,
  `mpc_baseline_pair_20261109`, SAC training/frozen evaluations and reference
  reports. Preserve failures as well as successes.
- Retired action-noise records: `eval_perturbed_main_20261109`,
  `eval_fixed_p62_20261109/perturbed_policy62` and their paired summaries.
  Retained for audit, excluded from the paper; see
  [archived protocol](archive/ACTION_PERTURBATIONS.md).
- `paper_evidence_20261110` is a historical folder name. Its perturbation figures
  are excluded from the paper. The reproduction recipe and builder now live
  under `configs/archive` and `scripts/legacy`.
- `videos_20261108` retains measured geometry NPZs, reports and MP4s. Generated
  PNG frame caches for three old videos were deleted; measured data/video
  checksums were checked before and after.
- Historical reliability training/calibration/used-test data recorded in RESULTS remain
  in their existing server locations. This cleanup did not open or move them.

## Multi-object task data (GPU7, 2026-11-11)

| Path | Role |
|---|---|
| `/dataHDD/dong/Research/food_meshes/omniobject3d/<object>/` | OmniObject3D scans (CC BY 4.0, from the HF mirror `AlbertS666/OmniObject3D-objects`): `Object.usd`, converted `surface.obj` |
| `/dataHDD/dong/Research/food_meshes/tet/` | Tetrahedral meshes (`prepare_food_mesh.py`) with their parameter JSON |
| `configs/food_scenes/*.json`, `backends/mujoco/runs/sweep/food_*` | DiSECt scene configs and exported scenes (apple_002, potato_003, cucumber_003, cucumber_002) |
| `runs/food_force_20261111` | Historical force-proxy measurements; interpret with later diagnosis |
| `runs/route_b1_20261110` | B1: five ensemble WMs, quantile head, `frozen.json`, dev and held-out reports |
| `runs/critic_abl_a250_20261106/training`, `runs/wm_reliability_supplement_20261108`, `runs/wm_frozen_fit_20261105` | Copies from the A6000 used by B1 (identical files; originals stay on the A6000) |

Deleted 2026-11-11 (wrong object shapes or superseded): material-swapped sweep
scenes and runs `material_sweep_20261110`, `material_sweep_long_20261110`, smoke runs
`mc_smoke_*`, `multi_condition_check_20261110`, all their videos, and the copies on
the Mac.

## 2026-11-12 cleanup

Kept: `runs/task_diagnosis_20261112/` (per-step traces of the deterministic base policy and the
reference trajectory on `s0_eval_train`; RESULTS 2026-11-12).

Deleted (failed or trained on the defective task definition; conclusions are in RESULTS 2026-11-12):
A6000 `runs/s3_base_20261111`, `runs/s3_risk_20261111`; GPU7 `runs/official_prism_20261111`
(including its unsynced offline W&B logs), temporary diagnosis/curve/evaluation folders under
`/dataHDD/dong/tmp`, and the hold-test scenes `backends/mujoco/runs/sweep/holdtest_*`.
W&B online runs `0xvf8zj7`, `5r1geoka` and their resumed runs remain on the server side.

## Data on other servers

**A6000** (`/dataHDD/dong/Projects/Cutting_DEV/runs`, reachable from GPU7 with
`ssh a6000`; not copied to GPU7):

| Path | Size | Role |
|---|---:|---|
| `critic_abl_a250_20261106/training/replay.pt` | 3.2 GB | Historical working-baseline replay; B1 fit data |
| `wm_reliability_supplement_20261108` | 8 MB | Supplement episodes: `dev` = route B1 calibration, `final` = B1 held-out screen (opened and used on 2026-11-10; not a fresh test set) |
| `wm_reliability_20261108`, `wm_route_compare_20261108` | 1 MB, 1.5 GB | WM reliability evaluations and the fixed-data original-vs-A fits |
| `a_pair_base_20261108`, `a_pair_head_20261108` | 4.6, 4.2 GB | A control pair: checkpoints at updates 150/200/250/300/last, frozen and perturbed evaluations |
| `a_pair_reference_20261108`, `a_pair_compare_20261109` | <1 MB | Perturbed evaluation of the reference run; case-paired comparisons (retired protocol) |

**GPU7 (additional):** `runs/main_compare_20261109` holds the case-paired
comparisons of the retired action-noise protocol (working baseline vs policy62
and SAC85k). `/dataHDD/dong/Archive/gpu9_20261110` holds GPU9-only code (git
bundles with all branches, uncommitted patches and untracked files of
Cutting_DEV_critic, Cutting_DEV_sac, DiSECt and DiSECt-Optimization; GPU9 tmp
scripts including the 2026-11-04 video renderers), checksum-verified; all of
that code is superseded by main.

**GPU9** is back online (2026-11-10) but no longer a run target for this project
(planned for other work). Its original historical data stay there and were not
copied: `SliceIt/runs` (1.7 GB; Gazebo SAC run 1, stopped at 57722 environment
steps, final `ckpt-13`) and `CuttingWorldModel/runs` (11 GB; 2026-11-02/03 early
experiments). GPU7 can reach it with `ssh gpu9`.

## Cleanup and recoverability

`runs/operations/cleanup_20261110.json` records moves, checkpoint hashes,
removed frame caches and exact worktree recovery commits. Four clean archived
worktrees were removed: fixedbottom, fixrealwm, p0 and wmgrad. Each exact commit
is reachable from main and can be restored with the recorded `git worktree add`
command. Dirty/unmerged archives (including bclin, mppi and wmterm) were retained.

Current runtime scripts do not import the retired perturbation module. Its
helper, CLI and old figure builder are in `scripts/legacy` with their regression
checks retained. Dated RESULTS commands keep historical paths; use the
[script map](../scripts/README.md) for today's entry points.

A6000's canonical checkout was observed at `71be9ec`, clean; it was not synced
or modified in this cleanup. No matching cutting training/evaluation process
was found in the snapshot. One A6000 was busy, so existing dependencies and
other jobs were left untouched. Machine notes are snapshots, not live status.
GPU9: see "Data on other servers".
