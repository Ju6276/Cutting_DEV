# Documentation map

Start with PLAN for decisions and next work, TASK_FREE_CUT and SHARED_SCENE for
the current task (pose_depth_v3), and DATA for server locations. A dated
experiment or archived launch is not a new training instruction.

## Current documents

| Document | Purpose |
|---|---|
| [PLAN](PLAN.md) | Current decisions, fixed task/algorithm settings, completed budgets and next work |
| [PAPER](PAPER.md) | Research scope and claims that still require evidence |
| [TASK_FREE_CUT](TASK_FREE_CUT.md) | pose_depth_v3 task: pose completion, board stop/latch, rewards, survival and diagnostic labels |
| [SHARED_SCENE](SHARED_SCENE.md) | Authoritative shared scene, reset distribution, physical knife/mount and backend differences |
| [METHOD](METHOD.md) | Implementation reference: active learner configuration, WM/critic/actor variants, physics interfaces, GPU memory policy |
| [RESULTS](RESULTS.md) | Dated measurements and limitations from 2026-11-16 on, including failed runs |
| [RUNNING](RUNNING.md) | Environment, tests, current training/evaluation/diagnostic commands, resume, W&B and rendering |
| [DATA](DATA.md) | Server paths, checkpoints, preserved evidence and cleanup ledger |

## Subfolders

- [methods/](methods/): [CURRENT_BASELINES](methods/CURRENT_BASELINES.md) (SAC, frozen
  WM+MPPI and nonlearned FDCC+MPPI, plus WM+BPTT: one current code line,
  version-specific evidence, inputs and provenance);
  [VISION_ROUTE1](methods/VISION_ROUTE1.md) (pre-cut visual goal selection, DP3
  midpoint predictor and physical handoffs).
- [hardware/](hardware/): [REAL2SIM](hardware/REAL2SIM.md) (real-robot interface and
  calibration plan); [UR10E_FDCC_SETUP_ZH](hardware/UR10E_FDCC_SETUP_ZH.md) (Chinese
  UR10e + ATI FDCC bring-up handoff for the control PC).
- [reference/](reference/): [CUTTING_MECHANISMS](reference/CUTTING_MECHANISMS.md)
  (DiSECt/BladeMaster cutting mechanisms and task signals);
  [SOURCES](reference/SOURCES.md) (upstream revisions, patches and provenance).
- [literature/](literature/README.md): per-paper evidence records and the reading
  list; [candidate topic search](literature/force_adaptive_search_20261112.md).

[Script map](../scripts/README.md) and [configuration index](../configs/README.md)
identify execution entry points and versioned configs.

## Archive

Archived files are evidence, not current instructions.

- Split from current docs on 2026-11-18:
  [RESULTS 2026-11-11 … 15](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/RESULTS_20261111_20261115.md),
  [METHOD historical sections](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/METHOD_historical_20261118.md),
  [RUNNING historical commands](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/RUNNING_historical_20261118.md).
- [11-16–18 baseline detail](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/BASELINES_20261116_20261118.md):
  historical controller implementations and tests; three-controller physical
  outcomes were v2, whereas the 11-18 SAC smoke was v3.
- 2026-11-16 launches (pose_depth_v2, ended):
  [shared WM+BPTT runs](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/TRAINING_SHARED_20261116.md),
  [GPU9 vector SAC](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/TRAINING_SAC_20261116.md),
  [previous shared-launch plan](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/PLAN_before_shared_long_20261116.md),
  [frozen policy mechanics](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/POLICY_MECHANICS_20261116.md),
  [framework provenance audit](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/FRAMEWORK_AUDIT_20261116.md).
- Fixed-prism era: [BENCHMARK contract](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/BENCHMARK.md),
  [official SuperMimic adaptation](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/OFFICIAL_SUPERMIMIC.md),
  [NOVELTY synthesis](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/NOVELTY.md),
  [2026-11-02–10 results](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/RESULTS_20261102_20261110.md),
  [2026-11-10 evidence](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/EVIDENCE_20261110.md),
  [unselected method variants](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/METHOD_variants_20261112.md),
  [retired perturbations](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/ACTION_PERTURBATIONS.md).
- BladeMaster integration: [backend review](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/BLADEMASTER_BACKEND_REVIEW.md),
  [DiSECt asset conversion](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/BLADEMASTER_DISECT_ASSETS.md),
  [batching review](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/BLADEMASTER_BATCHING_REVIEW_20261114.md),
  [three-world training](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/BLADEMASTER_MULTIOBJECT_TRAINING_20261114.md).
- 2026-11-14 launches: [scale/launch record](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/TRAINING_SCALE_20261114.md),
  [stopped 192-world A6000](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/TRAINING_20261114.md).
- Completed reviews: [contact coordinates](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/CONTACT_FIX_REVIEW.md),
  [ground contact](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/GROUND_CONTACT_REVIEW.md),
  [full wrench](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/WRENCH_OBSERVATION_REVIEW.md),
  [v4 board/coverage audit](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/BOARD_CUT_REVIEW.md).
- Earlier task equations/reviews: [v1–v3](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/TASK_FREE_CUT_V1_V3.md),
  [first review](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/FREE_CUT_REVIEW_20261112.md),
  [repair](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/archive/FREE_CUT_REPAIR_20261113.md).

Redundant plan/paper drafts were removed from the working tree on 2026-11-15.
Their original content remains at these immutable Git references:
- [PAPER_20261111.md](https://github.com/Ju6276/Cutting_DEV/blob/188c9eb/docs/archive/PAPER_20261111.md)
- [PAPER_20261112_pre_review.md](https://github.com/Ju6276/Cutting_DEV/blob/188c9eb/docs/archive/PAPER_20261112_pre_review.md)
- [PLAN_20261110.md](https://github.com/Ju6276/Cutting_DEV/blob/188c9eb/docs/archive/PLAN_20261110.md)
- [PLAN_20261111.md](https://github.com/Ju6276/Cutting_DEV/blob/188c9eb/docs/archive/PLAN_20261111.md)
- [PLAN_20261112_before_free_task.md](https://github.com/Ju6276/Cutting_DEV/blob/188c9eb/docs/archive/PLAN_20261112_before_free_task.md)
- [PLAN_20261112_pre_review.md](https://github.com/Ju6276/Cutting_DEV/blob/188c9eb/docs/archive/PLAN_20261112_pre_review.md)
