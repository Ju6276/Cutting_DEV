# Documentation map

Use PLAN for current work, TASK_FREE_CUT for the active task, and DATA for server
locations. A dated experiment or archived launch is not a new training instruction.

| Document | Role |
|---|---|
| [PLAN](PLAN.md) | Current decisions, GPU7/A6000 deployments, fixed budgets and next work |
| [SAC training](TRAINING_SAC_20261116.md) | GPU9 bounded vector SAC launch |
| [Shared training](TRAINING_SHARED_20261116.md) | Current task launch protocol, verification and online run records |
| [Shared scene](SHARED_SCENE.md), [knife](SHARED_KNIFE.md) | Authoritative shared settings, physical geometry and backend differences |
| [TASK_FREE_CUT](TASK_FREE_CUT.md) | Pose completion, board boundary, running rewards and diagnostic labels |
| [RUNNING](RUNNING.md) | Installation, tests and online W&B requirement |
| [DATA](DATA.md) | Current server paths, checkpoints, preserved evidence and cleanup ledger |
| [Current baselines](CURRENT_BASELINES.md) | SAC source search, current vector SAC and two model-based planners |
| [Vision](VISION_ROUTE1.md) | Route1 implementation, method2 fixed-heading pilot, controller integration status |
| [METHOD](METHOD.md) | Implemented learning methods and physical interfaces |
| [RESULTS](RESULTS.md) | Dated measurements and limitations, including failed runs |
| [PAPER](PAPER.md) | Research scope and claims still requiring evidence |
| [BENCHMARK](BENCHMARK.md) | Historical fixed-prism contract, not the active multi-object task |
| [OFFICIAL_SUPERMIMIC](OFFICIAL_SUPERMIMIC.md) | Official implementation adaptation and deviations |
| [BladeMaster backend](BLADEMASTER_BACKEND_REVIEW.md) | Initial integration history; current contract is SHARED_SCENE |
| [BladeMaster assets](BLADEMASTER_DISECT_ASSETS.md) | Candidate objects converted to prescribed-interface DiSECt |
| [NOVELTY](NOVELTY.md), [READING_LIST](READING_LIST.md) | Literature evidence and entry points |
| [Literature search](literature/force_adaptive_search_20261112.md) | Candidate topic search and evidence levels |
| [UR10e + ATI FDCC setup](UR10E_FDCC_SETUP_ZH.md) | Chinese handoff for the control PC: installation, interfaces and first hardware tests |
| [REAL2SIM](REAL2SIM.md) | Real-robot plan; UR10e hardware facts and conditional later work |
| [SOURCES](SOURCES.md) | Upstream revisions and provenance |

[Script map](../scripts/README.md) identifies execution and diagnostic entry points.

## Historical records

- [Old scale/launch record](archive/TRAINING_SCALE_20261114.md), [batching implementation review](archive/BLADEMASTER_BATCHING_REVIEW_20261114.md), [framework provenance audit](archive/FRAMEWORK_AUDIT_20261116.md). Current launch numbers are in the shared/SAC training records, not these historical files.

- Stopped launches: [192-world A6000](archive/TRAINING_20261114.md),
  [three-world BladeMaster](archive/BLADEMASTER_MULTIOBJECT_TRAINING_20261114.md).
- Completed reviews: [contact coordinates](archive/CONTACT_FIX_REVIEW.md),
  [ground contact](archive/GROUND_CONTACT_REVIEW.md),
  [full wrench](archive/WRENCH_OBSERVATION_REVIEW.md),
  [v4 board/coverage audit](archive/BOARD_CUT_REVIEW.md).
- Earlier evidence: [2026-11-02–10 results](archive/RESULTS_20261102_20261110.md),
  [unselected method variants](archive/METHOD_variants_20261112.md),
  [retired perturbations](archive/ACTION_PERTURBATIONS.md),
  [2026-11-10 evidence](archive/EVIDENCE_20261110.md).
- Earlier task equations/reviews: [v1–v3](archive/TASK_FREE_CUT_V1_V3.md),
  [first review](archive/FREE_CUT_REVIEW_20261112.md),
  [repair](archive/FREE_CUT_REPAIR_20261113.md).

Redundant plan/paper drafts were removed from the working tree on 2026-11-15.
Their original content remains at these immutable Git references:
- [PAPER_20261111.md](https://github.com/Ju6276/Cutting_DEV/blob/188c9eb/docs/archive/PAPER_20261111.md)
- [PAPER_20261112_pre_review.md](https://github.com/Ju6276/Cutting_DEV/blob/188c9eb/docs/archive/PAPER_20261112_pre_review.md)
- [PLAN_20261110.md](https://github.com/Ju6276/Cutting_DEV/blob/188c9eb/docs/archive/PLAN_20261110.md)
- [PLAN_20261111.md](https://github.com/Ju6276/Cutting_DEV/blob/188c9eb/docs/archive/PLAN_20261111.md)
- [PLAN_20261112_before_free_task.md](https://github.com/Ju6276/Cutting_DEV/blob/188c9eb/docs/archive/PLAN_20261112_before_free_task.md)
- [PLAN_20261112_pre_review.md](https://github.com/Ju6276/Cutting_DEV/blob/188c9eb/docs/archive/PLAN_20261112_pre_review.md)

Current scene/reset source and complete backend differences: [Shared scene](SHARED_SCENE.md).
