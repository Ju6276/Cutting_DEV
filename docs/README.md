# Documentation map

| Document | Role |
|---|---|
| [PLAN](PLAN.md) | Sole current task list: current state, decisions by the user, what has to be resolved, conditional options |
| [TASK_FREE_CUT](TASK_FREE_CUT.md) | Current simplified pose-task direction, board boundary, diagnostics, and explicit implementation status |
| [PAPER](PAPER.md) | Research scope (open): why the old problem statement was retired, task-first scope, preliminary prior art, evidence needed |
| [Literature search 2026-11-12](literature/force_adaptive_search_20261112.md) | Prior art for the candidate topic, searched outside the library (second-hand evidence levels) |
| [OFFICIAL_SUPERMIMIC](OFFICIAL_SUPERMIMIC.md) | Official SuperMimic (5efbf11) adapted to the cutting prism: component mapping and every deviation |
| [BENCHMARK](BENCHMARK.md) | Historical fixed-prism contract/commands, not the corrected multi-object task |
| [Training 2026-11-14](TRAINING_20261114.md) | Current authorized three-object launch, preflight and resume repair |
| [RUNNING](RUNNING.md) | Installation, execution and validation |
| [DATA](DATA.md) | Server data/checkpoint locations, status and cleanup record |
| [CONTACT_FIX_REVIEW](CONTACT_FIX_REVIEW.md) | Review handoff for the contact correction: derivation, evidence, test-coverage change and limitations |
| [METHOD](METHOD.md) | Implemented algorithms, retained variants and force-interface limitations |
| [RESULTS](RESULTS.md) | Measurements from 2026-11-11 on, with a table of what currently holds; earlier entries are archived |
| [NOVELTY](NOVELTY.md) | Literature evidence and candidate ideas, not implemented methods |
| [READING_LIST](READING_LIST.md) | Literature/data entry points |
| [REAL2SIM](REAL2SIM.md) | Conditional real-robot draft, not a scheduled task |
| [Ground-contact review](GROUND_CONTACT_REVIEW.md) | Ghost nodes versus actual contact geometry; corrected tofu diagnosis |
| [BladeMaster assets](BLADEMASTER_DISECT_ASSETS.md) | Four reproduction geometries and elastic parameters converted to prescribed-interface DiSECt scenes |
| [BladeMaster multi-object run](BLADEMASTER_MULTIOBJECT_TRAINING_20261114.md) | Three intact meshes, native cutting, validation and GPU7 engineering training |
| [BladeMaster backend](BLADEMASTER_BACKEND_REVIEW.md) | Independent UR5e/FDCC material backend, distinct task semantics and actual update/restart checks |
| [SOURCES](SOURCES.md) | Upstream versions and provenance |

[Script map](../scripts/README.md) distinguishes current execution, source
reproduction and targeted analysis.

## Archive

[PLAN_20261112_pre_review](archive/PLAN_20261112_pre_review.md) and [PAPER_20261112_pre_review](archive/PAPER_20261112_pre_review.md) (candidate topic definition, defect table, ordered steps and topic options before the review rewrite),
[PAPER_20261111](archive/PAPER_20261111.md) and [PLAN_20261111](archive/PLAN_20261111.md) (retired problem statement and S3 launch plan),
[PLAN_20261110](archive/PLAN_20261110.md) (route B screening rules and S0–S4 rationale),
[RESULTS_20261102_20261110](archive/RESULTS_20261102_20261110.md) (fixed-prism development measurements),
[METHOD_variants_20261112](archive/METHOD_variants_20261112.md) (variants not adopted),
[Action perturbations](archive/ACTION_PERTURBATIONS.md) and
[2026-11-10 evidence package](archive/EVIDENCE_20261110.md) preserve the retired
protocol and old tables. Their perturbation results are excluded from the paper.
RESULTS retains measured failures and superseded decisions for traceability;
use PLAN for what happens next. No historical evidence is deleted for looking
unfavorable.

Historical v4 coverage-task and frozen-policy audit: [BOARD_CUT_REVIEW](BOARD_CUT_REVIEW.md).

Historical task equations and code reviews: [v1-v3 contract](archive/TASK_FREE_CUT_V1_V3.md), [first review](archive/FREE_CUT_REVIEW_20261112.md), [v2/v3 repair](archive/FREE_CUT_REPAIR_20261113.md). They are retained for reproducibility, not additional current steps.
