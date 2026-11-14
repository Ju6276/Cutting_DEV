# Free-cut configuration index

| File | Purpose and status |
|---|---|
| `pose_cucumber.json`, `pose_prism.json`, `pose_potato.json` | Current pose-task development configurations; not long-training budgets |
| `interface_check.json` | Historical v1 rejected reward candidate; regression/reproduction only |
| `review_v2.json`, `review_v3.json` | Historical bounded reward/interface checks |
| `short_train_cucumber.json` | v3 short-run configuration used for separation-task engineering training |
| `eval_cucumber.json` | v3 frozen checkpoint's 64 development conditions; used by dated comparisons |
| `eval_board_cucumber_v4.json` | Retired coverage-task physical-evaluation template; not a training configuration |

Current direction is target-pose completion with a board execution boundary; see [task contract](../../docs/TASK_FREE_CUT.md) and [PLAN](../../docs/PLAN.md). The three `pose_*.json` files select the integrated pose task. The potato entry uses the re-exported natural support pose and 8 s per-material settling; its cost must be included in a training budget. The v1-v3 equations are retained in the [historical contract](../../docs/archive/TASK_FREE_CUT_V1_V3.md).

Keep task identities, explicit six-axis simulation limits, per-material settling and metadata checks. Do not overwrite an old config to make an old checkpoint appear to use new success labels. The simulated force limits/friction are not calibrated Mini45 settings.
