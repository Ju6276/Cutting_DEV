# Free-cut configuration index

| File | Purpose and status |
|---|---|
| `interface_check.json` | Historical v1 rejected reward candidate; regression/reproduction only |
| `review_v2.json`, `review_v3.json` | Historical bounded reward/interface checks |
| `short_train_cucumber.json` | v3 short-run configuration used for separation-task engineering training |
| `eval_cucumber.json` | v3 frozen checkpoint's 64 development conditions; used by dated comparisons |
| `eval_board_cucumber_v4.json` | Retired coverage-task physical-evaluation template; not a training configuration |

Current direction is target-pose completion with a board execution boundary; see [task contract](../../docs/TASK_FREE_CUT.md) and [PLAN](../../docs/PLAN.md). No file here is yet a launch configuration for that simplified integrated training task. The v1-v3 equations are retained in the [historical contract](../../docs/archive/TASK_FREE_CUT_V1_V3.md).

Keep task identities, explicit six-axis simulation limits, per-material settling and metadata checks. Do not overwrite an old config to make an old checkpoint appear to use new success labels. The simulated force limits/friction are not calibrated Mini45 settings.
