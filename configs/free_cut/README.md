# Free-cut configuration index

| File | Purpose and status |
|---|---|
| `pose_cucumber.json`, `pose_prism.json`, `pose_potato.json` | Current pose-task development configurations; not long-training budgets |
| `interface_check.json` | Historical v1 rejected reward candidate; regression/reproduction only |
| `review_v2.json`, `review_v3.json` | Historical bounded reward/interface checks |
| `short_train_cucumber.json` | v3 short-run configuration used for separation-task engineering training |
| `eval_cucumber.json` | v3 frozen checkpoint's 64 development conditions; used by dated comparisons |
| `eval_board_cucumber_v4.json` | Retired coverage-task physical-evaluation template; not a training configuration |

Current direction is target-pose completion with a board execution boundary; see [task contract](../../docs/TASK_FREE_CUT.md) and [PLAN](../../docs/PLAN.md). The three `pose_*.json` files select the integrated pose task. The potato entry now uses the unheld flat-bottom half and 1 s per-material settling. A production-reset check passed 128 randomized cases and 16 material/clearance corners; its nonzero cost still belongs in a training budget. The v1-v3 equations are retained in the [historical contract](../../docs/archive/TASK_FREE_CUT_V1_V3.md).

Keep task identities, explicit six-axis simulation limits, per-material settling and metadata checks. Do not overwrite an old config to make an old checkpoint appear to use new success labels. The simulated force limits/friction are not calibrated Mini45 settings.

`pose_three_object_check.json` is the three-world cucumber/half-potato/prism configuration used for the actual 36-transition and WM/actor/critic integration check. It is a runnable development input, not the long-training budget. Per-object settling is preserved in metadata.
