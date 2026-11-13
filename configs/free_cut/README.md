# Free-cut configurations

Current task: [free_cut_v3](../../docs/TASK_FREE_CUT.md).
[PLAN](../../docs/PLAN.md) defines the authorized continuation, not this directory.

| File | Role |
|---|---|
| `short_train_cucumber.json` | Current 256-world cucumber training/continuation; H=180 with derivation in file |
| `eval_cucumber.json` | 64 first-episode development conditions; same task version and horizon |
| `review_v3.json` | Bounded physical reference/reward check, H=100; not a replacement training config |
| `review_v2.json` | Historical v2 reward check; superseded by v3 |
| `interface_check.json` | Rejected v1 reward candidate, retained for reproduction |

All current groups select net feedback and per-material settling. Six-axis
limits are explicit simulation settings, not Mini45 calibration. Old configs
and checkpoints must keep their version; do not bypass compatibility checks.
The branch name `codex/free-cut-v1` is historical and does not describe the
current task version. Findings: [RESULTS](../../docs/RESULTS.md).
