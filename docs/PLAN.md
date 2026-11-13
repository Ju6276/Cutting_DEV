# Current plan

Updated 2026-11-13. This is the only current work list. Task equations belong in
[TASK_FREE_CUT](TASK_FREE_CUT.md), implementation in [METHOD](METHOD.md), measured
results in [RESULTS](RESULTS.md), and research claims in [PAPER](PAPER.md).

## Decisions in force

- One cut of a freely placed object. Passive sliding is logged, with no new
  sliding threshold or penalty; active rolling and multiple cuts are out of scope.
- Use the versioned v3 separation/reward contract. Simulation wrench limits are
  not Mini45-derived real-robot limits. Do not tune physics to manufacture difficulty.
- The current run keeps the learning method, network, reward and action mapping
  unchanged. Six-axis policy observation expansion is not implemented in this run.
- Research targets a shared policy across conditions eventually; today's evidence
  is one cucumber scene. No multi-object/SAC retraining, S3 restart, risk-penalty
  experiment or real-robot run is authorized by this document.

## Authorized continuation

| Item | Recorded setting |
|---|---|
| Question | With further joint training and new interactions, do physical performance and WM/critic prediction improve? |
| Training checkout | GPU7 `/dataHDD/dong/Projects/Cutting_DEV_infra`, clean commit `b1bcb92` |
| Diagnostic checkout | GPU7 `/dataHDD/dong/Projects/Cutting_DEV`, diagnostic fix `12b951f` |
| Resume | `runs/free_cut_short_20261112/training_resume/last.pt`, iteration 1441, policy update 43 |
| Output | `runs/free_cut_joint_20261113/training`; live status in `progress.json` |
| Task/config | `configs/free_cut/short_train_cucumber.json`, free_cut_v3, 256 worlds |
| Budget | 1250 additional vector steps (endpoint 2691), at most 9 hours; no automatic extension |
| Saving | Every global iteration divisible by 250: first 1500, then 1750, etc.; final save at termination |
| Evaluation | Same 64 first-episode development conditions from `configs/free_cut/eval_cucumber.json`; no action perturbations |
| Diagnostic output | `runs/free_cut_joint_20261113/eval/` |

This is approximately 500 **additional** policy updates if the iteration budget
finishes. If the wall limit ends it first, report the actual completed count.
Training and diagnostics share GPU7; elapsed-time estimates are not exclusive-GPU
throughput. The first periodic save is 59 steps after resume, not 250.

The old checkpoint lacked a saved fractional update phase. The infrastructure
repair restores its weights/optimizers/replay, avoids repeating collection
warmup, warns and records `legacy_defaults_origin0_phase0`. This is a declared
migration, not exact recovery of the lost phase; new saves preserve that state.

## Checkpoint measurements and decisions

Claude runs training and the diagnostic queue; Codex reviews code, plans and
results. Compare conditions by world/case ID, one initial episode each.

- Physics: success, workspace/force/time endings, speed error and episode return.
- Completion: separating-step recognition **and** incomplete-step false predictions.
- Workspace: detection **and** false alarms within the same future window.
- Critic: estimated value, targets recomputed with the training formula, and
  realized remaining return. Recomputed targets are not historical optimizer targets.
- Raw records include pre/post state, executed actions, reward, events and history.
  Diagnostic hard-event priority/absorption follows the training task.

The critic-target comparison is still to be added to the checkpoint summary.
Judge trends through the fixed budget; no 0.1/50%/50 gates, no freeze-actor
prerequisite, no automatic extra budget. A single weak checkpoint does not select
a new algorithm. Propose a targeted change only with sustained evidence; an actual
implementation error is handled as a bug, separately from prediction quality.

## Deferred and known limits

Six-axis policy observations, multiple geometries, same-task SAC/MPPI comparisons,
uncertainty penalties and real-robot work need their own scope after this run.
Material/friction and the force sensor/mount are uncalibrated; the predefined cut
interface limits represented geometry changes. Long-horizon per-spring numerical
repeatability remains unresolved. These are limitations, not an expanding
pre-training checklist.

No experiment is launched by this documentation cleanup. Active checkouts, assets,
replay, checkpoints and diagnostic outputs are unchanged.
