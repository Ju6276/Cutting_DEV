# Current plan — 2026-11-13

## Task decisions

- Single cut of a freely placed object; passive sliding is recorded without a displacement penalty. Active rolling and multi-cut planning are outside this task.
- The user authorized effective knife-path coverage plus arrival at the cutting board, within force/workspace limits. The knife may not pass through the board. Opt-in `free_cut_v4` implements this physical task; v1–v3 preserve their historical separation-based semantics.
- Report geometric path completion, material separation and force events separately. Path completion is not proof that the two pieces have physically separated.
- Keep the six-component simulated force/torque limits. These are not calibrated Mini45 hardware limits. No material/friction tuning to make the task harder.

## Current authorized delivery

Implement the board execution stop and evaluate the frozen 543-update cucumber policy against the same 64 development conditions. No new training, reward tuning, material damage changes, or replacement of historical results. Protocol, implementation and results: [BOARD_CUT_REVIEW](BOARD_CUT_REVIEW.md); experiment history: [RESULTS](RESULTS.md).

The board stop is a software motion interlock, not a compliant board-force model. Geometric coverage samples the moving intended section; it is not a continuum separation proof. v3 WM and critic labels are incompatible with v4 and must not be silently reused. The v4 training path is explicitly rejected pending a matching learned-task implementation and a separately approved training budget.

## Following work

After this audit, use its outcome to decide the next training/task change. Do not restart S3, change materials, relax coverage, or launch multi-object training automatically. Potato placement and prism residual links remain measured platform limitations, not prerequisites for completing this audit. Historical task and training plans are retained in Git and RESULTS; the previous separation-only definition does not define v4.

## Execution and storage

Remote SSH on GPU7 through tams61. Current isolated worktree: `/dataHDD/dong/Projects/Cutting_DEV_board`, branch `codex/board-cut-v4`. Preserve the original `codex/free-cut-v1` checkout and all checkpoints. Source, environments, assets and runs remain on servers. Check GPU usage before execution.
