# Current plan — 2026-11-14

## Selected task

One cut of a freely placed object. The user requested simplification: use the existing target-pose completion interface; retain the no-through-board execution boundary and force/workspace limits. Material separation and geometric coverage are separate evaluation diagnostics, not additional training requirements. Passive sliding is permitted; no new displacement penalty, no active rolling task. Contract: [TASK_FREE_CUT](TASK_FREE_CUT.md).

## Completed in this cleanup

- Rechecked the v4 implementation and fixed previous/current footprint alignment; stationary proximity no longer accumulates swept coverage.
- Invalid knife penetration aborts as a simulator error. No penetration penalty is added to RL. Numerical tolerance and board stop are distinct from force-limit failures.
- Preserved versioned source/v1-v3 code and tests for historical checkpoints, archived obsolete review instructions, and replaced conflicting current-task descriptions.
- Ran one bounded four-world frozen-policy boundary check on each existing free cucumber, prism and potato asset. These use the old v4 diagnostic contract, not the newly selected pose-task performance claim. Results and reset motion: [RESULTS](RESULTS.md), 2026-11-14.

## What remains before long training

1. The old target's translation-only blade endpoint is about 4.4 mm below the board (RESULTS). Connect the existing pose reward/termination to a board-aligned physical goal and the tested board interlock, with a new task identity. Do not add coverage/fragment progress to the WM. Validate physical/imagined task semantics in an actual update and resume; do not silently reuse the old separation labels.
2. Cucumber and prism can supply development interaction data; neither requires a fixed reference to break every spring before collection. Potato's current reset still has substantial centre-of-mass motion. Prepare a naturally stable resting pose before calling it stationary-object training data. Do not freeze it artificially or erase velocities to hide motion.
3. Write the bounded training configuration for the chosen available objects. Do not wait for every fracture mechanism to be explained, and do not start a three-object run with the current unsettled potato or the retired v4 training entry.

Material-ground overlap and residual connections remain physical-fidelity limitations to report. No claim of calibrated real cutting forces or complete physical separation follows from a pose-success rate. No new material/ground tuning or training was performed in this cleanup.

## Execution

GPU7 through tams61, isolated worktree `/dataHDD/dong/Projects/Cutting_DEV_board`, branch `codex/board-cut-v4` (branch name retains the audit origin). Source, assets, weights and run data remain on servers. Preserve old checkouts and checkpoints; their task definitions are unchanged. Current audit data: `runs/board_review_20261114/`. No long training is currently running from this work.
