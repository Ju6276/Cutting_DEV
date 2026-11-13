# Free-cut infrastructure repair for Claude review — 2026-11-13

Base: `c912fb4` on `codex/free-cut-v1`. Review branch: `codex/free-cut-infra`.
GPU7 checkout: `/dataHDD/dong/Projects/Cutting_DEV_infra`.
Evidence directory: `runs/infra_fix_20261113/` (server only, not tracked).
Claude's active checkout and diagnostic script are unchanged. No policy training launched.

## Scope and code

1. `online_schedule.py`, `train_gpu.py`, `train_coupled.py`:
   persist collection warmup origin, fractional update phase and ScheduleCfg. Restore
   them when replay is restored; explicitly refill if replay is intentionally absent.
   Old schedule files warn, use source defaults/origin 0/phase 0 and record that migration.
   Their prior phase was never saved; it cannot be asserted to match continuous training.
   This fixes schedule advancement, not mid-optimizer interruption replay or physical
   trajectory continuity. New checkpoints are written at existing save sites.
2. `evaluation_lifecycle.py`, `evaluate_multi.py`:
   guarantee close via `finally` for normal exits, timeout, exceptions and catchable
   termination; restore signal handlers. Already-written episode rows remain available.
3. `metric_counts.py`, `world_model.py`, `trainer.py`:
   accumulate int64 counts then divide once. No clipping to conceal rates above 1.
   Rates now pool valid transitions across the whole WM update; names keep their original
   unconditional denominator (missed/false-positive fraction of all transitions).
   Losses, gradient routes and optimizer weighting stay unchanged.

## Verification

- Continuous vs save/resume cadence: splits before/during/after warmup, repeated resumes,
  nondefault schedule and fractional or multiple updates per step; explicit empty replay;
  rejected corrupt state; documented legacy migration.
- Evaluation entrypoint: normal return, max steps, step exception, wall timeout and
  KeyboardInterrupt after a completed episode; environment closed exactly once, episode
  row retained. Separate real OS-signal tests reap a real child process under SIGINT/SIGTERM.
- Integer metric tests: poisoned padding, uneven window lengths, uneven microbatches and
  21 minibatches; exact pooled counts and bounded rates.
- GPU7 full root suite, including enabled CUDA assets and optional MPPI environment:
  **344 passed, 5 skipped**, 234.82 s. Backend: **38 passed**, 5.82 s.
  Logs: `runs/infra_fix_20261113/root_tests.log` and `backend_tests.log`.
  Existing upstream DiSECt AST deprecation warnings remain; no kernel changed.
- Against the exact `c912fb4` WM/trainer source, same v3 masked batch and initial weights:
  float32 and float64 loss, every parameter gradient, and parameters after two Adam
  steps are bitwise equal. Evidence: `baseline_equivalence.json` in the same run directory.
  This isolates the metric correction from the optimization; it is not a convergence test.

Commands (after sourcing the existing `runs/sac_env.sh`):

    CUTTING_BACKEND_ROOT="$PWD/backends/mujoco" PYTHONPATH=. OMP_NUM_THREADS=2 MAX_JOBS=2 /dataHDD/dong/Envs/cutting-mppi-20261108/bin/python -m pytest tests -q
    (cd backends/mujoco && PYTHONPATH=. OMP_NUM_THREADS=2 MAX_JOBS=2 ../../.venv/bin/python -m pytest tests -q)

The initial OS-signal regression expected shell-style 130 for SIGINT; Python's
`Popen` reports `-SIGINT` after an unhandled KeyboardInterrupt. The test now checks
that OS behavior, together with the child-reaped marker. Cleanup code was not changed
in response to that test expectation.

## Remaining investigation

No change to completion calibration, force prediction, critic targets, reward, a4 integration
or workspace limits. Claude's same-trajectory comparison is responsible for identifying
why the policy's predicted improvement differs from physical execution. These infrastructure
checks do not establish model accuracy or authorize long training.
