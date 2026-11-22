# Cutting DEV working rules

- Learning lives in `cutting_world_model/`; bundled physics lives in
  `backends/mujoco/`; SliceIt source versions and patches live in `integrations/`.
- Repository organization changes preserve physics, training settings and task
  thresholds. Keep active runs on their existing checkouts until they finish.
- Validate both root and backend suites on the 4090 before committing. Use the
  commands in `docs/OPERATIONS.md`; exclude assets, runs, credentials and environments
  from Git. Document source modifications in METHOD and preserve licenses.

- Follow docs/RESEARCH.md; it is the single current task list. The current task is
  pose_depth_v3 with `task.board.motion=project_continue_v2` (docs/METHOD.md): cumulative arrival >10, 2 mm tolerance, +100/-100,
  existing failure rules, 500-step horizon, no permanent board latch, no
  first-arrival/V4 task and no board reaction-force model. Older task versions
  (v2, latch-era V3, project_continue_v1, pose-v1, free_cut, fixed prism) are
  evaluate-only; do not mix replay/checkpoints across task metadata.
- Pose success is not validated separation; scalar material feedback is not a
  Mini45 contact wrench. The force-spike premise and S3 comparison are retired.
  Sawing and geometry input are user-favoured directions; start them from docs/RESEARCH.md,
  not because old code remains runnable.
- Retained implementation: SuperMimic-Reproduced `ba610534` adapted to the SliceIt
  FDCC + DiSECt/BladeMaster task in the unified MuJoCo Warp environment.
  The original SliceIt SAC in Gazebo stays an independent historical baseline.
- The SuperMimic source configuration stays runnable as the `source` baseline
  variant: training objectives, update cadence, effective batch sizes, optimizer
  settings and network widths unchanged. Method changes enter as named variants,
  one at a time, documented in docs/METHOD.md and compared against `source`.
  Document task-specific interfaces and scale provenance in METHOD.
- Every paid run needs an approved plan first: question, hypothesis, setting,
  metric, decision rule and wall/GPU budget. Docs separate measured results from
  inferences; search the literature before claiming novelty.
- Frozen-controller comparisons may retain different training budgets and inputs
  when disclosed. Matched-data/sample-efficiency claims require their own
  budget and information-access protocol.
- New material/geometry distributions are explicit, versioned tasks under docs/RESEARCH.md;
  record physical parameter provenance and do not change a contract silently.
  No scene-specific success shortcuts, damage projection or invented reward head.
- Action-perturbation evaluations are retired and excluded from the paper.
  Normal training exploration is a separate setting.
- Short-term loss/reward oscillation and a few weak episodes do not establish
  model failure. Use sustained learning curves and held-out physical evaluation.
  Fix demonstrated implementation errors; do not redesign from a tiny fit.
- One connected candidate at a time. Tests serve data alignment, equations,
  gradient correctness and actual learning execution.
- All remote projects, environments, caches and runs stay under /dataHDD/dong.
  Every long job has a wall budget, progress, checkpoint and cleanup behavior.
- Keep runs out of Git. Commit small changes with English messages after
  `PYTHONPATH=. .venv/bin/python -m pytest tests -q` passes on the 4090.
- Docs state runnable commands, current implementation, numeric results and
  next work. Remove replaced implementations and superseded docs; Git preserves
  their history.

- Development uses Remote SSH on GPU7. Keep project source, weights, datasets,
  caches and videos on the servers; do not download project copies to the Mac
  unless explicitly requested. GPU9 is reached through tams61.

- Contact search, interpolation and force scattering use
  `edge_coordinate_consistent_v2` (2026-11-12), recorded in task metadata for all
  feedback modes. Historical physics is at `38b85a5`; do not bypass metadata
  mismatches to present old checkpoints as same-physics results.

- Every long training run uses online W&B (--wandb-mode online); never substitute
  offline/disabled to work around login or network errors. Verify an
  authenticated account, a live run URL and remotely readable history before
  leaving a long run unattended. If online initialization fails, fix it or
  report the blocker before training. Server-side checkpoints/logs remain
  recovery copies. Short capacity/diagnostic scripts are not long training.
  GPU7 and A6000 credentials are in /dataHDD/dong/.config/wandb/netrc; use NETRC
  without printing file contents or keys.
