# Cutting DEV working rules

- Learning lives in `cutting_world_model/`; bundled physics lives in
  `backends/mujoco/`; SliceIt source versions and patches live in `integrations/`.
- Repository organization changes preserve physics, training settings and task
  thresholds. Keep active runs on their existing checkouts until they finish.
- Validate both root and backend suites on the 4090 before committing. Use the
  commands in `docs/RUNNING.md`; exclude assets, runs, credentials and environments
  from Git. Document source modifications in METHOD and preserve licenses.

- Current phase (2026-11-12): task/platform definition and problem-led literature
  review. The force-spike premise and first S3 comparison are retired. Historical
  pose success is not validated separation; scalar material feedback is not a
  Mini45 contact wrench. Do not restart S3 from a reproduction command. Sawing and
  geometry input are directions chosen or favoured by the user (PLAN); start work on
  them from PLAN, not because old code remains runnable. Follow PLAN.
- Retained implementation: SuperMimic-Reproduced `ba610534` adapted to the SliceIt UR5e +
  FDCC + DiSECt task in the unified MuJoCo Warp environment (`docs/BENCHMARK.md`).
  The original SliceIt SAC in Gazebo stays an independent baseline. PLAN is the
  single current task list; BENCHMARK defines the historical fixed-prism contract.
  The opt-in versioned free-cut task is defined in TASK_FREE_CUT; its rewards and training budget remain under review.
- The SuperMimic source configuration stays runnable as the `source` baseline
  variant: training objectives, update cadence, effective batch sizes, optimizer
  settings and network widths unchanged. Method changes enter as named variants,
  one at a time, documented in docs/METHOD.md and compared against `source`.
  Document task-specific interfaces and scale provenance in METHOD.
- Every paid run needs an approved plan first: question, hypothesis, setting,
  metric, decision rule and wall/GPU budget. Docs separate measured results from
  inferences; search the literature before claiming novelty.
- Use the shared benchmark entries and task definitions (docs/BENCHMARK.md).
  Frozen-controller comparisons may retain different training budgets and inputs
  when disclosed. Matched-data/sample-efficiency claims require their own
  budget and information-access protocol; they do not mandate retraining for
  every checkpoint comparison.
- Preserve the existing SliceIt-derived benchmark's rewards, thresholds and
  physics. New material/geometry distributions are explicit, versioned tasks
  under PLAN; record physical parameter provenance and do not change the old
  contract silently. No scene-specific success shortcuts, damage projection or
  invented reward head.
- Action-perturbation evaluations are retired and excluded from the paper;
  preserve historical records. Normal training exploration is a separate setting.
- Short-term loss/reward oscillation and a few weak episodes do not establish
  model failure. Use sustained learning curves and held-out physical evaluation.
  Fix demonstrated implementation errors; do not redesign from a tiny fit.
- One connected candidate at a time. Tests serve data alignment, equations,
  gradient correctness and actual learning execution.
- All remote projects, environments, caches and runs stay under /dataHDD/dong.
  Every long job has a wall budget, progress, checkpoint and cleanup behavior.
- Keep runs out of Git. Commit small changes with English messages after
  `PYTHONPATH=. .venv/bin/python -m pytest tests -q` passes on the 4090.
- Docs state runnable commands, implementation, numeric results and next work.
  Remove replaced implementations; Git preserves their history.

- Development uses Remote SSH on GPU7. Keep project source, weights, datasets,
  caches and videos on the servers; do not download project copies to the Mac
  unless explicitly requested. GPU9 is retired from the working plan.


- Authorized physical correction (2026-11-12): contact search, interpolation and
  force scattering use `edge_coordinate_consistent_v2`. It is recorded in task
  metadata for all feedback modes. Historical physics is at `38b85a5`; do not
  bypass metadata mismatches to present old checkpoints as same-physics results.
