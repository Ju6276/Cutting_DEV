# Cutting DEV working rules

- Learning lives in `cutting_world_model/`; bundled physics lives in
  `backends/mujoco/`; SliceIt source versions and patches live in `integrations/`.
- Repository organization changes preserve physics, training settings and task
  thresholds. Keep active runs on their existing checkouts until they finish.
- Validate both root and backend suites on the 4090 before committing. Use the
  commands in `docs/RUNNING.md`; exclude assets, runs, credentials and environments
  from Git. Document source modifications in METHOD and preserve licenses.

- Migrate SuperMimic-Reproduced `ba610534` to the existing SliceIt Gazebo + DiSECt
  UR5e robot/compliance task. Keep the original SAC baseline independent.
- The SuperMimic source configuration stays runnable as the `source` baseline
  variant: training objectives, update cadence, effective batch sizes, optimizer
  settings and network widths unchanged. Method changes enter as named variants,
  one at a time, documented in docs/METHOD.md and compared against `source`.
  Document task-specific interfaces and scale provenance in METHOD.
- Every paid run needs an approved plan first: question, hypothesis, setting,
  metric, decision rule and wall/GPU budget. Docs separate measured results from
  inferences; search the literature before claiming novelty.
- All methods are evaluated through `scripts/evaluate_benchmark.py` with matched
  real-transition and wall budgets (docs/BENCHMARK.md).
- Keep SliceIt rewards, thresholds, material configuration and physical behavior.
  No scene-specific success shortcuts, damage projection or invented reward head.
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
