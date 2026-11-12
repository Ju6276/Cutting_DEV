# Multi-condition configurations

2026-11-12: reproduction inputs for the old task, not approved settings for a
new long run. See [PLAN](../../docs/PLAN.md) and [RESULTS](../../docs/RESULTS.md).

- `s0_train.json`, `s0_eval_train.json`: old S3 training/in-distribution evaluation.
- `s0_heldout_material_low.json`, `s0_heldout_material_high.json`,
  `s0_heldout_start.json`, `s0_heldout_geometry.json`: old one-factor test designs.
- `smoke.json`: interface check, not learning/physical-validity evidence.

Numbers/paths remain for provenance and checkpoint compatibility. They still
use the pose-goal task and inherited reward. No corrected separation rule or
fixture is supplied. S0's 50 N / 5 N·m were simulation choices, not Mini45
calibration or real safety guarantees. The bottom hold and fixed-displacement
reference are diagnosed issues. A corrected task needs an explicit new version.
