# Historical reproduction tools

These tools retain old result reproducibility. Action-perturbation evaluation
is retired and excluded from the current paper. No new experiment is scheduled
by these files.

- `evaluate_perturbed.py`: historical simulation and paired-case comparison.
- `perturbed_eval.py`: case streams and historical result summaries.
- `build_evidence.py`: rebuild the old evidence package from saved files.
- `collect_reliability_data.py`, `wm_reliability_eval.py`, `wm_branch_check.py`, `wm_fixed_fit.py`,
  `wm_loss_gradients.py`: world-model reliability diagnostics of the retired problem statement
  (docs/archive/PAPER_20261111.md).
- `value_eval.py`, `check_value_learning.py`: critic diagnostics of the 2026-11-04..06 critic study.
- `b1_force_risk.py`: route B1 force-risk screen (completed 2026-11-10).

Run from the repository root using `python scripts/legacy/NAME.py`.
[Protocol](../../docs/archive/ACTION_PERTURBATIONS.md) ·
[Evidence archive](../../docs/archive/EVIDENCE_20261110.md) ·
[Current plan](../../docs/PLAN.md).
