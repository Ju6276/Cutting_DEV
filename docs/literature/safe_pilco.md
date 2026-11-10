# Safe Policy Search Using Gaussian Process Models
Reviewed 2026-11-10. [Full paper](https://arxiv.org/pdf/1712.05556).
Read: §§1–6, supplementary §§7.1–7.3; bibliography used for provenance only.
Remote PDF/text: /dataHDD/dong/Research/cutting_literature_20261110/contact_risk/1712.05556.*.
## Paper
- §§3–4: learn GP state increments; propagate Gaussian moments through the policy and model. Optimize expected return plus a weighted predicted trajectory-safety probability.
- §4.5: reject a candidate before physical execution if predicted safety is insufficient; increase the penalty and optimize again. Decrease it for excessively conservative candidates.
- §5: car collision avoidance, simulated building temperatures, and Swimmer joint constraints; report return/error, violating steps/episodes, and interactions. Initial random collection is excluded from some violation counts.
- §7: moment-matching covariances, alternative sum/log/probability surrogates, task parameters. The Gaussian approximation and product of marginal safe probabilities are approximations, not exact correlated rollout guarantees.
- Assumptions: fully observed low-dimensional state, smooth GP dynamics, known unsafe sets, model-conditional safety. No cutting or within-control-step wrench peaks.
## Public implementation inspected
[Repository](https://github.com/nrontsis/PILCO), commit a89c36a428c59fa1ddd92e8e5e76cf05254f614d.
Read pilco/models/pilco.py, safe_pilco_extension/safe_pilco.py,
safe_pilco_extension/rewards_safe.py, examples/safe_swimmer_run.py.
- PILCO.propagate combines policy and GP moments; training_loss is negative predicted reward. optimize_policy freezes GP parameters, retaining gradients through GP predictions to controller parameters.
- The actual Swimmer example uses additive negative constraint coefficients, computes a separate trajectory-risk gate, and adapts coefficients before admitting more interaction.
- Do not transplant the helper blindly: SingleConstraint passes a covariance diagonal as Normal.scale; SafePILCO.predict adds mu times risk while the base class maximizes reward. These static inconsistencies need independent execution checks before reuse; this review did not run upstream training.
## Relevance and limits
Direct prior art for uncertain learned dynamics + analytic policy gradients + safety checks.
A calibrated force-risk term is not novel merely because gradients cross a learned model.
Useful lesson: separate the optimization surrogate from the operational risk test and disclose their approximation.
Code dependencies and helper inconsistencies make this a conceptual reference, not the preferred integration base.
