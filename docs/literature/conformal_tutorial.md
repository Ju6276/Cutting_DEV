# A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification

- Primary paper: [Angelopoulos and Bates, arXiv:2107.07511v6](https://arxiv.org/abs/2107.07511v6), December 2022, 51 pages. Read on 2026-11-10: Sections 1–7 and Appendices A–D, including coverage proofs, distribution-shift qualifications and risk-control examples. Bibliography entries were used as pointers, not counted as separately read papers.
- Author repository: [aangelopoulos/conformal-prediction](https://github.com/aangelopoulos/conformal-prediction), inspected commit `0426de356dbfa71c4c46be81d1973255ac66d050`.
- Remote evidence: `/dataHDD/dong/Research/cutting_literature_20261110/contact_risk/2107.07511.pdf`, `code/conformal_tutorial/`.
- No notebooks or model fitting were executed. Only the notebook source was read.

## Core construction and its exact guarantee

Split conformal uses a model fixed before calibration, an error score, and the corrected order statistic at rank `ceil((n+1)(1-alpha))`. With exchangeable calibration and future examples it gives **marginal prediction-set coverage**. It does not guarantee coverage at each state, at every time step, or for each new optimized policy.

Appendix D is explicit about the small-sample corner case: if the requested rank exceeds the number of calibration observations, the mathematically valid threshold is infinity. Silently clipping to the largest finite residual changes the guarantee.

Section 2 gives classification sets, CQR, scaled residuals and Bayesian-density sets. The scalar-uncertainty construction already includes ensemble standard deviation, dropout and learned residual magnitudes. Therefore “ensemble plus calibrated upper margin” is a standard method, not by itself our algorithmic contribution.

## Evaluation and finite-sample interpretation

Section 3 separates marginal coverage from adaptivity and conditional coverage. It recommends checking coverage within feature groups and set-size groups; overall 90% coverage can coexist with zero coverage for a rare 10% subgroup.

The variability of coverage conditional on the calibration sample is described by a Beta law; finite validation samples add a beta-binomial component. The tutorial's rough recommendation of 1,000 calibration examples is an illustration under its setup, not a universal requirement to impose on our project.

For our 13 force-limit events, discovery count is an event-detection metric, not a conformal coverage proof. Correlated frames from one episode must not be passed off as independent calibration trials. Calibration, development selection and final evaluation have different roles.

## Shift, sequential data and risk control

- Section 4.5 handles covariate shift with known density ratios and unchanged conditional label law. Updating a policy does not automatically satisfy those conditions.
- Section 4.6 gives a drift bound with a weighted total-variation penalty. Unknown shift magnitudes do not disappear by choosing a rolling window.
- Section 5.3 uses a moving-window temperature example and explicitly states that the unknown TV distances prevent a formal numerical coverage bound there.
- Conformal risk control applies to bounded losses monotone in the tuning parameter. It controls an expected loss; it is different from a high-probability statement about a fixed deployed model.
- Appendices A/B explain Learn then Test: test candidate risk constraints and control familywise error across selected thresholds. It is not valid to try many thresholds on a small final test set and only report the best.
- Appendix C's concentration accounting assumes the stated sampling construction. Reusing one finite data pool across random splits does not magically create independent robot episodes.

## Code inspected

Read all executable cells of:

- `notebooks/meps-cqr.ipynb`: loads cached predictions, splits calibration/validation, uses `max(y-upper, lower-y)`, computes the corrected quantile and reports coverage.
- `notebooks/meps-uncertainty-scalar.ipynb`: normalizes absolute residual by an uncertainty scalar, then rescales the calibrated interval.
- `notebooks/weather-time-series-distribution-shift.ipynb`: `get_weighted_quantile` uses only a preceding window and compares against cumulative calibration. It does not train a controller or propagate gradients through a world model.
- `notebooks/correctness_checks.ipynb`: Beta/beta-binomial calculations and simulation of calibration/validation variability.

The core routines are NumPy/SciPy post-processing of fixed predictions. There is no actor BPTT, force simulator or differentiable conformal optimizer in these notebooks. They use fixed demonstration sample sizes and are not general-purpose production APIs for arbitrarily tiny calibration sets.

A source-level caution: the uncertainty-scalar notebook calculates its coverage using rescaled intervals, but its final plotting cell reuses the CQR-style additive bounds. The illustrative plot should therefore not be copied as a verified visualization of the scalar method without correction. This is a static observation, not a reproduced numerical failure.

## Implications for Route B

A calibrated upper force bound may be a useful component and diagnostic. It cannot alone justify “safe cutting” or a new policy-learning algorithm.

In particular, with a constant calibrated radius `U(s,a)=f_hat(s,a)+q`, the action derivative remains `dU/da=df_hat/da`. Better numerical coverage does not repair an incorrect action-dependent force gradient. An action-dependent uncertainty term can change this derivative, but its reliability still requires evidence under the actions the optimizer actually chooses.

For a small pilot, freeze the operating point on development data, preserve episode-level pairing, report gained and lost detected events and false alarms, and treat any count threshold as a resource-allocation rule. Do not rename it statistical significance or deployment safety.

## Remaining limits

Only the four listed notebook sources were read; the other examples and data-generation scripts were not audited. The tutorial's linked literature was not automatically counted as read. No independent proof audit of every theorem or runtime reproduction was performed.
