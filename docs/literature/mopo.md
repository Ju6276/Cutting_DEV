# MOPO — pessimistic offline model-based policy optimization

> Project interpretation updated 2026-11-19; original paper/code reading scope is unchanged.
> Superseded project hypotheses are [archived](../archive/LITERATURE_POSITIONING_20261110.md).

Reviewed 2026-11-10. [Paper, 2005.13239](https://arxiv.org/abs/2005.13239).
Read scope: §§3–6, assumptions, practical uncertainty estimator, main tables.
Core author code inspected at
[tianheyu927/mopo, e5efe3eaa548850af109c920bfc2c1ec96bf2285](https://github.com/tianheyu927/mopo/tree/e5efe3eaa548850af109c920bfc2c1ec96bf2285).
No reproduction executed.

## Method and evidence

An offline probabilistic dynamics ensemble produces short synthetic rollouts
for an off-policy learner. Predicted rewards are penalized by uncertainty.
The lower-return-bound analysis assumes an admissible model-error upper
estimator for all relevant states/actions. The practical maximum predicted
standard-deviation norm is explicitly heuristic; the theorem does not certify
that estimator.

D4RL results report normalized returns across twelve dataset/task combinations;
task-relabeling experiments test behavior beyond the original dataset.
Performance varies with coverage, including weak results on narrow medium
datasets. This is not online force-safe BPTT or a peak-force guarantee.

## Code evidence

`mopo/models/fake_env.py::step` samples an ensemble prediction, calls the
environment-specific termination function on predicted state, and subtracts
`penalty_coeff * penalty`. Options distinguish spread of ensemble means from
norms of each member's learned standard deviations. The diagnostic
`step_ph` path does not apply the same penalty; it is not evidence of a ready
differentiable constrained actor implementation.

## Relevance to the current method

Model pessimism is prior art, not a component of the selected main configuration. This paper is background for distinguishing model error from control performance; it does not establish a calibrated sensor-force bound for our task.
