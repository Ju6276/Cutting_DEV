# Cautious Model Predictive Control Using Gaussian Process Regression

- Primary paper: [Hewing, Kabzan, Zeilinger, arXiv:1705.10702v4](https://arxiv.org/abs/1705.10702v4), 12 pages. Read on 2026-11-10: all main Sections I–VI, derivations, experiments and implementation approximations. No separate appendix was present in the retrieved PDF.
- Remote paper: `/dataHDD/dong/Research/cutting_literature_20261110/contact_risk/1705.10702.pdf`.
- Code search: checked the paper/arXiv entry and exact-title GitHub searches. No author repository specifically reproducing this paper was verified. Third-party implementations are not counted as author code. Evidence is **paper only**.

## Model and uncertainty propagation

The model is known nominal dynamics plus a GP residual in selected state dimensions, with Gaussian process noise. Independent scalar GPs model the residual outputs; the covariance and input-state uncertainty enter future predictions.

The paper contrasts mean-equivalent prediction, Taylor approximation and moment matching. Neglecting state/input covariance can yield either overconfidence or excessive conservatism. A local ancillary feedback term influences the covariance evolution.

The resulting optimization is a nonlinear MPC problem with uncertain states and inputs. Probabilistic reachable sets tighten deterministic constraints. In the polyhedral/box case the support-function calculation must preserve the sign-independent magnitude of each constraint row; this is relevant when translating prediction intervals into constraints.

## Assumptions and practical limits

Chance constraints rely on the GP/Gaussian and propagation approximations. They are not finite-sample, distribution-free guarantees, and they are not hard safety guarantees for arbitrary model mismatch. A valid one-step predictive interval alone does not establish a recursively valid full-horizon constraint.

The sparse GP uses dynamically selected inducing points near the preceding planned trajectory. The experiments deliberately make additional approximations for real-time operation: some covariance quantities are held fixed from the preceding solution, and the racing setup tightens only part of the horizon.

This matters for our comparisons: practical learned-model MPC is not synonymous with exact uncertainty propagation, and its compute/safety tradeoff must be described rather than hidden.

## Tasks, data and measurements

- An underwater-vehicle example learns residual dynamics online, with a small GP data set and periodic updates.
- An autonomous RC-car racing experiment uses roughly 325 measurements, a 30-stage horizon and a 20 ms control period. Uncertainty tightening is applied to the first 20 stages in the reported implementation.
- The racing results include lap time and solver-time statistics. An example comparison changes lap time from approximately 10.32 s to 9.61 s, while reported solver statistics show that the cautious implementation remains compatible with the 20 ms deadline in most iterations.

These are navigation/dynamics tasks, not cutting, force-peak regression or BPTT-trained policy learning. The gains cannot be transferred numerically to our simulator.

## Relation to Route B

Already established: learned dynamics + propagated model uncertainty + online optimization under chance constraints. Therefore simply putting an upper prediction bound into MPPI/MPC is not a new general principle.

Useful borrowing: distinguish observation noise, model error and rollout uncertainty; state exactly which approximation is propagated; budget online compute; account for action-dependent uncertainty.

What remains open for our implementation is not answered here: partially observed, irreversible cutting state; force maxima between control samples; and whether a model's action gradient points toward a physically lower-force action. Calibrating a marginal force bound is only one part of that control question.

## Remaining limits

No official source was verified or executed. The derivations were read but not independently re-proved in full. This note does not claim an exhaustive code search or that later author projects contain no related implementation.
