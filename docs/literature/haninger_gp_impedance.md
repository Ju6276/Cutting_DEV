# Model Predictive Impedance Control with Gaussian Processes for Human and Environment Interaction

- Primary paper: [arXiv:2208.07035v2](https://arxiv.org/abs/2208.07035v2), 17 pages, December 2022 version. Read on 2026-11-10: complete main Sections 1–7, equations, experimental setup and discussion. No separate appendix was present in the retrieved PDF.
- Author code linked by the arXiv entry: [gp-mpc-impedance-control](https://gitlab.cc-asp.fraunhofer.de/hanikevi/gp-mpc-impedance-control).
- Code status: the official GitLab host timed out both through web retrieval and direct remote HTTPS/git access. Therefore **paper-read, not code-read**; no commit or executable behavior is claimed.
- Remote paper: `/dataHDD/dong/Research/cutting_literature_20261110/contact_risk/2208.07035.pdf`.

## Problem and model

The robot performs compliant interaction with people and environments whose forces are unknown. The learned quantity is a six-dimensional force/torque field as a function of Cartesian tool pose, with a separate GP associated with each interaction mode. A force/torque sensor measures the combined interaction; it is not an independently observed material constitutive state.

A small number of kinesthetic demonstrations (up to three in the reported setting) supplies pose–wrench pairs. Known Cartesian admittance dynamics advance the robot response; the GP supplies force means, variances and local derivatives. MPC selects the reference force and impedance parameters, while an intention/mode estimate accounts for different human behaviors.

This is already a learned-force-model plus uncertainty-aware compliant-control method. It should be cited before claiming that prior model-based control only considers abstract safety costs.

## Optimization and guarantees

The objective combines tracking/ergonomic terms with uncertainty and disturbance-response considerations. Predicted force derivatives enter a damping condition of the form `D >= 2 zeta sqrt(M K_e)`, with local effective contact stiffness obtained from the GP.

The force chance constraint is applied at the **final prediction stage** in the practical formulation to limit computation. It is not a bound on every substep peak throughout the horizon. Gaussian approximations and the learned force representation are assumptions; the paper does not provide distribution-free protection for arbitrary changed policies or irreversible material damage.

The experiments also demonstrate that the nominal force bound is not a hard peak-force guarantee: transients can exceed it. Increasing damping can be more practically stable than an infeasible chance-constrained solution that repeatedly retreats.

## Experiments and metrics

Tasks include rail assembly, polishing and peg insertion, with compliant robot interaction and measured force. The paper studies contact-position changes, several physical stiffnesses, prediction/control behavior and computational rate. The reported outer-loop rate is approximately 10–20 Hz, with individual configurations around 13 or 22 Hz.

Reported comparisons involve force tracking, damping/disturbance response and task behavior. They are not RL sample-efficiency experiments, and the demonstrations are not a dataset of SAC actions or food-cutting trajectories.

The discussion explicitly identifies difficult stiff-contact effects: very small physical time scales, incorrectly anticipated contact, and force noise not fully described by the GP.

## What transfers to our cutting work

Useful prior structure:

1. Keep known controller dynamics explicit instead of learning their deterministic update unnecessarily.
2. Learn interaction force conditioned on the relevant state/history, while treating measurement and process uncertainty distinctly.
3. Match the safety quantity to the actual constraint: end-of-horizon Gaussian force constraints do not validate interval peak-force safety.
4. Compare against compliant feedback control, not only another learning algorithm.

What cannot be copied as a novelty claim: GP/learned force prediction, uncertainty-aware force constraints, and optimizing compliant control parameters are all present here. What cannot be assumed: a static pose-to-force map remains adequate after material damage or cut separation.

## Remaining limits

The official source exists as an author-provided link, but was inaccessible during this audit; no code verdict, runtime reproduction, or license assertion is made. The paper alone cannot establish the exact implementation of covariance propagation, force-limit handling or optimizer fallback.
