# ContactNets — structured contact identification
Reviewed 2026-11-10.

## Sources and reading scope
- [Paper, CoRL 2020](https://arxiv.org/abs/2009.11193): central formulation, identification objective, experiments and limitations, PDF pp. 2–8. This is a method/result read, not verification of every appendix proof.
- [Official code](https://github.com/DAIRLab/contact-nets/tree/c3dc21c94577ff4ed99123f0b00bfe57bb2d1d90), pinned c3dc21c94577ff4ed99123f0b00bfe57bb2d1d90.
- Inspected `contactnets/train/poly_ground_3d_surrogate.py`, particularly `compute_loss`; the forward LCP distinction is also recorded in NOVELTY. No training executed.

## Method, data and metrics
Known smooth rigid-body dynamics determine the momentum change that contact must explain. Networks learn signed distances/contact Jacobians. The normal Jacobian derives from the distance function rather than being an unrelated output. An inner convex QP fits latent contact impulses using impulse matching, complementarity, nonpenetration and friction/maximal-dissipation terms; differentiation of the fitting objective avoids simply regressing an impact-discontinuous next-state map.

The real experiment uses tracked cube tosses (750 collected, 570 retained), with camera-based pose tracking and compliant corner pads. Evaluation includes long-rollout position/orientation errors and penetration, rather than force-limited learned cutting control.

## Code correspondence
The surrogate constructs the observed impulse from velocity change minus known gravity/control, forms normal/tangential contact terms and solves a QP through `solve_qp_osqpth`. This is the identification gradient. The provided LCP prediction solver detaches into NumPy/Lemke; it is not automatically a differentiable rollout for actor BPTT.

## What it establishes and what it does not
Coupling contact geometry, impulse and state change is established prior art. It is a strong design reference for avoiding independent inconsistent motion/force predictions. However, a rigid inelastic impulse is not our filtered, compliant, deformable-cutting sensor peak. Known smooth dynamics, contact assumptions and solver gradients must be revisited before reuse. This is not evidence for calibrated force-risk bounds or our policy's control advantage.
