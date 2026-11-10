# DiSECt — direct physical trajectory optimization
Reviewed 2026-11-10.

## Sources and scope
- [Journal paper](https://arxiv.org/abs/2203.10263): cutting mechanics pp. 4–7, trajectory optimization and robot validation pp. 15–20, parameter/example details pp. 29 and 34. The complete calibration-results section and every appendix table were not exhaustively audited.
- [Official source](https://github.com/NVlabs/DiSECt/tree/ff90165070335bb8033000373d2d20c6aac13edd), pinned ff90165070335bb8033000373d2d20c6aac13edd.
- Read `examples/optimize_slicing.py`, `examples/parameter_inference.py` and `cutting/motion.py`'s `SlicingMotion`. Reading is distinct from our previously logged failed/successful execution attempts.

## Physical model and learning/optimization
Tetrahedral FEM with stable Neo-Hookean elasticity and prepared virtual-node cut interfaces represents the deforming material. Contact weakens cutting springs on that prepared interface. SDF contact/penalty forces couple the knife. Parameter inference uses measured cutting response; the paper discusses point estimation and uncertainty methods.

Direct trajectory optimization differentiates the complete physical rollout with respect to parametrized knife motion. Smooth trajectory parameters encode downward motion and lateral sawing. MDMM handles trajectory constraints while gradient updates reduce mean cutting force. This is offline trajectory optimization, not a learned WM, SAC or an already implemented online MPC controller.

## Concrete source differences
The example uses a public sphere/apple configuration, 0.6-s rollout, 2e-5-s material step and 100 optimization iterations. It initializes a five-waypoint sawing motion and invokes `cut()` preprocessing before optimization; `mean_force.backward()` supplies trajectory gradients. The example's single sine frequency and settings must not be silently equated with every equation/configuration in the paper.

## Results and limits
The simulated example reports approximately 89.7 to 76.7 N mean force; real apple/banana/cucumber experiments report substantially larger relative reductions. Those robot results use their own calibrated scenes, attachment and execution, not our current prism. They do not guarantee convergence on every public mesh or our modified configuration. A prepared interface limits where new cuts can be optimized.

Useful role: per-scene physical optimization reference, with explicit geometry/trajectory/robot-interface differences. It does not replace a feedback policy over unknown materials and should not block the current shared-policy research route.
