# Learning on the Fly — hybrid forward model and surrogate derivatives
Reviewed 2026-11-10.

## Sources
[RA-L paper](https://arxiv.org/abs/2508.21065), method §§III-A–F, evaluation/ablations §§IV-A–C.
[Official source](https://github.com/uzh-rpg/learning_on_the_fly/tree/cba6e5370773ace8a08107f02810eecabf16c793), pinned cba6e5370773ace8a08107f02810eecabf16c793.
Inspected `lotf/objects/quadrotor_obj.py:step/_step_jvp`, `lotf/utils/residual_dynamics.py` fitting, and hovering reward termination path. No execution.

## Mechanism
Deploy a pretrained quadrotor policy at 50 Hz, continually collect real interaction, fit an ensemble residual acceleration model, and adapt policy through differentiable rollouts. Analytical rigid-flight dynamics supply the nominal model. Paper examples refit residuals every 3 s and update the policy every 5 s. References here are desired flight trajectories, not training transitions.

The source uses a **custom JVP**: forward simulation can include residual/high-fidelity dynamics, while backward p/R/v derivatives come from simplified analytical dynamics. Thus its policy gradient is deliberately a surrogate, not the exact derivative of every forward component. Residual fitting uses MSE and a matrix-norm regularizer.

## Evidence and limits
Hover/trajectory errors, update speed, LoRA/full adaptation and dynamics-fidelity ablations on quadrotors. Visual features in robot tests use motion-capture-derived keypoints in a hardware-in-loop setup. Timings and smooth-flight assumptions do not transfer to discontinuous cutting.

## Relevance
Known dynamics + learned residual + BPTT is established. Surrogate gradients can be an explicitly evaluated method, but must not be silently added to our model to make a test pass. This is a useful reference for separating forward accuracy from derivative quality; it is not a cutting-force solution.
