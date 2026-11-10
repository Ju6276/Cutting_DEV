# BladeMaster — forward cutting mechanics, separate from WM learning
Reviewed 2026-11-10.

## Evidence/status
[Paper](https://arxiv.org/abs/2609.27342), main method/experiments §§III–IV and conclusion. An official released algorithm repository was not verified in this literature pass. Our historical BladeMaster-Warp reproduction is a separate local implementation, not author-source evidence.

## Mechanics
Reference-space TLMPM stencils remain fixed. Online blade sweeps assign persistent ternary side labels; compatible logical grids exchange momentum while conflicting sides remain separate. Explicit material contact enables recontact without reconnecting. The directional contact-work progress variable modulates resistance; swept geometry determines side labels. The work variable is not itself a continuous constitutive damage law.

Tool–particle impulses produce equal-and-opposite force and torque, accumulated for a lower-rate rigid-body interface. Reference surface reconstruction is display-only. The paper shows partial/intersecting/sequential cuts and manipulation after separation.

## Evidence and limits
Reported faster-than-real-time timings use an RTX5090 and exclude reconstruction/rendering. Banana-force comparisons distinguish material cutting from later support contact. These results are not measured speed of our modified differentiable solver. Persistent discrete side assignment does not become differentiable merely because kernels use an AD-capable language.

## Current project role
Potential future forward data generator; it can support a learned WM without differentiating through its topology. Replacing DiSECt now is outside S0–S4 and would confound the learning comparison. This review neither revives the old solver-rewrite route nor claims it is a fully differentiable cutting system.
