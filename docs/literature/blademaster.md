# BladeMaster — forward cutting mechanics, separate from WM learning
Reviewed 2026-11-10.

> Project interpretation updated 2026-11-19; original paper/code reading scope is unchanged.
> Superseded project hypotheses are [archived](../archive/LITERATURE_POSITIONING_20261110.md).

## Evidence/status
[Paper](https://arxiv.org/abs/2609.27342), main method/experiments §§III–IV and conclusion. An official released algorithm repository was not verified in this literature pass. Our historical BladeMaster-Warp reproduction is a separate local implementation, not author-source evidence.

## Mechanics
Reference-space TLMPM stencils remain fixed. Online blade sweeps assign persistent ternary side labels; compatible logical grids exchange momentum while conflicting sides remain separate. Explicit material contact enables recontact without reconnecting. The directional contact-work progress variable modulates resistance; swept geometry determines side labels. The work variable is not itself a continuous constitutive damage law.

Tool–particle impulses produce equal-and-opposite force and torque, accumulated for a lower-rate rigid-body interface. Reference surface reconstruction is display-only. The paper shows partial/intersecting/sequential cuts and manipulation after separation.

## Evidence and limits
Reported faster-than-real-time timings use an RTX5090 and exclude reconstruction/rendering. Banana-force comparisons distinguish material cutting from later support contact. These results are not measured speed of our modified differentiable solver. Persistent discrete side assignment does not become differentiable merely because kernels use an AD-capable language.

## Current project role
The project now includes a BladeMaster-style forward material backend in the shared UR5e training framework, alongside DiSECt. The learned WM is trained from backend transitions; the policy update does not differentiate through the material solver. Backend integration is an engineering implementation, not evidence that the paper's results or unrestricted cutting capabilities have been reproduced. Paper mechanics, local implementation checks, and task outcomes must be reported separately; see [shared scene](../SHARED_SCENE.md) and [current results](../RESULTS.md).
