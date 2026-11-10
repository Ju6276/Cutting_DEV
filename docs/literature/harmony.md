# HarmonyDream — balancing world-model objectives
Reviewed 2026-11-10.

## Sources
[ICML 2024 paper](https://proceedings.mlr.press/v235/ma24o.html), §§2–4 and discussion §6; correct arXiv ID is 2310.00344.
[Official code](https://github.com/thuml/HarmonyDream/tree/0a22e33fb47489361a9f239867a1cf34b958b320), pinned 0a22e33fb47489361a9f239867a1cf34b958b320.
Inspected `wmlib-torch/wmlib/agents/dreamerv2.py` optimizer registration and complete harmonized-loss branch. No runtime reproduction; JAX variant not exhaustively audited.

## Mechanism
Observation, reward and dynamics losses can dominate one another. Learn positive scale parameters sigma=exp(s), optimize loss/sigma + log(1+sigma) per harmonized task. The rectified log term avoids extreme inverse-loss weights. This balances loss scales, not directly gradient norms or directional conflicts.

Code registers three trainable scales for reward/image/KL. Discount/continuation remains separately weighted. Head gradient flow is selected through `grad_heads`; harmonization is not a general detach of auxiliary heads.

## Evidence and implications
Meta-World, RLBench, distracted DMC and game experiments report success/return versus environment steps and ablations. Observation domination can create wrong action–object–reward relations, even with plausible reconstructions. Clean tasks have smaller gains.
Our BCE-gradient dominance concerns supervised physical state and termination, not identical image-loss dominance. Reweighting would be an attributed comparator, not a new contribution by itself; equal losses do not imply equal parameter gradients.
