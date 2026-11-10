# Model-Augmented Actor-Critic (MAAC)
Reviewed 2026-11-10.

## Sources and scope
- [Paper](https://arxiv.org/abs/2005.08068): formulation, gradient-error argument and main experiments, pp. 2–8; the same-model MPC comparison is in §5.4. Appendix proofs were not independently rederived.
- No author implementation was verified in this review; no code-read claim.

## Algorithm
A stochastic policy is optimized through a short learned-model rollout, followed by a discounted endpoint Q estimate. Pathwise derivatives flow through the model and value function. Model ensembles, entropy and value learning provide the wider actor–critic setup. The paper evaluates return versus physical interaction and compares execution-time MPC.

The gradient-error analysis depends on errors in model/value derivatives and smoothness assumptions. Small state MSE alone does not establish accurate policy gradients. The horizon trades model-induced error against reliance on endpoint value.

## Implications
Short learned rollouts + endpoint value + BPTT, and comparing their learned policy with model-predictive execution, are already explicit precedents. Neither is a priority claim for this project. Our hard terminations, hidden material state and contact transitions do not automatically satisfy the theoretical assumptions.

The most useful methodological question is whether the model gets the *effect of candidate actions* on future force and task progress right. That is narrower than merely fitting states, but a new action-aware objective would still require comparison with decision-/value-aware model-learning literature. MAAC does not itself provide cutting data or a force-risk guarantee.
