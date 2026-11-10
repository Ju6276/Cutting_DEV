# Constrained Policy Optimization

- Paper: [arXiv:1705.10528v1](https://arxiv.org/abs/1705.10528v1), ICML 2017. Read on 2026-11-10: full main Sections 1–9 and Appendix Sections 10.1–10.3, including performance-bound derivations, optimization solution and experiment details.
- Official implementation: [jachiam/cpo](https://github.com/jachiam/cpo), linked by the [authors' BAIR article](https://bair.berkeley.edu/blog/2017/07/06/cpo/). Code pinned to `5c83925a84bc21dfc9f8bceb82f85daf2937e595`.
- Remote paper/source: `/dataHDD/dong/Research/cutting_literature_20261110/contact_risk/1705.10528.pdf` and `contact_risk/code/cpo`.
- Scope: paper derivations read, not independently reproved; central optimizer/policy/safety-predictor source inspected, not executed.

## Main Sections 1–4: objective and guarantee

CPO maximizes expected discounted reward under bounds on expected discounted auxiliary costs in a CMDP. This differs from requiring every state, every force component or every material substep to remain safe.

The analysis bounds the change in expected performance using policy divergence. The ideal update constrains a surrogate cost plus a divergence-dependent error term. The practical algorithm uses locally linear reward/cost surrogates and a quadratic KL trust region.

The practical constrained update solves a quadratic approximation in policy-parameter space. With one cost constraint, the dual has an analytic solution. Conjugate gradient approximates inverse-Fisher products and a line search checks the sampled nonlinear surrogates. When the local feasible problem fails, a recovery update prioritizes reducing cost.

The theoretical statements rely on specified policy-divergence/advantage bounds. Finite-sample estimates, local approximation and neural optimization mean the experimental algorithm is not a universal hard-safety controller.

## Main Sections 5–8: experiments and learned safety shaping

Experiments use Point, Ant and Humanoid navigation/locomotion tasks, including Circle and Gather constraints. Reported metrics are return, average constraint cost and learning behavior, across several random seeds. These are not food cutting, material-force prediction or BPTT-through-WM experiments.

A useful precedent is **learned constraint shaping**: a network estimates near-future unsafe events and adds a nonnegative safety bonus to the cost. Therefore “learn a failure predictor and use it for safer policy optimization” is already present in this early work. Its prediction does not have conformal coverage and is not a dynamics model.

The appendix describes GAE, discounts, network/trust-region settings and cost-gradient estimation. These implementation details matter: a safety-cost advantage with the wrong scale changes a constrained update, even if an unconstrained policy gradient would only care about direction.

## Official code and actual gradient flow

At commit `5c83925a84bc21dfc9f8bceb82f85daf2937e595`:

- `algos/safe/cpo.py:CPO.__init__` instantiates `ConjugateConstraintOptimizer` and enables the constrained policy-gradient path.
- `algos/safe/policy_gradient_safe.py:PolicyGradientSafe.init_opt` constructs `likelihood_ratio * advantage` reward and safety surrogates and an average policy KL. Gradients are with respect to policy parameters from sampled observations/actions. There is **no gradient through the environment or a learned physical transition model**.
- `PolicyGradientSafe.optimize_policy` supplies the sampled safety-cost estimate and explicitly rescales the safety surrogate gradient from a state-action mean to a trajectory-cost scale.
- `optimizers/conjugate_constraint_optimizer.py:PerlmutterHvp` computes Hessian-vector products of KL. `ConjugateConstraintOptimizer.optimize` computes reward/safety gradients, conjugate-gradient solves, feasible/infeasible cases, dual multipliers, a recovery direction and backtracking. The line search checks sampled reward, KL and safety surrogates; recovery can prioritize constraints rather than reward.
- `algos/safe/sampler_safe.py:process_single_batch` computes reward/cost returns and GAE; `create_samples_dict` computes `safety_eval` and `safety_rescale`. The code also logs raw undiscounted safety cost separately from the optimized statistic.
- `safety_constraints/predictor_bonus_base.py:ParameterizedBonusSafetyConstraint` fits a sigmoid network using squared error. `prepare_inputs` labels whether a violation occurs within `lag_time` future samples; `get_bonus` adds a scaled nonnegative prediction. It does not calibrate a physical force upper bound.
- `experiments/CPO_point_gather.py` fixes one example's policy, reward/cost baselines, cost budget and training settings. Other paper tasks use their own settings; these values are not proposed replacements for the cutting configuration.

## Implications for Route B

CPO is a relevant constrained-RL comparator or conceptual baseline, not a drop-in WM+MPC module. Constraining average cost is not equivalent to predicting and avoiding a peak force event.

A method that replaces the CPO cost with force still needs to state whether it constrains expected accumulated cost, probability of any violation, a filtered sensor component, or an interval maximum. Conformal coverage does not turn CPO's expected constraint into a hard guarantee.

A worthwhile cutting mechanism must address what the model knows about changing contact/material history and how its action sensitivity affects control. The existence of this paper rules out treating “separate safety cost plus policy optimization” as a new idea.
