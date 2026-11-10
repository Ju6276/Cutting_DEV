# Safe Exploration in Continuous Action Spaces

- Paper: [arXiv:1801.08757v1](https://arxiv.org/abs/1801.08757v1), Dalal et al., 2018. Read on 2026-11-10: full Sections 1–8, Proposition 1 and its proof, all environment/ablation details and discussion. No separate appendix in this version.
- Remote paper: `/dataHDD/dong/Research/cutting_literature_20261110/contact_risk/1801.08757.pdf`.
- Author source status: no official implementation verified after checking the paper/arXiv links and title/author/code searches.
- Available **third-party** implementation inspected: [AgrawalAmey/safe-explorer](https://github.com/AgrawalAmey/safe-explorer), commit `828673a0f8d6d41c3df05763707c141f93b3294e`. Its README identifies it as a PyTorch implementation, not the paper authors' release. The implementation differences below must not be attributed to Dalal et al.

## Sections 1–5: learned model and data

The paper learns a local model of each safety signal rather than a complete world model:

`c_i(s_next) ≈ c_i(s) + g_i(s)^T a`.

The neural network predicts an **action-sensitivity vector** `g_i(s)`. It is fitted by squared next-step safety-signal error on arbitrary-action transitions. Thus learning how risk changes with action is already an explicit 2018 mechanism, not a new idea created by applying BPTT to a force head.

Offline data are collected with random actions from random starts, ending on a time limit or violation. The model is pretrained once per task; continued fitting was considered but did not improve the reported experiments. The paper does not require expert successful demonstrations.

The key assumptions are a useful local linearization in action, sufficiently smooth/short-time dynamics, a measurable safety signal, and an action that can still prevent the next violation. Irreversible high-speed impacts and delayed force peaks need additional care.

## Section 6: action correction and limits of the closed form

A safety layer projects the proposed action onto learned linear constraints while minimizing squared action change. For a single active constraint the correction is analytic, using the positive Lagrange multiplier and the corresponding sensitivity vector.

The closed form assumes feasibility and no more than one active constraint. General intersecting constraints require a QP or explicit active-set handling; the paper discusses this limitation. Multiple force and torque components at a cutting contact cannot automatically inherit the single-active-constraint result.

The projection consists of dot products and max operations and is differentiable almost everywhere. The paper proposes composition with policy learning. The resulting empirical absence of violations is not a theorem that an inaccurate learned model will protect every unseen physical interaction.

## Section 7: experiments and a key warning

Experiments use synthetic continuous-control Ball and Spaceship tasks in one/three dimensions and corridor/arena layouts, with DDPG and reward-penalty comparisons across ten seeds. The constraints concern position/regions and include safety margins inside the physical domain. These are not fracture/contact-force tasks.

Metrics include violations and task return. The paper reports successful safety-layer behavior where reward shaping still violates constraints.

Section 7.3 makes a particularly relevant observation: **lower held-out prediction MSE need not produce a better safety layer.** Changing the random data-collection action pattern can improve the regression metric yet worsen control. Coverage of relevant action effects matters. This is direct prior evidence for distinguishing predictive accuracy from useful control sensitivity.

## Third-party code: actual implementation, not paper-author evidence

At `828673a0f8d6d41c3df05763707c141f93b3294e`:

- `safe_explorer/safety_layer/safety_layer.py:SafetyLayer._sample_steps` collects random transitions and safety values.
- `_evaluate_batch` fits `c + g(s)^T a` with MSE, one network per constraint; `constraint_model.py:ConstraintModel` outputs an action-dimensional vector.
- `get_safe_action` converts predicted sensitivities to NumPy and applies the single-largest-multiplier correction. It is **not an autograd-preserving safety layer** in this implementation.
- `safe_explorer/ddpg/ddpg.py:DDPG._get_action` applies that modifier only after converting the actor output to NumPy for environment execution.
- `DDPG._update_batch` trains the actor through the critic on the raw actor output; its target path also uses the raw target-actor output. The safety projection is not included in the actor/target training graph. Therefore this repository must not be cited as evidence that the paper's differentiable projection is trained end to end.
- The optional random-start exploration branch bypasses the modifier; `get_safe_action` also divides by `g_i^T g_i` without an explicit zero-denominator guard. These are static implementation limitations, not reproduced numerical failures.
- The environment implementations are separate PyTorch/Gym-era reproductions. No experiment or checkpoint from this repository was executed in this audit.

## Implications for cutting and Route B

A calibrated force upper bound and an action correction are not sufficient novelty by themselves. The literature already has learned constraint sensitivities, projections and evidence that prediction loss can be misleading.

A cutting-specific opportunity would need a concrete mechanism for reliable action effects across contact changes and irreversible damage history, plus evidence that the chosen actions improve physical outcomes. Merely adding a global conformal residual constant to the mean does not repair its derivative with respect to action.

This paper also motivates an honest comparator: a simple learned local constraint-response model can be an informative baseline against a much larger WM. Whether it is adequate for the current cutting interval is an empirical question, not a reason to silently replace the existing task or force definition.
