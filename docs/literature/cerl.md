# CERL: Safe Exploration Using Bayesian World Models and Log-Barrier Optimization

> Project interpretation updated 2026-11-19; original paper/code reading scope is unchanged.
> Superseded project hypotheses are [archived](../archive/LITERATURE_POSITIONING_20261110.md).
Reviewed 2026-11-10. [Paper v1](https://arxiv.org/pdf/2405.05890).
Read all eight pages: §§1–6, assumptions, experiments and limitations.
## Paper
- §§3–4: partially observed CMDP with a Bayesian RSSM represented by probabilistic ensembles. Select the most pessimistic predicted cumulative constraint across sampled models.
- Backpropagate reward/constraint through model rollouts to the actor. Log-Barrier SGD adapts optimization steps to remain in the estimated feasible interior.
- Formal motivation assumes the true model is in the plausible set; feasibility arguments additionally require smoothness, safe initialization and unbiased gradient estimates. Authors explicitly acknowledge these are hard to verify.
- §5: image-based Safety-Gym Point tasks, five seeds, five million interaction steps, 1,000-step episodes and cost budget 25. Report return and cumulative training costs.
- §6 explicitly distinguishes expected episode-cost constraints from state-wise safety.
## Code availability
Paper points to https://anonymous.4open.science/r/safe-opax-F5FF/ rather than a versioned GitHub release.
Repository retrieval/commit and executable-core inspection remain pending; no code-derived claims or reproduction claim are made.
## Relevance to the current method

Pessimistic model costs, BPTT and constrained policy updates are prior art. CERL is background, not the selected main algorithm. Its expected-cost constraint does not establish per-sample wrench safety in the current task.

Follow-up code access on 2026-11-10: the official anonymous root, README endpoint and repository API returned HTTP 403 from GPU7; no code-read evidence or verified replacement release was obtained.
