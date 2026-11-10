# STEVE — stochastic ensemble value expansion

Reviewed 2026-11-10. [Paper, 1807.01675](https://arxiv.org/abs/1807.01675).
Read scope: §§2–6, ensemble-target construction and main/ablation experiments;
appendix proofs not independently verified.
Official code was removed from current TensorFlow Models. Inspected its archived
[research/steve at 5eb294f84bd3f415b548980e69fee63db1f6f1df](https://github.com/tensorflow/models/tree/5eb294f84bd3f415b548980e69fee63db1f6f1df/research/steve),
the parent of the removal commit. No training executed.

## Mechanism and experiments

Dynamics, reward and Q ensembles generate value targets for several rollout
lengths. Inverse estimated variance weights those targets per example.
The default derivation neglects cross-target covariance and unknown bias:
agreement is not a calibrated error bound. DDPG learns a deployed policy;
its actor differentiates Q, rather than using our WM BPTT objective.

Main experiments report return versus physical samples over four runs;
ablations include uniform weighting, TD-lambda and covariance-aware weighting.
Wall-time experiments use separate model/value learners and parallel collection.
These are locomotion benchmarks, not force-limit or cutting evidence.

## Code evidence

`valuerl.py::build_training_graph` uses stopped value targets and actor
`-Q(s, policy(s))`. `build_Q_expansion_graph` combines reward, survival
probability and old-Q estimates, then computes ensemble variance and
`1/(variance+1e-8)` confidence. The optional TD-k loss weights imagined states
by reach probability.

`worldmodel.py::transition` predicts a delta state, then predicts termination
from predicted-next-state plus current state/action. The fitting code includes
state, reward and termination losses; the terminal network is not inherently
independent of dynamics gradients. Separate learners reload model checkpoints.

## Implication

Useful precedent for uncertain value targets and model/value coupling.
Neither ensemble weighting nor a termination head guarantees accurate cutting
force peaks. Do not call our physical lambda-return critic STEVE or infer that
a narrow ensemble can certify an unseen contact event.
