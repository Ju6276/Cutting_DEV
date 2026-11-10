# Lower Expectile Q-learning (LEQ)
Reviewed 2026-11-10.

## Evidence
[Paper](https://arxiv.org/abs/2407.00699), method §§4.1–4.4, experiments/limitations §§5–6, Appendix A training details. [Author code](https://github.com/kwanyoungpark/LEQ/tree/b5f8d43577108a0e5876279a1c8ac813b830f0d7), pinned b5f8d43577108a0e5876279a1c8ac813b830f0d7; read `algos/leq/actor.py` (lambda/multistep paths), `critic.py:lambda_update_q`, and learner update dispatch. No execution; proofs not independently rederived.

## Algorithm
Offline model-based RL with ensemble dynamics, lower-expectile conservative value targets, lambda returns, and pathwise policy improvement. Real-transition Bellman losses mix with imagined losses; an EMA critic regularizes prediction drift. BC and FQE initialize policy/value for state-based experiments. The actor implementation computes action Jacobians of imagined lambda returns using `jax.jacrev`, then combines them with policy derivatives and asymmetric expectile weights. The target critic regularizes values; current critic values construct bootstrap targets in the inspected lambda path.

## Results and caveats
D4RL/NeoRL/visual benchmarks measure returns or maze success across seeds. The paper explicitly reports abrupt collapses when models miss falling/terminal events and imaginary wall-crossing. §6 assumes access to true task termination; replacing it with a learned function substantially degrades the reported maze aggregate. Even “true termination” is applied to predicted states and therefore cannot repair wrong predicted geometry. H=10 outperforms both shorter and longer horizons in the aggregate, not universally.

## Relevance
Real data in value learning, short imagination, BC initialization and termination failures all have close precedents. LEQ itself does not eliminate model exploitation. It is offline and uses a different conservative objective; copying one of its settings is not a proof that our online cutting critic is repaired or that H8 is theoretically optimal.
