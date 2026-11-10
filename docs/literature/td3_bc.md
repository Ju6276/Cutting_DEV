# TD3+BC — minimal algorithm, substantial empirical evidence
Reviewed 2026-11-10.

## Sources
[Paper](https://arxiv.org/abs/2106.06860), main text pp. 2–9 including instability, method, evaluation and ablations.
[Official code](https://github.com/sfujim/TD3_BC/tree/8791ad7d7656cb8396f1b3ac8c37f170b2a2dd5f), pinned 8791ad7d7656cb8396f1b3ac8c37f170b2a2dd5f; `TD3_BC.py` read in full. No training.

## Method/code
Keep TD3's twin critics, target smoothing, delayed actor updates and Polyak targets. Add replay-action squared error to the actor objective and normalize dataset state features. The code sets the RL multiplier to alpha divided by mean absolute Q, detached from gradients; alpha=2.5 is the reported default. This scale adaptation is not simply an unnormalized fixed BC weight.

## Evidence
Offline D4RL experiments use five seeds, learning curves, final evaluation averages, ablations and compute comparisons. The paper explicitly observes variation across episodes/checkpoints even for deterministic policies. Its simplicity is a methodological argument backed by comparisons, not a theorem that every simple combination is novel or stable.

## Our boundary
Replay regularization has an explicit precedent; our online actor's model-return/BC objective is an adaptation, not original offline TD3+BC. Do not describe our replay as expert demonstrations. A small method can be publishable, but must isolate a real mechanism and show evidence; adding complexity is not a novelty requirement.
