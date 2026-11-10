# DeepMPC — learned cutting dynamics before current “world model” terminology
Reviewed 2026-11-10.

## Sources and scope
- [Author-hosted extended paper](https://www.cs.cornell.edu/~ianlenz/DeepMPC.pdf), central model/training/controller and robot experiments pp. 5–13. Distinguish this extended version from the shorter RSS paper.
- No algorithm repository or dataset release was verified; paper evidence only.

## Model and controller
The recurrent model represents latent material response using recent interaction history, future commands and multiplicative features. It predicts the response to candidate control sequences without requiring a supplied material identity. Training proceeds through feature pretraining, short prediction and warm-history multistep fitting. AdaGrad optimizes a future command sequence online against downward progress, sawing range and smoothness/control costs; only the immediate command is executed before replanning.

The study collected 1,488 cuts across 20 materials using fixed, tuned and randomized cutting controllers. Ten 10-ms samples form a model block; the reported planning horizon is 0.5 s. Real PR2 tests cover hundreds of trials, material/temperature and knife changes. These are learned dynamics + receding-horizon control, regardless of whether the authors use the modern phrase “world model.”

## Metrics and limitations
Metrics include prediction quality and cutting rate/task performance on held-out conditions. Command limits (reported force commands bounded per axis) are not a guarantee on measured contact peaks. The unchanged controller transfers to new temperature conditions and several
straight knives, but fails with the tested serrated knife. Generalization differs by knife/material, and the paper is not a calibrated uncertainty-bound or BPTT policy-learning study.

## What it rules out
Learning a cutting response from history, adapting to multiple materials without an explicit material label, and using a learned model for cutting MPC are established. A compact history-conditioned model is not automatically our novelty. This is an essential comparison when motivating material adaptation; it need not be faithfully reimplemented before every engineering step.
