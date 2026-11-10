# Data-efficient Model Learning and Prediction for Contact-rich Manipulation Tasks
Reviewed 2026-11-10.

## Sources
- [Khader et al., RA-L 2020](https://arxiv.org/abs/1909.04915): methods, prediction algorithm, assumptions and robot experiments, pp. 3–7.
- [Author code](https://github.com/shbz80/model_learning/tree/5e3823e534714c008e146eecda82a5b21c43db39), pinned 5e3823e534714c008e146eecda82a5b21c43db39. This review found and inspected the implementation, upgrading the previous paper-only entry.
- Read `model_leraning_utils.py` transition-model/SVM training and `model_learning_core_yumi.py` transition fitting and probabilistic propagation. No execution.

## Method
DPGMM clustering identifies motion/contact modes. Each mode has a GP dynamics model; transitions have separate reset-map GPs, and an SVM predicts guards/next modes. Prediction propagates state uncertainty, samples mode transitions, splits probability mass and merges compatible Gaussian branches. A reset map models the jump; it is more than a mixture of smooth regressors.

YuMi robot experiments include contact with wood/steel and varying control gains. The study compares predictive error/likelihood and data requirements (15 versus 40 training trials with held-out trials), not cutting-policy success or force-bound calibration.

## Code evidence
`train_trans_models` builds input/output pairs where adjacent mode labels differ and fits the transition GP. `SVMmodePredictionGlobal.train` trains next-mode classification. The YuMi rollout samples state/action, predicts allowed guards, normalizes branch mass, chooses within-mode versus reset models and merges branches. These Python 2 research scripts contain experiment-specific paths; they are not a ready GPU/BPTT package.

## Boundary for our use
The rigid, static-environment state assumption does not cover hidden deforming/damaged material. Separating uncertainty in contact time from uncertainty in response is relevant; calling that decomposition new would be incorrect. A cutting extension needs causal material history and measured sensor response, and must demonstrate control benefit rather than only improved trajectory likelihood.
