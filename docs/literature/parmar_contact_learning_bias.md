# Fundamental Challenges in Deep Learning for Stiff Contact Dynamics

- Primary paper: [Parmar, Halm, Posa, arXiv:2103.15406v1](https://arxiv.org/abs/2103.15406v1), eight pages. Read on 2026-11-10: main Sections I–VI and Appendices I–III, including experimental definitions and recurrent/residual-model discussion. This note does not claim the proposed remedies were implemented by the authors.
- Official code: [DAIRLab/ContactLearningBias](https://github.com/DAIRLab/ContactLearningBias), inspected commit `65d23dd0ef2e7c39f517cc771ea74ce7b968d6a2`.
- Remote evidence: `/dataHDD/dong/Research/cutting_literature_20261110/contact_risk/2103.15406.pdf` and `code/contact_bias/`. No experiments were run.

## What the paper actually establishes

The study separates three sources of prediction error: sensitivity to noisy initial observations, approximation/training error, and finite-data generalization. Increasing contact stiffness makes otherwise similar learning problems harder; the authors do not conclude that simply adding a recurrent network solves contact.

The prediction target is rigid-body motion, not food fracture or measured cutting force. The central experiment tosses a cube in MuJoCo with several contact-stiffness settings. The state has position, orientation, linear velocity and angular velocity (13 numbers with quaternion orientation). Networks predict velocities; position/orientation are advanced with known integration. The comparisons include fully connected networks and a GRU with a history window of 16. Datasets contain 10,000 training trajectories and 1,000 test trajectories per setting; noisy observations and different training-set sizes are controlled separately.

Both one-step errors and 50-step recursive rollouts are reported. A recurrent model can reduce one-step error while still producing worse long rollouts for hard contact. At high stiffness the sample requirement changes substantially, with roughly two orders of magnitude differences in some comparisons. These are task-specific findings, not a universal scaling law.

## Mechanism and assumptions

A small error in contact timing changes which regime the system is in. A smooth model trained by ordinary pointwise regression is asked to approximate a sharply changing transition map. Sensor noise therefore changes the learning target even when the simulator itself is deterministic.

The discussion of residual learning is especially relevant: a learned residual may have to cancel an incorrectly positioned contact discontinuity in the nominal model. Adding a physics term is consequently not automatically beneficial. Implicit/contact-structured learning is discussed as a possible remedy, but this paper is an analysis rather than a validated cutting-control solution.

## Code reading and actual computation

Read the full executable logic of `RNNPredictor.py`, `train.py`, `learning_utils.py`, `eval_utils.py`, and `data-scripts/generate.py` at the commit above.

- `getRecurrentDataset` forms histories from CSV trajectories and prepares the velocity prediction targets.
- `train_predictor` fits the predictor with MSE and Adam.
- `rollout_trajectory` repeatedly feeds predictions back and advances the pose analytically. Evaluation uses NumPy / no-grad paths.
- No policy, reward, force-limit classifier, actor BPTT, conformal calibration or MPC is implemented in the inspected experiment.
- The README links a public Google Drive dataset and the generation script creates cube-toss CSV trajectories. These are simulated state sequences, not robot cutting demonstrations.

Static portability issues were also visible: the MLP training branch refers to `MLPPredictor` without the matching import in the inspected `train.py`, and scripts contain author-specific filesystem paths. These observations are not a runtime reproduction and do not invalidate the reported paper results.

## Relation to Route B

This is strong evidence against treating our force-peak error as merely an insufficient loss weight or an MLP capacity issue. Contact-time uncertainty, partial observations and force filtering can matter before calibration is considered.

Useful borrowing: report contact-onset timing and short recursive rollout behavior separately from aggregate one-step MSE. Do not soften our simulator or lower contact stiffness merely to improve the model fit.

Not supplied by this work: a calibrated force bound, a safe policy update, or a proof that a particular cutting representation fixes the problem. It motivates examining whether the model gives the correct *action-dependent change* in force, rather than only the correct average force.

## Remaining limits

The code was read, not executed. Other branches, downstream reproductions and every external reference cited by the authors were not audited. No claim is made that this paper is the only work studying contact-learning bias.
