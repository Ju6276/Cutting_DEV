# Kuo et al. — uncertainty-aware contact-safe MBRL
Reviewed 2026-11-10. [Paper](https://arxiv.org/abs/2010.08169).
Read the model, predictive controller, uncertainty construction, simulation/robot evaluations and limitations (PDF pp. 3–7). No official source repository was retrieved.

## Mechanism
A Gaussian-process approximation predicts dynamics with propagated mean and covariance. Probabilistic MPC tightens acceleration/control bounds as predicted position uncertainty increases. Crucially it also translates those bounds in the velocity-opposing direction, preserving deceleration; shrinking the action interval alone can leave a moving robot unable to brake. One-step-ahead planning compensates for online computation delay, and models update between trials.

## Evidence and task
Simulation and real mixing/scooping experiments evaluate tracking and large contact forces/torques while learning. The paper compares force tails/top-percentile quantities and tracking rather than simply showing a smaller model loss. Conservative uncertainty handling can also harm task completion.

## What it does and does not establish
This is already physical contact-risk-aware model-based learning, not only an abstract Safety-Gym cost. However its uncertainty-to-acceleration rule does not directly predict a calibrated next-interval peak wrench, nor does it prove a cutting-force bound under damage and changing material.

## Transfer
Preserve braking authority and measure completed task performance alongside force. A scalar margin that makes every action infeasible or stalls the knife is not a useful control improvement. Do not claim “first model-based RL considering contact force”; a narrower mechanism and comparison are required.
