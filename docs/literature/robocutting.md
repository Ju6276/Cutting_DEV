# Luo and Yuan — perception/manipulation food cutting
Reviewed 2026-11-10. [Paper](https://arxiv.org/abs/2607.04367), [project](https://robotproject8.github.io/robocutting/).

## Evidence and scope
Read the real system, classification, PPO formulation and robot experiments (PDF pp. 2–7). A released training repository/dataset was not retrieved; project availability alone is not code/data release.

## Method
A UR5e with wrist force/torque sensing uses a knife-selecting trial cut, then learned cutting control. An SVM classifies trial-cut force features to choose between a fruit and serrated knife. PPO operates at 10 Hz above a 100 Hz force/position controller, changing desired downward force and sawing speed. The six policy features include velocities, measured forces and current reference commands. The policy is not image-driven.

The downward reference is bounded to [-30,0] N and saw speed to [0,2] cm/s. These are action bounds, not a guarantee on contact peaks. Reward combines downward progress with a mechanical-work/force-displacement term. The paper discusses boundary-seeking control as a consequence of that objective.

## Experimental interpretation
The trial-cut classification dataset has 14 foods, multiple knife/motion conditions and five repeats (420 trials). The RL studies include per-food from-scratch learning and knife-specific pretrained controllers subsequently adapted for new foods. Do not describe this as one frozen universal policy across all materials. The human comparison uses separately instrumented motion/force; its camera/AprilTag measurements are not policy observations.

## Relevance
Very close real-robot RL precedent: force-aware compliant control and multi-food adaptation are already demonstrated. Distinguish fresh training, fine-tuning and frozen generalization. Our proposed contribution cannot be merely applying RL to a UR5e food task, and simulation-only results must not be presented as equivalent to their hardware validation.
