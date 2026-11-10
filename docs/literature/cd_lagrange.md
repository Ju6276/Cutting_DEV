# Learning Contact Dynamics using Physically Structured Neural Networks
Reviewed 2026-11-10.

## Sources and scope
- [AISTATS 2021 paper](https://arxiv.org/abs/2102.11206): methods/experiments pp. 2–7 and the contact derivation in the appendix; not every appendix proof rederived.
- [Official implementation](https://github.com/libeanim/contact-symplectic-integrator-network/tree/1a7f27d5821549c91ef7bed818364c6f6535b36a), pinned 1a7f27d5821549c91ef7bed818364c6f6535b36a.
- Read `models/cd_lagrange_network.py` and the labelled-training versus prediction paths identified in NOVELTY. Not executed.

## Algorithm
A neural potential models smooth forces; a contact activation model and an impulse/restitution calculation model nonsmooth velocity changes. A contact-aware symplectic step combines these components. State prediction loss and supervised contact BCE train the model. Bouncing-ball and Newton's-cradle tests use idealized contact information; missing touch labels makes separation of contact, observation noise and smooth dynamics difficult.

## Actual code
The TensorFlow model builds potential and contact networks and a smooth-plus-impulse step. Training supplies contact labels. At prediction, absent labels are replaced by a hard decision using `ctf.numpy() > 0.5`, then a constant tensor; branch selection uses hard conditions. Consequently the supplied activation path does not preserve action gradients through contact onset. Paper-level “differentiable model” must not be confused with every inference decision being differentiable.

## Relevance
Separating continuous response from event updates is a useful prior, not our novelty. Direct reuse would require different mechanics for compliant deformable cutting and a declared treatment of event derivatives. It does not justify silently inserting straight-through gradients, thresholds or a soft-contact surrogate into our task. Its evidence is dynamics prediction, not safe BPTT policy training across materials.
