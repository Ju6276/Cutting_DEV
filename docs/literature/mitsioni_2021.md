# Mitsioni et al. 2021 — recurrent cutting dynamics
Reviewed 2026-11-10. [Paper](https://arxiv.org/abs/2003.09179).

## Evidence and scope
Read architecture, training curriculum, MPC, data, experiments and accompanying implementation details (PDF pp. 3–7). No official source implementation was retrieved. Do not substitute SafeDDC code for this model.

## Method and difference from 2019
The model separates commanded reference force from sensed force. Each block contains position/displacement and both force quantities; a 90-unit input layer feeds two LSTM layers with hidden size 9 and an output layer predicting 30 displacement values. Ten 5 ms samples form a 50 ms block. Training gradually increases the prediction horizon rather than fitting only a one-step loss.

MPC samples 25 candidate controls with a five-block (0.25 s) horizon. It optimizes predicted cutting motion, not a learned policy via BPTT. The real platform is YuMi with force sensing.

## Data, metrics and caveats
The paper uses 34 training trials/210,564 samples and 15 validation trials/93,447 samples, with six training foods and held-out cheese, potato and lemon. It evaluates prediction quality, executed control costs and qualitative cutting behavior. Its discussion makes clear that prediction accuracy alone does not determine control quality.

Trials stopped by the hardware torque limit were repeated/discarded, and some difficult items used an initial slit. These details matter: the resulting cutting table is not a complete all-starts force-safety evaluation. Do not transplant its success rate into our failure accounting.

## Relevance
Strong precedent for history-based cutting dynamics and multi-food MPC. It supports a same-model planner comparator and honest separation of command force, sensed force and realized motion. It does not resolve within-control-interval impact peaks, calibrated risk under a changing policy, or a shared BPTT-trained controller's benefit.
