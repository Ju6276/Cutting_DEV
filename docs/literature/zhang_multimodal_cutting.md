# Leveraging Multimodal Haptic Sensory Data for Robust Cutting
Reviewed 2026-11-10.

## Sources/scope
[Humanoids 2019 paper](https://arxiv.org/abs/1909.12460), setup, event monitoring, DMP adaptation, data collection and experiments §§III–VIII. No complete author algorithm/data repository verified in this pass; a sound-device utility is not the full release.

## Algorithm/data
Two Franka arms hold a knife and tongs. Four contact microphones and robot force/torque feedback provide synchronized features. SliceNet detects six interaction/event classes; FoodNet classifies material and selects human-tuned DMP amplitude/downward-displacement settings. A regression variant predicts these settings continuously for held-out material evaluation. Kinesthetic demonstrations fit the DMP forcing function. This is adaptive imitation/event monitoring, not WM planning or RL.
Force history contributes 60 features per 0.1-s window; audio descriptors/MFCCs supply vibration information. The dataset includes 25 labelled classes including aged variants.

## Metrics and limitations
Report event/material F1, parameter error under held-out classes and real slicing time (five trials per item). Force alone performs substantially worse for material classification than vibration; it is more useful for event detection. Hard objects/cores remain limitations. The DMP parameters are manually chosen useful values, not optimal solutions. Classification F1 is not a calibrated force-peak or force-limit safety metric.

## Implication
Adapting cutting behavior from contact history across many foods predates this project. It supplies a realistic sensing/data reference and warns against equating material classification with causal response prediction. Adding a phase classifier alone would not establish a new method.
