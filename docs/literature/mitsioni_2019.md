# Mitsioni et al. 2019 — learned cutting dynamics and MPC
Reviewed 2026-11-10. [Paper](https://arxiv.org/abs/1903.03831).

## Evidence and scope
Read the learned model, data collection, online planner and experiments (PDF pp. 3–6). No author-released implementation was retrieved for this paper; SafeDDC is a different later work.

## Method
A learned recurrent model predicts displacement using position, sensed forces and reference forces. Relative-position blocks avoid dependence on absolute workspace coordinates. The architecture combines feed-forward encoders and recurrence, trained through autoencoding, one-step and then multi-step prediction. Recorded interaction trajectories, not motion-capture demonstrations, supply supervised transitions.

An important approximation is using recorded measured force as a proxy for the action/reference force during training, relying on force tracking. The 2021 sequel explicitly revisits this. This model predicts motion conditioned on force; it is not a calibrated predictor of every future contact-force peak.

Online random-shooting MPC samples bounded reference forces (reported maximum 8 N), predicts the trajectory, scores progress to the cutting height plus sawing/effort terms, and executes the first action at about 10 Hz. Bounding the commanded reference force is not a guarantee that measured impact force stays below that value.

## Data and results
The paper reports 513,587 training samples from cutting interaction, using a prescribed sawing/downward collection procedure and manually selected controller settings. Tests include foods absent from model fitting. Stiff-contact cases show reactive changes/retreat, but the experiments are not evidence for our 50 N component-wise sensor-limit success metric.

## Transfer and limits
Direct precedent for learned cutting dynamics + online MPC and adaptation through force history. Useful design lessons are relative coordinates, explicit reference versus measured force, and multi-step fitting. It is not a pathwise policy-training algorithm and does not validate long-horizon BPTT or calibrated force-risk bounds. A claim that “world models have not been used for cutting” is incorrect even if these authors used the name dynamics model.
