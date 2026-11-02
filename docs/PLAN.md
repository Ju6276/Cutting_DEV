# Training plan

Target: SuperMimic-Reproduced's online learning algorithm on the existing
SliceIt UR5e robot/compliance task, compared with the unchanged SAC baseline.

1. **Task/data interfaces — checked.** A complete 41-step episode matches the
   original observations, rewards, force-limit decisions and termination labels.
2. **Cutting state and analytic heads — checked on recorded transitions.**
   Vertex/joint/knife dynamics and contact signals are learned; URDF kinematics,
   control accumulation and SliceIt observation/reward formulas are analytic.
   Action derivatives and effective-batch gradient accumulation have unit checks.
3. **Connected training — full-size update check passed.**
   Original replay, 8-step prediction, 32-step BPTT, GAE, source batch sizes,
   optimizers, clipping, learning-rate schedule and update ratio. One physical
   environment; microbatches preserve effective batch size on the 4090.
4. **Learning and physical evaluation.** First operational budget: six hours,
   checkpoint on exit; source schedule remains 320000 steps. Inspect completed
   updates, held-out physical returns, success, force-limit failures, released
   springs and runtime. Resume from the checkpoint for further learning.

Short-term loss/reward fluctuations are expected. Small-run interface checks
establish execution and equation consistency, not model quality. Architecture
changes require sustained evidence, a specific cause and a source mapping.
Do not change rewards, task thresholds or source optimization parameters to
improve a short curve. Repair data/gradient/runtime bugs directly.

The initial collector measured 104 transitions in 495 s concurrently with SAC.
Measure the connected learner's throughput before estimating time for a full
320000-step schedule. Long-job progress, checkpoints and wall budgets are
mandatory; no multi-day run is started without that measured estimate.
