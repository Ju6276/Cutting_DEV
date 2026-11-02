# Training plan

Target: SuperMimic-Reproduced online learning on SliceIt's UR5e/FDCC and public
DiSECt prism. Keep source objectives, effective batches, optimization schedule,
rewards and termination thresholds.

1. **Coupled GPU interfaces — checked.** Batched robot, material, controller and
   sensor history; independent reset and frozen terminal worlds. A6000 and 4090:
   backend 19 tests and model 27 tests pass.
2. **Source learning loop — connected.** Original 8-step prediction, 32-step BPTT,
   GAE, 2048/4096 effective learning batches, 40-step warmup and UTD 0.4.
   The GPU state now includes controller/filter recurrence (8944 coordinates).
3. **Training scale — 1024 worlds.** The 1280-world full reference cut passed,
   but training exhausted memory when switching from a learning update back to
   MuJoCo collision processing. Resume the saved learner at the user-selected
   1024 worlds, retaining original replay capacity and learning batches. Verify
   repeated physics/learning transitions during training.
4. **Learning and physical evaluation.** First operational budget: six hours,
   online W&B and checkpoint on exit; original learning-rate schedule remains
   320000 vector steps. Evaluate saved policies in the physical environment
   without exploration noise. Judge sustained physical returns, success, force
   failures, spring release and runtime.

Short-term loss/reward fluctuations do not trigger architecture changes. Repair
confirmed data/gradient/runtime bugs; do not tune task thresholds to pass.

The old Gazebo SAC/model processes retain their disconnected material-force
wiring. Subsequent Gazebo comparisons use the repaired launch and source SAC
parameters; the GPU material-force path is directly connected.
