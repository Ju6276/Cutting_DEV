# Current plan

Run numbers used in discussion and W&B:

- **1 — SAC + Gazebo + DiSECt**: active fresh training, original 100100-step
  endpoint and 72-hour wall budget. W&B `064z1a85`; physical episode return is
  `reward/episode`; `reward` / `reward/step` and `reward/episode_mean_20` are
  streamed through an attached producer sharing the original W&B service.
  All use `reward_iteration` as the physical-step axis; training is unchanged.
- **2 — SuperMimic + Gazebo + DiSECt**: diagnostic run `3lvaf2vr` stopped cleanly
  after 176 physical steps and 54 updates. The first positive value anomaly was
  captured at update 16, so work now uses the saved states rather than waiting
  for the original 800-step endpoint. Physical `reward` (176 rows) and
  `episode_return` (3 rows) are backfilled and verified on W&B; the next launcher
  must stream these during learning. MuJoCo is a separate backend.

Keep source objectives, network widths, effective batches, learning rates,
rewards and physical task unchanged.

Completed source repairs:

- Native ROS collection exports FDCC solver memory, filter history and the full
  wrench queue: 8990 state coordinates. Equal measured joints with different
  internal solver velocity produce different next commands in the native solver.
- Reset leaves physics paused and captures post-reset observations. Configuration
  writes complete before physics; knife FK uses a joint sample from the completed
  step. Configured 50 ms actions execute 50 simulation ticks.
- Workers unpause for the source controller shutdown, including constructor and
  later setup failures. Independent agents reviewed these changes.

Reassessment, 2026-11-02:

- Connected collection and two learning updates ran; a separate 45-transition
  episode has zero terminal mismatches and reward error 1.42e-14. Sustained run 2
  fails in world-model update 16 before policy/critic amplification.
- Exact Adam replay reproduces saved weights bitwise. Its tenth sampled-batch
  objective jumps 8.97483 -> 11097.31520. This establishes finite-step overshoot;
  the optimizer implementation has no demonstrated defect in this trace.
- 8990 is the recorded state size, including 7122 particle coordinates. The
  model independently predicts some deterministic cache readouts. A
  six-coordinate coupling cache contributes 36.61% of pre-update state loss;
  99.29% of that cache loss comes from two reset-start windows.
- Replay then contains 80 rows and 64 overlapping eight-step windows. Source
  cadence/batches transferred from 4096 worlds to one world preserve numerical
  settings but change data diversity and reuse substantially.

Current candidate, approved by the user: control-history prediction. Network
outputs are joint position/velocity, observed force history and interval peak
force (36 coordinates). Ten causal frames give 860 state inputs; particles remain
in physical recordings. Methods and state assumptions are in METHOD.

Next work, in order:

1. Validate causal packing, reset isolation, analytic task parity and multi-step
   gradients with independent review. Keep original task and training settings.
2. Fit on three complete recorded episodes and evaluate a separate complete
   episode at 1/8/32 steps. Budget: 50 source world-model updates / 600 seconds,
   progress each update/minibatch, atomic checkpoint on budget or failure.
   Compare against the untrained constant-velocity/force-persistence prior;
   report physical-unit errors, reset-start errors and missed force terminations.
3. Use prediction results to decide the bounded connected learning experiment;
   validate physical episode returns and bootstrap behavior independently. Finite
   history sufficiency and generalization to new cut histories remain empirical
   questions. No concurrent latent-model or optimizer change.

The ROS task uses `native_ros_recurrent_v1` and
`synchronous_parameters_physics_and_joint_sample_v1`. Restart metadata rejects
checkpoints from older state/timing interfaces. The A6000 MuJoCo backend retains
its separate `coupled_recurrent_v1` layout; ROS timing fixes do not validate it.
