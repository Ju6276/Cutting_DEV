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

Next work, in order:

1. Specify one exact-identity transition candidate, retaining the recorded
   packet, existing error terms, physical state, controller/filter memory and
   training settings. Start with queue readouts and controller tip FK whose
   identities hold at recorded boundaries. Preserve coupling pose memory until
   a source-correct query gate exists: it survives reset, and its initialized
   flag alone is insufficient for replacing it with current FK.
2. Compare that candidate against saved update 16 with identical replay, RNG and
   optimizer state. Evaluate common physical fields and 8/32-step state growth,
   reward/terminal parity and policy gradients. Obtain an independent review.
3. If that controlled comparison passes, evaluate held-out physical episodes and
   sustained connected learning. If the same failure remains, reassess the
   single-world data/update regime before adding another model change. Keep
   graph models as a separate future candidate, not a simultaneous replacement.

The ROS task uses `native_ros_recurrent_v1` and
`synchronous_parameters_physics_and_joint_sample_v1`. Restart metadata rejects
checkpoints from older state/timing interfaces. The A6000 MuJoCo backend retains
its separate `coupled_recurrent_v1` layout; ROS timing fixes do not validate it.
