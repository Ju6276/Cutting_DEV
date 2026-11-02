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

Remaining work, in order:

1. Connected check completed: 45 physical steps, two complete source learning
   updates, return to physical simulation, checkpoint and normal cleanup. A separate
   45-transition episode has zero terminal mismatches and reward error 1.42e-14.
2. Diagnose update 16 using the retained pre-world-model, pre-policy and
   post-update weights, replay and RNG. Compare frozen rollouts with identical
   starts while swapping one component at a time. Replay states and reset
   histories are finite and correctly aligned; the recorded model rollout grows
   rapidly before large raw actions and terminal values appear. Determine the
   initiating update before changing the model or training settings. Frozen
   phase substitution localizes the first departure to world-model update 16:
   pre-fit rollout is stable; post-fit rollout diverges even with zero actions.
   Exact replay now isolates finite-step Adam overshoot: the tenth sampled-batch
   full objective jumps 8.97483 -> 11097.31520, with final weights matching the
   captured model bitwise. Keep the optimizer unchanged while reviewing data
   coverage and state conditioning; any altered warmup, scaling or step acceptance
   is a separate method change requiring a paired experiment.
3. Review the causal finding independently, repair a demonstrated error if found,
   then run sustained learning and held-out physical policy evaluation. Report
   task success separately from interface checks and finite learning updates.

The ROS task uses `native_ros_recurrent_v1` and
`synchronous_parameters_physics_and_joint_sample_v1`. Restart metadata rejects
checkpoints from older state/timing interfaces. The A6000 MuJoCo backend retains
its separate `coupled_recurrent_v1` layout; ROS timing fixes do not validate it.
