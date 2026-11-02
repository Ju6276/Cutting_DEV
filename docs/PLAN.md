# Current plan

Long training remains stopped. Keep the source objectives, network widths,
effective batches, learning rates, rewards and physical task unchanged.

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
2. Diagnose the historical positive critic bootstrap explosion using captured
   replay, weights, optimizer state and RNG. Split reward and terminal bootstrap;
   compare measured, teacher-forced and autoregressive transitions. The 30-update
   fixed-replay experiment has finite values but reaches |V|=1007.20 and has not
   explained the historical positive explosion. Do not change the model from two
   updates or a small dataset's prediction errors.
3. Review the causal finding independently, repair a demonstrated error if found,
   then run sustained learning and held-out physical policy evaluation. Report
   task success separately from interface checks and finite learning updates.

The ROS task uses `native_ros_recurrent_v1` and
`synchronous_parameters_physics_and_joint_sample_v1`. Restart metadata rejects
checkpoints from older state/timing interfaces. The A6000 MuJoCo backend retains
its separate `coupled_recurrent_v1` layout; ROS timing fixes do not validate it.
