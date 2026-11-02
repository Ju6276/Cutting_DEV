# Current plan

Run numbers used in discussion and W&B:

- **1 — SAC + Gazebo + DiSECt**: stopped at 3257 steps / 133 episodes after
  a joint-message timeout; final checkpoint and replay saved. Diagnose joint
  publication/callback timing before resuming the original 100100-step job. W&B `064z1a85`; physical episode return is
  `reward/episode`; `reward` / `reward/step` and `reward/episode_mean_20` are
  were streamed through an attached producer sharing the original W&B service.
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
force (36 coordinates). Ten causal frames give 860 rollout coordinates and 480 dynamics inputs; particles remain
in physical recordings. Methods and state assumptions are in METHOD.

Completed compact-candidate checks: causal packing/reset isolation, original
reward/termination parity over 182 recorded transitions, and trained 8/32-step
BPTT finite differences (relative error below 8.5e-7). Independent agent reviewed
implementation and results. Three training episodes provide 137 transitions;
one held-out episode provides 45. The 600-second budget completed 328 Adam steps.

Current failure: 8-step held-out RMSE improves to 0.03493, but 32-step RMSE is
14.68 and force predictions diverge on training episodes too. Do not launch a
connected policy job from this checkpoint. The recorded sample does not yet
separate inadequate data coverage, finite-history ambiguity and rollout feedback.

The frozen input intervention reduces 32-step RMSE 14.6728 -> 0.08983 by replacing
only derivative-history inputs, identifying an amplification path. The active v2
candidate removes derived observations from dynamics inputs; joints, force and
commands/history remain. Analytic observations/rewards and critic retain the
complete compact rollout state. No recorded future state enters this candidate.

Next work, in order:

1. Independently review the physical/analytic input separation and gradient tests.
2. Repeat the same 137/45-transition episode split and 600-second fitting budget
   with v2, unchanged targets, optimizer and batches. Evaluate 1/8/32-step physical
   errors and force-limit predictions, then actual trained BPTT finite differences.
3. Decide the next data/learning experiment from those results; physical policy
   training requires accurate model predictions over its 32-step horizon. Keep
   whole-episode validation and rare terminal events explicit.

The ROS task uses `native_ros_recurrent_v1` and
`synchronous_parameters_physics_and_joint_sample_v1`. Restart metadata rejects
checkpoints from older state/timing interfaces. The A6000 MuJoCo backend retains
its separate `coupled_recurrent_v1` layout; ROS timing fixes do not validate it.
