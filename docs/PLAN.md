# Current plan

## Numbered runs

- **1 — SAC + Gazebo + DiSECt**: resumed training in SliceIt
  `runs/sac_resume_20261103`, loading the retained TensorFlow checkpoint and
  replay buffer. The source run endpoint remains 100100 environment steps with
  the existing 72-hour execution deadline.
  W&B [xofe4ps8](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/xofe4ps8).
- **2 — SuperMimic + Gazebo + DiSECt**: stopped at iteration 872/update 316
  after critic divergence. Frozen episode return and bootstrap probes identify
  value extrapolation, terminal-head errors and excessive single-world data reuse.
- **3 — SuperMimic + MuJoCo Warp + DiSECt**: A6000 GPU 1 validation uses
  control_history_v2 and the 4096-transition source schedule. The first 1024-world
  update occurs at vector step 168; a 176-step/two-hour run crosses two updates
  and resumes physics. Rewards, force limits and optimizer settings are unchanged.
- Run 1 training stays on ROS 11321/Gazebo 11355. Frozen SAC evaluation uses
  separate ROS 11323/Gazebo 11357 and the source ten-episode deterministic evaluator.

## Current implementation

The world model predicts 36 coordinates: joint position/velocity, observed wrench
history and interval peak wrench. Ten frames supply 480 physical/command inputs.
The 860-coordinate rollout state also retains analytic observation/reward history
and success count. Particles remain in DiSECt and recorded data. Source widths,
effective batches 2048/4096, eight-step model fitting, 32-step BPTT, learning rates,
update ratio and task thresholds are unchanged. Microbatch 64 controls storage.

Both runs use the repaired post-step and zero-time reset joint snapshots.
Shutdown independently saves artifacts, closes the environment within wall-time
bounds and finishes W&B; a run-specific guard cleans up remaining child processes.
The completed SAC task check had 31 transitions, 30 updates, return 84.00210,
reset pose/history errors 0 and force-read errors 0. Old failed SAC checkpoint
files were deleted; diagnostic logs and replay remain. New world-model metadata
uses `synchronous_paused_joint_snapshot_v1` and rejects incompatible recordings
or resumed checkpoints.

## Next work

1. Evaluate the frozen SAC checkpoint in a separate simulator, verifying zero
   optimizer updates and unchanged actor weights.
2. Run A6000 connected sampling, model/actor/critic updates and subsequent physics;
   record real episode outcomes, terminal calibration and value diagnostics.
3. Use measured returns, frozen-rollout decomposition and source-scale replay
   accounting to assess the candidate before resuming long model-based training.

Implementation differences are in METHOD; commands and numbers are in RESULTS.
