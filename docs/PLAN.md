# Current plan

## Numbered runs

- **1 — SAC + Gazebo + DiSECt**: resumed training in SliceIt
  `runs/sac_resume_20261103`, loading the retained TensorFlow checkpoint and
  replay buffer. The source run endpoint remains 100100 environment steps with
  the existing 72-hour execution deadline.
  W&B [xofe4ps8](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/xofe4ps8).
- **2 — SuperMimic + Gazebo + DiSECt**: stopped at iteration 872/update 316
  after critic divergence. Frozen episode return and bootstrap probes identify
  large value bias and terminal-head errors. The relative causal contributions
  remain to be established; single-world data reuse is a measured regime difference.
- **3 — SuperMimic + MuJoCo Warp + DiSECt**: bounded training and frozen
  evaluation finished on A6000 GPU 0. `critic_online_terminal_coverage_20261104`
  finishes at policy52/WM552 after 2872 complete episodes; the frozen zero-noise
  policy succeeds 8/8 in 18 steps. Value MAE is 141.6256. No GPU training process
  remains from this run.
  W&B [4s8uxh4o](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/4s8uxh4o).
- Run 1 training stays on ROS 11321/Gazebo 11355. Frozen SAC evaluation uses
  separate ROS 11323/Gazebo 11357 and the source ten-episode deterministic evaluator.

## Current implementation

The world model predicts 36 coordinates: joint position/velocity, observed wrench
history and interval peak wrench. Ten frames supply 480 physical/command inputs.
The 860-coordinate rollout state also retains analytic observation/reward history
and success count. Particles remain in DiSECt and recorded data. Source widths,
effective batches 2048/4096, eight-step model fitting, 32-step BPTT, learning rates,
update ratio and task thresholds are unchanged. Microbatch 64 controls storage.

Gazebo runs 1 and 2 use the repaired post-step and zero-time reset joint snapshots.
Shutdown independently saves artifacts, closes the environment within wall-time
bounds and finishes W&B; a run-specific guard cleans up remaining child processes.
The completed SAC task check had 31 transitions, 30 updates, return 84.00210,
reset pose/history errors 0 and force-read errors 0. Old failed SAC checkpoint
files were deleted; diagnostic logs and replay remain. New world-model metadata
uses `synchronous_paused_joint_snapshot_v1` and rejects incompatible recordings
or resumed checkpoints.

## Next work

1. The shared run2/run3 learner applies the entering terminal reward once,
   then freezes state and zeros later rewards/bootstrap; the same-checkpoint
   imagination diagnostic confirms this behavior. Run3's complete physical
   evaluation exposes model disagreement: its H32 prediction calls the
   successful step18 a workspace failure.
2. The `SequenceReplay` adapter now includes completed short terminal sequences
   with a valid-step mask. Full H8 loss/gradients and mixed-length microbatch
   updates match their unpadded references. The previous run's 1121 collision
   failures were excluded from WM fitting; that run saved episode counts but
   no replay transitions for those failures.
3. Collect fresh transitions and validate the repair from policy52/WM552 with
   1024 worlds, the original 40-step refill and unchanged H8/H32 schedule.
   Budget: ten joint updates or 1800 seconds, followed by eight frozen complete
   physical episodes, within 2700 seconds total. Save short real terminal
   sequences and count their actual use in WM fitting. Evaluate critic values
   against the new policy's complete physical returns; fixed-state model
   comparisons keep the old policy52 actions separate. Independent review checks
   sampling, masked-loss equivalence and return alignment.
4. Run1's completed frozen SAC evaluation has ten successes, mean return
   91.02107 and zero optimizer updates. Keep its training independent of run2/3
   model and critic changes.

Implementation differences are in METHOD; commands and numbers are in RESULTS.

## External implementation evidence

[ndm-gym](https://github.com/Frank-ZY-Dou/ndm-gym/tree/0171f5eadee50492d6ff30fa04b55e16cc11b61e)
provides full-update equivalence checks, independent physical evaluation and
500 world-model-only iterations; all three are now applied here. Its RunningNorm
critic and analytic soft termination are separate method choices. The current
candidate keeps the Cuboi learned head and static scales. SliceIt physical
thresholds and reward remain fixed. Analytic termination alone cannot restore
a force-limit event that the dynamics model failed to predict.
