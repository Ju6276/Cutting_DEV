> Historical — superseded by [PLAN](../PLAN.md) and [RUNNING](../RUNNING.md) (pose_depth_v3), kept for traceability.

# Shared-task engineering training — 2026-11-16

## Authorization and question

The user requested: recheck outdated documents and visualization, synchronize
code, then start long training on GPU7 and A6000 after checks pass.
Question: can the existing WM+BPTT algorithm learn useful three-object control
under the current shared pose-depth task and randomized conditions?
Hypothesis to test, not assume: physical pose completion/return improves with
interaction. This is engineering training, not an algorithm superiority claim.

## Frozen settings

- Frozen source:5ec9f85 (exact manifests retained); native BladeMaster:c51ef73.
  Old branch names are launch provenance, not current development entry points.
- Scene: configs/shared/scene_v1.json; pose_depth_v2, final only, pitch off.
- Objects: cucumber003, flat-bottom half potato, prism; equal allocation.
- GPU7:3072 worlds; A6000 GPU1:510 worlds (170/object), full capacity verified.
- Fresh networks/replay, seed0. Real critic32, actor8, BC.25; gamma1.
- 30720 warmup transitions;640 transitions/update;500 WM-only rounds.
- Microbatch512, critic microbatch8192; effective batches2048/4096 unchanged.
- Each run:24h or1,280,000 new transitions, first bound; no extension.
  The transition stop is checked on vector-step boundaries, with at most one
  vector batch of granularity; compare the recorded actual counts.
- Paired checkpoints about every98k transitions; actual cadence depends on
  vector size and is recorded. Snapshots at vector steps1 and120.
- Online W&B mandatory; server recovery files do not substitute for cloud history.

## Readiness evidence

GPU7 current-task3072-world capacity:40 physical vector steps,122880 transitions,
actual WM/actor/critic updates, masked reset and replay restore, all finite;
5.27GiB free after learning and6.13GiB after restore. Report:
runs/long_readiness_20261116/disect3072_complete/report.json.
Runner fixes afterward do not change physical/model allocation; regression:
root562 passed/5 skipped initially; after offline rendering changes,
root569 passed/5 skipped and bundled backend41 passed (RESULTS).

Visualization uses evaluate_pose_policy --record-geometry, the shared export
adapter and render_episode_still; all-world snapshots use render_training_snapshot.
Six current shared scenes already passed physical-blade alignment checks and
render review in runs/interface_recheck_20261116. No duplicate blade, blue edge,
invented shank or artificial crack overlay. Scan appearance is approximate where
topology differs and is not presented as recovered UV.

A6000 receives the same repository commit, native source and checked robot/
geometry assets. Its old installation is preserved. A new510-world full-learner
probe on GPU1 checks actual allocation,40-step collection,32-step critic windows,
updates, masked reset, snapshot and replay restore with>=4GiB headroom.
The complete probe passed in2044.1s:40 physical steps/20400 transitions,
20 WM optimizer steps,2 policy steps and40 critic steps including restoration.
It exercised real critic32 and actor8 with effective batches2048/4096;
masked reset preserved untouched worlds, snapshot capture preserved physics,
episode sampling restored and all learning outputs were finite.
Free memory was7.912GiB after learning and7.904GiB after restore.
No episodes ended during these40 steps; this capacity probe is not a
task-success evaluation or evidence of every physical terminal branch.

## Settling time and scope

Each reset settles its selected objects for1.0 simulated second using that
episode's material parameters. It is a fixed-duration physical preparation, not
a loop certified to converge below a universal velocity threshold. The first
initialization also constructs robot/material state and compiles kernels when
needed. Measured GPU7 full initialization was436.3s; A6000's preflight was651.3s
including first-use compilation. Neither number is a guaranteed per-reset cost.

Initialization can fail (for example invalid contact/setup, nonfinite state or
allocation failure); logs/process status must be checked. Finishing the settling
interval alone is not proof of perfect rest for every randomized material.
The bounded checks establish runnable initialization for the sampled cases.

## Metrics and decision

Report per-object pose completion, failures/timeouts, return, speed error,
wrench utilization and elapsed time. Preserve transition counts and optimizer
counts; differing vectorization/hardware is not a matched compute comparison.
After the fixed budget, evaluate frozen checkpoints on identified development
conditions and report outcomes including failures. No automatic reward retuning,
material-range change or budget extension.

## Final status — 2026-11-17

- A6000 BladeMaster: state budget_complete (24-hour bound) at 20:02 CET,
  iteration 946, 482,460 transitions, 705 WM / 205 policy updates; last.pt and
  replay.pt written.
- GPU7 DiSECt: state failed at 17:55 CET after 80,640 s, iteration 304, 933,888
  transitions, 1406 WM / 906 policy updates, 5 pending learning rounds.
  Error: RuntimeError("Failed to allocate 8388608 bytes on device 'cuda:0'") in
  mujoco_warp collision (convex_narrowphase) during environment reset. W&B system
  metrics: device memory 16.7 GB at launch, stepwise growth to 23.8 GB (92.4 %)
  by 14:11, flat until 17:54, 24.37 GB at the failure; no external spike visible
  (the metric is device-wide, so other processes cannot be fully excluded).
  Last named paired checkpoint: checkpoint_00000288; last.pt/replay.pt were
  written at the failure. Not restarted (no extension rule).
- Both runs used pose_depth_v2, which is now read/evaluate-only; their results
  are history, not candidates for the v3 method.

## Verified launch status — 2026-11-16 20:12 CET

Both runs are running from clean, frozen source5ec9f85, native BladeMaster
c51ef73. Both manifests confirm real critic32, actor8 and BC.25.
The development branch includes later offline graphics and documentation only;
it does not alter the active source checkouts.

| Host/card | Worlds | Trainer PID | Online W&B |
|---|---:|---:|---|
| GPU7 RTX4090 |3072 (1024/object)|1188729|[scr1v5np](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/scr1v5np)|
| A6000 GPU1 |510 (170/object)|2614499|[pvord4sx](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/pvord4sx)|

The remotely queried W&B API reports both runs running and returns nonempty
physical-transition history. Local evidence: each frozen checkout's
runs/shared_long_20261116/cloud_verified.json, supervisor.json, training.log,
launch_plan.json and *_training/{manifest,progress,wandb}.json.
A6000 additionally retains launch_gate.json and bm510_preflight/report.json;
the gate started its supervisor only after the complete capacity check passed.

At verification GPU7 had passed80k physical transitions and was fitting the WM;
A6000 had recorded its first510 transitions, about563s after process start.
Both policies still had0 updates. These are launch/learning-pipeline states,
not physical success results. The500-round WM-only phase remains in place.
A6000 GPU0's unrelated job was untouched. GPU9's historical native fault remains
unresolved and is not part of this launch.

## Actual recording and chart checks

GPU7's actual first vector step includes all3072 worlds. The shared VTK renderer
produced runs/shared_long_20261116/actual_disect_3072.png in the development
checkout. Its display grid does not modify physical positions. Rendering needs
the dedicated VTK Python plus SliceIt/ROS source roots (RUNNING); the MPPI Python
has no VTK. Initial invocations missing those dependencies failed and were retained
as logs; the corrected documented command succeeded.

The separate wrench-panel compositor was checked on historical recordings from
both backends. It preserves the studio video and adds20Hz filtered six-axis
sensor curves, not500Hz peaks. These historical movies are format checks, not
current shared-task results.
