# Shared-task engineering training — 2026-11-16

## Authorization and question

The user requested: recheck outdated documents and visualization, synchronize
code, then start long training on GPU7 and A6000 after checks pass.
Question: can the existing WM+BPTT algorithm learn useful three-object control
under the current shared pose-depth task and randomized conditions?
Hypothesis to test, not assume: physical pose completion/return improves with
interaction. This is engineering training, not an algorithm superiority claim.

## Frozen settings

- Source: current codex/board-cut-v4; exact launch commits recorded in runtime
  manifests. Native BladeMaster:c51ef73 (codex/shared-disect-knife).
- Scene: configs/shared/scene_v1.json; pose_depth_v2, final only, pitch off.
- Objects: cucumber003, flat-bottom half potato, prism; equal allocation.
- GPU7:3072 worlds; A6000 GPU1:510-world candidate, subject to full capacity.
- Fresh networks/replay, seed0. Real critic32, actor8, BC.25; gamma1.
-30720 warmup transitions;640 transitions/update;500 WM-only rounds.
- Microbatch512, critic microbatch8192; effective batches2048/4096 unchanged.
- Each run:24h or1,280,000 new transitions, first bound; no extension.
- Paired checkpoints about every98k transitions; actual cadence depends on
  vector size and is recorded. Snapshots at vector steps1 and120.
- Online W&B mandatory; server recovery files do not substitute for cloud history.

## Readiness evidence

GPU7 current-task3072-world capacity:40 physical vector steps,122880 transitions,
actual WM/actor/critic updates, masked reset and replay restore, all finite;
5.27GiB free after learning and6.13GiB after restore. Report:
runs/long_readiness_20261116/disect3072_complete/report.json.
Runner fixes afterward do not change physical/model allocation; regression:
root562 passed/5 skipped; bundled backend41 passed (RESULTS).

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
Do not start its long run on a failed or partial probe.

## Metrics and decision

Report per-object pose completion, failures/timeouts, return, speed error,
wrench utilization and elapsed time. Preserve transition counts and optimizer
counts; differing vectorization/hardware is not a matched compute comparison.
After the fixed budget, evaluate frozen checkpoints on identified development
conditions and report outcomes including failures. No automatic reward retuning,
material-range change or budget extension.

## Launch status

Deployment and A6000 capacity verification in progress. No launch is claimed
until its PID, manifest, online W&B URL and first remote history are checked.
GPU0 on A6000 is occupied; use GPU1 only. GPU9's historical native fault remains
unresolved and is not part of this launch.
