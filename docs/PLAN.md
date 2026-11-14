# Current plan — 2026-11-14

## Active task and decisions

One cut of freely placed cucumber, flat-bottom half potato and prism.
The active task is `free_cut_pose_v1`: +100 for prescribed pose completion
without force/workspace failure; the finite blade stops at the cutting board.
Material separation is recorded independently. Retain v3 running rewards,
180 control steps (9 s), gamma 1, and simulation wrench limits
[50,50,50,5,5,5]. These are not calibrated Mini45 hardware limits.

Half potato uses 1 s per-material settling; cucumber/prism use 0.75 s.
Each episode samples E and cut-spring stiffness scales 0.5–2 and density
0.8–1.25; target speed is 1–5 cm/s. Initial clearance 0.5–3 cm is sampled
once per world. No new geometry or material calibration is claimed.
The old v3 checkpoint/replay is not resumed into this task.

## Authorized restart: collection-scaled, full wrench observation

The user rejected the 192-world DiSECt and 3-world serial BladeMaster runs as
long-training configurations. Both were stopped with SIGTERM and their last
checkpoints/replay preserved. Neither had started policy updates. Their launch
records are historical, not active jobs.

Current work:
1. Implemented opt-in sensor_wrench6_v1 actor observation (34 dimensions, retaining target
   pose/velocity axes, reward and six-dimensional control actions). The old
   SliceIt-derived selected-axis input stays available.
2. Physical-transition cadence implemented: at least 10240-transition warmup (increased for enough consecutive vector steps at large world counts), then one
   ten-minibatch WM round per 640 new transitions; unchanged source mode retained.
   Preserve cadence/replay on restart, version metadata and collection-aligned LR.
3. Capacity test >=256 worlds on each backend, including partial reset,
   replay and actual WM+actor+critic updates. BladeMaster requires genuinely
   batched material kernels, not simply 256 sequential solvers.
4. Latest user decision: validate both backends on GPU7, then restart ONLY the DiSECt main run there. BladeMaster long training waits for the user's later arrangement. Fresh networks: seed0, real critic,
   actor horizon8, BC.25, 500 WM-only rounds. Each run has a 24h or 1280000
   physical-transition ceiling, whichever first. No automatic extensions.
   Do not touch unrelated jobs or claim equivalent physics/data distributions.

A6000 GPU1 is unavailable: another user's openpi training occupies ~44GiB. Do not stop it. Its 768-world learning probe was intentionally interrupted during initialization; no learning failure is inferred. Earlier A6000 physical evidence: 384-world physical-step/reset probe passed (128 per object);
768-world physical-step/reset probe also passed (~13.2s/step); full learning check next. Choose using measured throughput and full
learning memory, not initialization alone. Do not change physics to fit memory.
GPU7: BladeMaster 256-world physical/learning/restore validation passed. Its root suite passed 420 tests (5 skipped), backend 39. DiSECt 3072-world full learning, selected reset, geometry capture and save/restore passed; combined regression passed: root425/skip5, backend39.
New training has NOT launched yet. GPU7 DiSECt selected scale is3072 (1024/object), with3.449GiB free after full learning. Launch after commit/synchronization, using warmup30720 and the bounded recipe in the restart record. Capture actual long-run geometry for the requested screenshot.
See [restart record](TRAINING_SCALE_20261114.md) and
[force observation](WRENCH_OBSERVATION_REVIEW.md).

DiSECt tofu remains outside the main distribution. The former apparent 10mm
overlap was a ghost-node diagnostic error; physical contact geometry is tracked
separately, with no changed physical parameters. See GROUND_CONTACT_REVIEW.

## Completed evidence and scope

144 half-potato randomized/corner stability cases passed. Three-object small
batch collected 36 transitions and completed actual WM/actor/critic updates.
30 nominal pose-reward reference episodes completed, including failures/timeouts;
no physical force-limit failure occurred in that set.
BladeMaster banana and four native preset probes are interface/trajectory tests,
not learned policies. Banana/tofu/luncheon split into two components in the
latest probe; cake remained connected.
See [RESULTS](RESULTS.md), [task](TASK_FREE_CUT.md),
[assets](BLADEMASTER_DISECT_ASSETS.md) and [backend](BLADEMASTER_BACKEND_REVIEW.md).
Do not wait to explain every residual cutting spring before progressing with
the selected pose task. Keep all assets/checkpoints on servers.
