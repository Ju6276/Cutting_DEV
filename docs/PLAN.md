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

Completed for this restart:
- Versioned34-dimensional full-wrench actor observation; same6 controls/task.
- Physical-transition cadence, real critic32, actor8, BC.25.
- Both backends validated on GPU7; integrated root425/skip5, backend39.
- DiSECt3072 (1024/object) completed full learning/reset/restore preflight, with
  3.449GiB free after full learning. BladeMaster256 completed its separate
  full-learning probe; its long run remains deferred.

Active DiSECt job: GPU7 PID1048242, frozen code02203b1, checkout
/dataHDD/dong/Projects/Cutting_DEV_pose_long_20261114.
Output:runs/pose_wrench6_3072_20261114/training.
Fresh seed0,500 WM-only rounds, warmup30720 transitions,640 new transitions per
WM round; microbatches512/8192. Ceiling1280000 new transitions or24h, whichever
first. Named checkpoints every32 vector steps retain paired replay. Offline
W&B and JSON logging; capture actual geometry at steps1/120.
Runtime manifest and first physical steps are verified: critic real/32, actor8, gamma1, BC.25, observation34. The first-step screenshots show the initial collection phase (WM/policy updates0), not a learned cutting result.

A6000's development checkout is synchronized to02203b1 but no job is launched
there; another user's GPU workloads are untouched. The old192/3 runs remain
stopped and frozen. No automatic budget extensions or BladeMaster launch.

Actual training overview/detail images have been rendered from step1 and delivered. Next: retain per-object learning/episode metrics through the fixed budget; assess the saved checkpoints without changing task or budget mid-run.
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

### A6000 BladeMaster capacity — completed 2026-11-14

The user's HOAP job was stopped as requested; other users' jobs remain active.
With openpi occupying 26,224 MiB of GPU0, the three-object full training
preflight passed at 192 and 208 environments. At 256, updates passed but robot
reset ran out of memory. 208 (70 cucumber / 69 half-potato / 69 prism) is the
largest validated shared-device candidate, with 3.81 GiB free after restore;
exclusive 48 GB capacity is still unmeasured. No BladeMaster long run started.
See RESULTS: "A6000 shared-device BladeMaster training capacity" and
runs/bm_a6000_capacity_20261114. The active GPU7 main run is unchanged.
