# Current plan — 2026-11-16

## Current audit and launch status

2026-11-16 live resource check: GPU7 and GPU9 each exposed one idle RTX4090;
A6000 (tamsGPU5) exposed two idle A6000 cards. At 18:48 CET all four cards
were idle after GPU7 completed the bounded 3072-world current-contract probe.
No new long training was launched.
The old GPU7 run is budget-complete; the old GPU9 run exited with SIGILL.
The A6000 deployment is still 2b73009 and is not the current shared-scene code.
Resource availability is a snapshot, not a reservation. The capacity evidence
and unexecuted 24-hour / 1,280,000-transition launch draft are under
runs/long_readiness_20261116. The 3072-world DiSECt learner/reset/restore probe
passed on code 7db74f2: 122880 transitions, 5.27 GiB free after learning and
6.13 GiB after restoration. This certifies bounded execution at that scale, not
policy quality or indefinite stability. The proposed fresh-run budget remains
unexecuted. BladeMaster has its current small-scale functional checks, but no
new large-scale A6000/GPU9 deployment check; GPU9's native fault remains unresolved.


[FRAMEWORK_AUDIT_20261116](FRAMEWORK_AUDIT_20261116.md) records local design choices,
material-preset provenance, the removed fabricated display connector and current
preflight scope. No new long training has been launched. Historical resource
settings below are not the new shared-scene launch configuration. The subsequent
controller/recording recheck fixed the all-world obsolete-knife rendering path;
RESULTS records fresh shared scenes, matching physical blade frames, studio and
overview images. UR10e is a potential port, not implemented or deployed.

## Active task and decisions

User-selected position modes: final only (default) or path plus final, selected by
common.task.tool_motion.enabled. Both use EE distance from the prescribed target
in robot-base XY. Final radius <40 mm; optional path cost has 25 mm free / 60 mm
failure / weight20. Particle displacement and fragment opening are not rewarded
or penalized in the current shared contract. Existing force/workspace/board
constraints remain. The current policies predate these settings; no new long run.

One cut of freely placed cucumber, flat-bottom half potato and prism.
The current task version is pose_depth_v2: shared maximum downward-depth progress
for both backends, with no spring-release reward or spring-weighted speed cost.
+100 remains prescribed pose completion without force/workspace failure and with
the finite-blade board boundary. Material separation stays independent.
180 control steps (9 s), gamma 1 and simulation wrench limits [50,50,50,5,5,5]
are retained in verification configs, not calibrated Mini45 hardware limits.
The completed runs below used pose-v1, not this version. No new long run launched.
See TASK_FREE_CUT for equations, state migration and verification status.
The initial particle-RMS motion implementation is retained for historical
reproduction only. It has been replaced in the shared contract by the two
robot-position modes above. Current checks and numeric results are in RESULTS.
Current scene/reset source is [SHARED_SCENE](SHARED_SCENE.md) and
configs/shared/scene_v1.json: paired per-world 5–30 mm gaps above each
settled object's top; no separate fixed-5-mm check configuration.
Per-world/per-episode material sampling is now implemented in both backends:
E .5–2, density .8–1.25, cutting resistance .5–2 (spring stiffness versus
cutting energy), and target speed 1–5 cm/s. Each changed material is settled for
1 s from raw state before knife alignment. The stream is saved for resume.
Randomized physical, learner and regression validation is recorded in RESULTS;
the earlier nominal checks were root 487 / skip 5 and backend 41.
No new long training is launched.
The shared physical tool revision is implemented and has passed three-object
reference probes; see SHARED_KNIFE. Regression completion is recorded in RESULTS.
The existing two backend runs retain their original tools and are not an equal-knife comparison.

In the completed DiSECt run (historical configuration), half potato used 1 s per-material settling;
cucumber/prism used 0.75 s. The new shared configuration uses 1 s for all three.
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
  full-learning probe; the subsequently authorized GPU9 long run is below.

Completed DiSECt job: GPU7 PID1048242, frozen code02203b1, checkout
/dataHDD/dong/Projects/Cutting_DEV_pose_long_20261114.
Output:runs/pose_wrench6_3072_20261114/training.
Fresh seed0,500 WM-only rounds, warmup30720 transitions,640 new transitions per
WM round; microbatches512/8192. Ceiling1280000 new transitions or24h, whichever
first. Named checkpoints every32 vector steps retain paired replay. Originally launched with offline W&B; recovered to online history forwarding
without restarting (see tracking section below). Capture actual geometry at steps1/120.
Runtime manifest and first physical steps are verified: critic real/32, actor8, gamma1, BC.25, observation34. The first-step screenshots show the initial collection phase (WM/policy updates0), not a learned cutting result.

A6000's development checkout has the shared training code931e2b5 but no job is launched
there; another user's GPU workloads are untouched. The old192/3 runs remain
stopped and frozen. No automatic budget extensions; the later user-authorized
BladeMaster GPU9 launch is recorded below.

Actual training overview/detail images were rendered from step1. Training reached its budget: 346 vector steps, 1,062,912 transitions, 1109 policy updates. Frozen evaluation is the current step; do not silently extend training.
See [restart record](TRAINING_SCALE_20261114.md) and
[force observation](archive/WRENCH_OBSERVATION_REVIEW.md).

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
exclusive 48 GB capacity is still unmeasured. No BladeMaster long run started on A6000; the later GPU9 launch is recorded below.
See RESULTS: "A6000 shared-device BladeMaster training capacity" and
runs/bm_a6000_capacity_20261114. The active GPU7 main run is unchanged.

### Online experiment tracking — required 2026-11-15

Every future long run must initialize online W&B and expose a verified cloud URL.
No offline fallback for authentication/network problems. See AGENTS and RUNNING.
The already running GPU7 job remains uninterrupted: its original offline history
was uploaded. A CPU-only online forwarder now resumes the same cloud run and
polls the original history file every 30 seconds. It forwards complete original
metric records without restarting training. The history axis remains iteration;
the online transport counter is not the training iteration. It finishes the
cloud run when training exits. W&B CLI sync of an incomplete offline file marked
the cloud run failed despite live training; periodic CLI sync was stopped and
replaced by this online forwarder. Forwarder system-resource stats are disabled
so they cannot be confused with training GPU usage.
Run: https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/k48zc1lh
Watcher and status: runs/pose_wrench6_3072_20261114/forward_cloud.py and cloud_sync.json.
This is recovery for the existing offline launch; future launches use online
tracking directly. Server checkpoints/logs remain recovery copies.

### BladeMaster GPU9 deployment — authorized 2026-11-15

The user moved the requested BladeMaster long run from the occupied A6000 to
GPU9. GPU9 is reachable through tams61 and its RTX4090 was idle at deployment.
That occupancy was a launch-time snapshot. At the final evaluation check, GPU0 was idle and GPU1 hosted tu's feeding job; do not stop other users' work.

Frozen training checkout: /dataHDD/dong/Projects/Cutting_DEV_bm_long_20261115,
commit931e2b5 (includes the opt-in cache guard). External material source: e2e5221, same revision used for the
GPU7/A6000 batching and learning checks. UR5e model, source reset, controller,
task config and intact object assets were copied from the validated deployment
and hashes compared. Existing GPU9 historical checkouts are untouched.

Validated launch:256 worlds (86 cucumber,85 half-potato,85 prism); the local
learning/reset/restore capacity check completed before launch. Same approved
24-hour/1280000-transition budget; online W&B required. This is a nominal
BladeMaster engineering run, not the DiSECt randomized-material distribution.
Live records and exact command:
/dataHDD/dong/Experiments/Cutting_DEV/runs/bm_gpu9_20261115/
(plan.json,launch_command.json,preflight256/report.json,training/manifest.json).
Full preflight passed with the optional cache release, leaving4.28GiB free
after restored updates. Long training launched as PID138581 onGPU9, online
W&B, from fresh networks. 256 worlds remain86/85/85; actor8,real critic32,
BC.25,500 WM-only rounds,10240-transition warmup,640 transitions/update.
Microbatch512/critic8192; effective WM2048 and actor4096 batches unchanged.
Budget24hours or1280000 transitions, paired checkpoints every384 vector steps.
This is an engineering training run, not evidence of trained policy performance.
See RESULTS for the failed headroom candidates and final functional checks.
Online W&B verified (first read:iteration4,1024 transitions,policy updates0):
https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/aqoyihk2

## Repository and storage maintenance — 2026-11-15

Completed: retired preflight tensors and hash-identical data copies removed;
17 inactive worktrees removed after source/patch snapshots. Active run checkouts,
paired checkpoints/replay, assets and Python dependencies are preserved.
[DATA](DATA.md) is the current location and recovery index. Historical launch
and completed-review notes are under archive; redundant old plan/paper drafts
are accessible through immutable Git links in the document index.
This maintenance changes no training code, task, budget or running process.



## Current follow-up — completed evaluations and unresolved GPU9 exit (2026-11-15)

GPU7 completed24 hours (policy1109 /1,062,912 transitions). Frozen evaluation:
0/48 pose successes,48 timeouts,mean return-10.97. Earlier policy373 on exactly
the same48 randomized development conditions also failed:48 workspace exits
at step55,mean return-99.89. Earlier completed-training success rates do not
establish successful deterministic execution. Raw traces are saved for targeted
investigation; no task, reward or training change has been made.

GPU9 stopped early with SIGILL (-4), located by the kernel in libcuda.so.595.80.
The triggering cause is unresolved; no nearby visible Xid/OOM record or usable
core was found. CUDA smoke checks and physical evaluation passed, which does
not establish that long training is repaired. Latest saved policy866:
48/48 nominal-replica pose successes,mean return106.94,mean34 steps.
The process reached policy998 before the crash; do not conflate it with the
evaluated saved checkpoint. W&B's finished badge is not budget completion.

See RESULTS's2026-11-15 frozen-evaluation entry and DATA for evidence/paths.
DiSECt random conditions and BM nominal replicas are not a matched algorithm
comparison. All three evaluations are completed, with no automatic restart.
Next: investigate main-policy execution failures using the saved trajectories;
preserve GPU9 exit evidence and diagnose its runtime fault before another long
run. No new long-training budget has been selected here.

## 2026-11-16 — Authorized shared knife and frozen-policy interpretation

Use the main DiSECt physical knife as the common geometric/mounting reference.
Keep original run checkouts and checkpoints frozen. Validate geometry, rigid
transform, force/torque transformation, board clearance and three-object contact
before treating the replacement as usable. No new long training in this task.

Use the two existing policies' saved first episodes for force/action/motion plots
and decoder-feature PCA. Reconstruct all ten history frames, verify reconstructed
actions against the capture, label object/material/initial-condition confounds.
A separate input-history ablation measures open-loop action sensitivity.
Do not infer stiffness identification or safety from clusters alone.
Initial analysis uses validated cucumber, half potato and prism assets;
tofu/carrot are not silently substituted into these policies' training conditions.
Budget: CPU analysis of existing 96 episodes; bounded tool probes up to 20 minutes
each, sequential on GPU7, plus regression tests. No policy updates.

Frozen-policy plots and the limits of their interpretation are in
[POLICY_MECHANICS_20261116](POLICY_MECHANICS_20261116.md). Force-history changes
affect raw outputs; clipping suppresses most executed-action changes in the main
policy. Existing clusters do not establish material identification. No new
training was launched.
