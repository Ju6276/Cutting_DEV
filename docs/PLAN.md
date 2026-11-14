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

## Authorized main run

The user requested an A6000 long run after checking code/configuration.
Use GPU1 only; GPU0's unrelated SuperMimic run is left untouched.
Checkout: /dataHDD/dong/Projects/Cutting_DEV_pose_train_20261114.
Configuration: [pose_three_object_train](../configs/free_cut/pose_three_object_train.json).
64 worlds per object, 192 total, parallel object groups and one shared learner.
Start fresh, 500 WM-only updates, real-data critic targets, actor horizon 8,
BC 0.25, microbatch 64 and critic microbatch 1024. No risk-head variant.

Budget: at most 5000 vector iterations (960000 physical transitions) or
24 wall hours, whichever occurs first; no automatic extension. Save every
250 vector iterations, at WM warmup completion, first joint update and exit.
Offline W&B plus local progress/episode/update JSON are sufficient; no login
or key rotation is required for this run.

Question: can the existing algorithm learn the current three-object pose task?
Report per-object pose success, force/workspace/timeout outcomes, separation,
speed error, return, force peaks and utilization, physical transition count,
WM/critic diagnostics, reset/physics/update time and memory.
This is a development run, not a matched-budget algorithm comparison.
Do not redesign the reward from a few early failures. Nonfinite values,
corrupt data or execution failures stop the run; ordinary timeouts do not.
Wall-budget expiry saves and ends; checkpoint evaluation is a separate bounded job.

Preflight is in progress: verify asset hashes, full suites, a same-batch short
run including learning/checkpoint restore, and actual memory/throughput.
The short probe reduces WM warmup only for interface validation and is discarded;
the long run starts fresh with the 500-update warmup.
[Launch record](TRAINING_20261114.md) records the actual outcome and command.

## Parallel work, separate physical identities

- DiSECt tofu: diagnose actual contact geometry separately from ghost FEM nodes,
  correct measurements and fix demonstrated physics errors only. It is not in
  the main training distribution; do not tune main physics for this candidate.
- BladeMaster: prepare uncut cucumber/half-potato/prism geometry, validate native
  cutting, reaction coupling and multi-object learning. GPU7 training is
  conditionally authorized only after those checks. Serial independent solvers,
  initially one world per object; nominal parameters, different fracture law
  and depth-progress task. It is not the same distribution/reward as the main run.
  Its wall/iteration cap is at most 24 h / 5000 steps, without automatic extension.
  Coordinate GPU7 validation with the tofu diagnostic and root regression suites.

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
