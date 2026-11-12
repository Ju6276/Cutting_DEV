# Results

Dated measurements from 2026-11-11 on (multi-object task). Earlier entries (2026-11-02..10, fixed
prism development) are in [archive/RESULTS_20261102_20261110](archive/RESULTS_20261102_20261110.md).
[PLAN](PLAN.md) says what happens next; old next-step prose in entries is not a plan.

## What currently holds

| Finding | Entry |
|---|---|
| The multi-object task definition has four defects (success rule, bottom hold, reward, reference) | 2026-11-12 |
| The bottom hold changes the measured cutting force up to about 7x (cucumber_003: 37 N held, 5.6 N unheld); the unheld potato is pushed away | 2026-11-12 hold tests |
| With the inherited reward the trained base cuts at 20-27 cm/s against a 1-5 cm/s command | 2026-11-12 |
| potato_003 needs DiSECt substep 8e-6 s to be stable over the S0 material range | 2026-11-11 instability entry |
| Parallel object groups: 2.5x faster, consistent with sequential within run-to-run variation | 2026-11-11 |
| Risk penalty and official SuperMimic adaptation: failed first runs, causes identified | 2026-11-12 |
| The simulated sensor force is the sum of contact-force magnitudes (direction-blind, no torque); lateral sawing has no effect on it | 2026-11-12 sawing check |

**Superseded:** the 2026-11-11 "force needed to cut the held scanned objects" table (hold artefact
for cucumber_003) and every success/force-limit rate of the S3 runs (defective task definition).
Fixed-prism results of the base method (8/8 completions, archive) hold for the fixed prism only.

## 2026-11-11 — Multi-object task and force-risk constraint: implementation check

Code `6460f67`…`e9b1704`. Functional checks only; no learning result.

- Start height by IK: four worlds lifted 0/1/2/3 cm put the DiSECt knife 1 cm apart
  (0.0882…0.1182 m); reset restores each world's start; zero lift reproduces the
  captured start exactly.
- Grouped environment (prism, scanned apple/potato/cucumber, reference control):
  per-world conditions sampled and reported; the prism and cucumber reach the goal.
  Groups run sequentially: ~10.8 s per control step for four groups (GPU7).
- `train_gpu.py --conditions configs/multi_condition/smoke.json` (16 worlds, 46
  iterations) completes WM-only and joint updates with and without
  `--risk-constraint` (risk head fitted each update, lambda logged, checkpoint saved).
- Root tests 232 passed / 15 skipped; backend 28 passed.

Scanned objects: OmniObject3D (CC BY 4.0) apple_002, potato_003, cucumber_003,
cucumber_002; fTetWild with a ~1 mm envelope gives 284–744 vertices, minimum edge
1.3–5.1 mm, and DiSECt cut meshing succeeds (65–298 cut springs). Earlier attempts
were discarded: dense fTetWild meshes (0.2 mm slivers) diverged at the first step;
6 mm voxel meshes were stable but blocky; three scans failed DiSECt cut meshing.
Earlier sweeps over material-swapped DiSECt prism/cylinder/sphere scenes and their
videos were deleted (wrong object shapes; the scanned apple started intersecting
the knife). The per-object force-limit scale tried on 2026-11-10 was removed: all
objects share one limit and the learner gets no object information.

Cut-force measurement (in progress, `runs/food_force_20261111`, knife 1 cm above the
object, force-limit termination off, speeds 1/2/5 cm/s): apple at 5 cm/s reached
200 N with 77/298 springs released when the run ended (55 steps), i.e. not cut.
The one force limit is chosen from the complete measurement (PLAN step 1).

## 2026-11-11 — Parallel object groups

`MultiConditionEnvironment(parallel=True)` (config `"parallel": true`): one persistent
process per object group on the same GPU, sampling and the learner in the parent,
fixed-size fields exchanged on the CPU. `scripts/check_parallel_groups.py`, smoke
config (4 groups × 4 worlds, random actions, GPU7 shared with three other jobs):
sequential 20.6 s per control step, parallel 8.3 s (2.5×). Parallel vs sequential
max differences over 10 steps: joint 8.7e-4 rad, peak wrench 4.5 N, no ending
mismatch. Sequential vs sequential (same seed and actions) differs as much: identical
for two steps, then 0.29 → 8.1 N peak wrench by step 6 — the GPU physics is not
bitwise repeatable under contact. The parallel path is therefore consistent with the
sequential reference within the simulator's own run-to-run variation.

## 2026-11-11 — Force needed to cut the held scanned objects

`scripts/measure_cut_force.py`, `runs/food_force_held_20261111`: knife 1 cm above the
object, straight downward reference at 1/2/5 cm/s, force-limit termination off, object
held by its bottom 6 mm (except near the cut plane). Largest sensor force component
before 90% of the cut springs were released (a separation proxy):

| Object | 1 cm/s | 2 cm/s | 5 cm/s | Object displacement |
|---|---:|---:|---:|---|
| potato_003 | 9.7 N (cut) | 9.7 N | 10.3 N | ≤ 1.4 mm |
| cucumber_003 (17 cm) | 33.8 N (cut) | 34.5 N | 41.4 N | ≤ 0.6 mm |
| cucumber_002 (9 cm) | 16.9 N | 16.1 N | 19.9 N | ≤ 1.2 mm |

Larger peaks after the cut are the knife pressing into the held layer and board as
the reference keeps descending; they are not cutting forces. Without the hold the
potato slid 26 mm sideways and was not cut; holding also raised cucumber_003's force
(8.5 → 41 N at 5 cm/s), so the hold is part of the task definition. The apple
(earlier run, not held) reached 179–200 N at 1–5 cm/s without completing; it was not
repeated with the hold. These numbers inform the task force limit, which is set from
the external Mini45 sensor (not the inherited 50 N).

## 2026-11-11 — Multi-object functional run with the ensemble-risk comparator

`configs/multi_condition/smoke.json` (prism, potato_003, cucumber_003; 4 worlds each;
parallel groups), `train_gpu.py --risk-constraint --risk-estimator ensemble`, 46
iterations, then `evaluate_multi.py` for 12 steps. Completed: WM-only update, four
ensemble members fitted per update (losses 0.40–0.64), joint update with the ensemble
risk in BPTT (mean upper estimate 0.11, penalty 0 at this stage, lambda 10). The
evaluation loaded the 8-dimensional-goal policy and ran; no episode ended within 12
steps, so its per-episode rows are not yet exercised. Object-group processes exited
with EOF tracebacks after training (harmless, the parent had not closed them); fixed
by closing the environment at the end of training and ending workers quietly on EOF.


## 2026-11-11 — Codex repair review after c76deb5

Scope authorized by the user: repair the identified implementation issues; defer
multi-object SAC implementation and expanded evaluation. No policy training,
new force sweep or performance evaluation was launched.

- Ensemble LR: `set_lr` previously scheduled only the main WM, actor and critic.
  The four other WM optimizers retained the initial LR. They now follow the same
  schedule, also after restore; the quantile head keeps its separate fitting LR.
  The old ensemble smoke remains a functional run, not a matched-schedule result.
- Limits: multi-object construction now requires six explicit finite positive
  limits before loading assets or spawning workers. There is no silent legacy
  fallback. The smoke config explicitly retains its historical 50 N / 10 N·m
  settings solely to reproduce the interface check, not as the S0 decision or
  Mini45 rating. Single-scene source defaults remain unchanged.
- Lifecycle: close the parent's child-pipe copy, surface unexpected worker exits,
  clean up initialized siblings on setup failure, and close all workers even
  when learner setup/checkpoint loading/finalization fails. Close is idempotent;
  a worker that ignores close is warned about and terminated/reaped after the
  existing 30-second grace period, with kill fallback.
- GPU reset test: its force-limit fault injection still indexed the old shared
  vector. The backend now stores [world, component] limits; inject into component
  [0, 0] using that same indexed limit. Physics and assertion tolerances were not
  changed.

Validation logs are on GPU7 under `runs/codex_review_20261111/`. The first root
invocation omitted `runs/sac_env.sh` and eight physics tests failed at the missing
`TORCH_EXTENSIONS_DIR`; that invocation did not reach those physics checks.
With the environment sourced, 260 passed / 5 skipped and the stale test indexing
failed. Correcting that indexing made the physical reset/force-stop test pass.
Final validation, with the documented environment and optional MPPI dependencies:
root **261 passed / 5 skipped** (186 s); backend **28 passed** (4 s).
The remaining warnings are from upstream DiSECt's deprecated Python AST API.
Logs: `root_final.log`, `backend.log`, `reset_limit.log` and `targeted.log`.
The 5 skipped tests are not counted as passes. Changed-document links and
`git diff --check` also pass.

The process tests use actual spawned CPU workers for clean exit, abrupt exit and
stuck-worker cleanup. They also cover partial setup failure and pre-loop learner
failure; these are implementation checks, not cutting-performance results.
Evaluation interface matching and policy-independent case sampling remain
explicitly deferred in PLAN. S0 limits/material ranges and the method objectives
are not changed by this patch.

## 2026-11-11 — S3 first two arms launched (multi-object, from scratch)

Run record. Code `a28898e` (A6000 checkout synced from GPU7 by git bundle; clean);
config `configs/multi_condition/s0_train.json` (sha256 `ab375876d371722a…`); backend
scenes copied from GPU7 with identical checksums (prism `1bc719ce…`, potato_003
`b9c0cbe9…`, cucumber_003 `f619a823…`); A6000 robot model `robot.mjb` and controller
parameters identical in content to GPU7.

- Task: prism, potato_003, cucumber_003, 256 worlds each (768); per episode Young's
  modulus and cut-spring stiffness ×0.5–2, density ×0.8–1.25, target speed 1–5 cm/s;
  per world knife start 0.5–3 cm above the object; objects held by their bottom 6 mm.
  Force limits `[50, 50, 50 N, 5, 5, 5 N·m]` for every object — a simulation setting
  chosen for this run, not a real-robot safety threshold. `w_speed` 1.0.
- Initialization: fresh weights, seed 0, 500 WM-only updates, then joint training;
  critic `real`, actor horizon 8, BC 0.25, microbatch 64, critic microbatch 8192.
- Budget (both arms): 5000 iterations × 768 worlds = 3.84 M physical transitions
  (`--max-iterations 5000`, 48 h wall cap). Checkpoints every 250 iterations plus the
  stage checkpoints; replay snapshot with each checkpoint.
- Arms: `runs/s3_base_20261111` (GPU1, W&B `0xvf8zj7`); `runs/s3_risk_20261111`
  (GPU0, `--risk-constraint --risk-estimator quantile`, W&B `5r1geoka`). Only the risk
  switch differs. SAC and the ensemble comparator are not started.
- Throughput at start: ~15 s physics per iteration (three groups share one GPU at 100%
  utilization), i.e. roughly a day per arm.
- Logged: per-episode rows (object, sampled conditions, ending, return, steps, peak
  material force, peak limit ratio, released springs) in `episodes.jsonl`; per-object
  success/force-limit/workspace/time-limit rates, returns and peaks, WM/critic/policy
  metrics in W&B and `updates.jsonl`.

#### 2026-11-11 — base arm stopped at iteration 1474: potato_003 material instability

- Failure: `FloatingPointError: Nonfinite DiSECt material state` in the potato_003 group
  process (iteration 1474, joint stage, 74 policy updates). Not a learning failure.
- Cause (measured): with the exported substep 1e-5 s, the potato_003 mesh diverges at the
  first step with the knife at rest at Young ×2 / density ×0.8 (inside the S0 range), for
  every cut-spring scale. Prism and cucumber_003 are stable on the full grid
  (Young ×1–2, density ×0.8–1.25, cut springs ×0.5–2; rest plus reference cut). The
  explicit step limit scales with √(mass/stiffness); log-uniform sampling reaches this
  corner rarely, so the run met it only after ~2500 episodes.
- Fix: per-group DiSECt substep (`sim_dt` in the object group config); potato_003 uses
  8e-6 s in every S0 config. At 8e-6 s it is stable up to Young ×3 / density ×0.8, and
  reference-cut peak forces equal the 1e-5 s values within 0.1 N where both are stable.
  Force limits, success criteria and material ranges are unchanged.
- Resume: `--resume` now also restores the replay snapshot written with the checkpoint
  (checked to belong to that checkpoint and iteration); episodes open at the checkpoint
  are closed so no training window spans the resume. Both arms continue from their own
  latest checkpoint with this code, so they keep identical physics.

## 2026-11-12 — S3 first runs stopped; task-definition diagnosis

Runs (all stopped; training data deleted, see DATA): base (iteration 3819/5000),
base + quantile risk penalty (2939/5000), official SuperMimic adaptation on the prism
(3750/3750 completed). Numbers below marked *training* come from training episodes
with exploration noise; *deterministic* from `diag_rollout` (32 worlds per object,
`s0_eval_train`, base checkpoint 3500).

**Outcomes under the then-current task definition**
- Base (*training*, iterations 2500-2790): prism 65% success / 35% force limit;
  potato_003 25% / 75% workspace; cucumber_003 0% success.
- Risk arm: worse than base (potato 0% from iteration 2000, prism 7-50%). The dual
  variable rose 10 -> 4269 because the 2% force-limit target was not reachable; the
  penalty term (about 26-77 per step against a task reward of about -1.4) dominated
  the actor objective. Risk head itself: coverage 0.88-0.98, event recall 0.72-0.88.
- Official SuperMimic adaptation: 25 successes in 170,578 episodes. The cutting step
  reward is <= 0, so the official soft termination ends negative reward early; an
  adaptation error, not evidence about the official algorithm.

**Diagnosis (deterministic policy, base checkpoint 3500)**

| Object | Ending by the task definition | Cut springs released >= 95% |
|---|---|---|
| prism | 22 goal, 10 force limit | 22/32 |
| potato_003 | 2 goal, 30 workspace (x) | 32/32 |
| cucumber_003 | 0 goal, 29 workspace (x), 3 force limit | 30/32 |

- Potato and cucumber are fully cut by control step 10-13. The knife then stays
  2-10 mm above the goal height with a residual force (13 N potato, 34 N cucumber),
  never satisfies the 2 mm goal tolerance for 10 steps, drifts in x at a constant
  rate and leaves the 5 cm workspace limit.
- Knife speed: 20-27 cm/s peak downward speed against commanded 1-5 cm/s. Prism
  force-limit endings occur at about 22 cm/s (58 N at step 8). Under the normalised
  weights the step cost (-0.2) and the +100 goal bonus outweigh the speed term
  (at most -0.2 per step), so fastest is optimal; x motion has no cost.
- Reference trajectory on the same conditions: goal in 6/32 prism, 1/32 potato,
  2/32 cucumber worlds, only where the start lift is about 0: it is a fixed
  displacement and stops short by the start lift. With lift ~0 the potato reference
  ends at 1.8 mm from the goal with 14.7 N residual force (inside the tolerance by
  0.2 mm); cucumber_003 holds 47 N after all 65 springs are released.

**Hold test (reference trajectory, default start, force limit disabled)**

| potato_003 hold | Cut springs | Task ending | Object centroid shift | Peak force |
|---|---|---|---|---|
| bottom 6 mm, +-1 cm around the cut plane free (current) | 245/245 | not goal in 120 steps | 0.1 mm | 20.6 N |
| bottom 6 mm, +-3 cm free | 56/245 | goal | 34.6 mm | 33.3 N |
| no static vertices | 57/245 | goal | 32.3 mm | 34.0 N |

cucumber_003 current hold: 65/65 cut, not goal, 37.2 N before cut-through
(the +-3 cm and no-hold variants: see the next entry when measured).

**Conclusions**
- "Goal = knife at one fixed pose" is wrong in both directions: a fully cut held
  object is not a success, and an object pushed away uncut is.
- Unheld objects are pushed away (ground friction 0.2 does not keep them).
- The earlier "contact-onset force spike" observations were made with policies
  moving at several times the commanded speed; they do not establish a world-model
  problem. The earlier cut-force table for cucumber_003 (34-41 N) includes the
  squeeze of the bottom-held halves.
- Training code (WM, BPTT, resume, logging) showed no defect in this diagnosis.

#### 2026-11-12 — hold test, cucumber_003 (same protocol as the potato table above)

| cucumber_003 hold | Cut springs | Task ending | Object centroid shift | Peak force before cut-through |
|---|---|---|---|---|
| bottom 6 mm, +-1 cm around the cut plane free (current) | 65/65 | not goal in 120 steps | 0.1 mm | 37.2 N |
| bottom 6 mm, +-3 cm free | 65/65 | not goal in 120 steps | 0.3 mm | 23.2 N |
| no static vertices | 65/65 | goal at step 39 | 2-3 mm | 5.6 N |

Most of the force measured on the held cucumber is the squeeze of halves that cannot
spread: unheld, the same cut needs 5.6 N and the knife reaches the goal. The unheld
potato is pushed 32 mm and stays uncut (table above). The bottom hold therefore
changes the measured cutting force by a factor of up to about 7 and is not a neutral
way to keep objects in place; how objects are constrained has to be decided per object
geometry (scene preparation, not policy input) before any force limit, success rule
or reward is fixed. The 2026-11-11 "force needed to cut the held scanned objects"
table measured this artefact for cucumber_003.

## 2026-11-12 — Sawing check: the simulated force signal is direction-blind

`scripts/measure_sawing.py` (reference trajectory, 3 cm/s downward, optional lateral
sinusoid along the blade, force limit disabled).

- Fixed-bottom prism, Young x1 and x2, +-2 cm at 1 Hz (knife lateral speed up to 8.7 cm/s,
  verified in the DiSECt knife state): mean 5.33 vs 5.34 N, peak 8.43 vs 8.42 N, same
  cut-through step; the per-step force series agree to about 0.1 N. Sawing has no effect.
- Cause 1 (code): the sensor force is `sum_i |f_i|` over per-spring knife contact forces, put
  on the vertical axis with zero torque (`coupling.feedback`, inherited from SliceIt's bridge;
  DiSECt's own knife force is the same scalar). Contact-force directions are discarded.
- Cause 2 (parameters): tangential friction in `knife_contact_force` scales with
  `|v_t| / sdf_kf`; with the official `sdf_kf = 50` it is negligible at cutting speeds.
- Held cucumber_003, pure press: the scalar reaches 35.4 N while the vector sum of the same
  contact forces is 1.0 N vertical and 2.7 N sideways. The scalar mostly counts the opposing
  squeeze of the two cut faces on the blade; with sawing the vector sum changes by about 0.1 N.

Consequences: (1) in the current simulation a policy has no reason to saw; (2) the simulated
"force" is a calibrated proxy (DiSECt fits this scalar to measured force), not the net wrench
a wrist sensor measures, and it carries no torque, so the torque limit is never active;
(3) the hold-test forces above are values of this proxy. A sawing task needs a net-wrench
force signal and a friction model calibrated to real curves (DiSECt reports real mean force
reductions above 40 % from slicing).
