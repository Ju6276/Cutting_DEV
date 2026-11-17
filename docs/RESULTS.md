# Results

Dated measurements from 2026-11-11 on (multi-object task). Earlier entries (2026-11-02..10, fixed
prism development) are in [archive/RESULTS_20261102_20261110](archive/RESULTS_20261102_20261110.md).
[PLAN](PLAN.md) says what happens next; old next-step prose in entries is not a plan.

## Current evidence index — 2026-11-14

- Selected task direction: target-pose completion plus board boundary; see TASK_FREE_CUT and PLAN. v1-v3 separation and v4 coverage results retain their own definitions.
- Latest v3 checkpoint: 543 policy updates. The 64-case board audit found old successes involving knife penetration; do not treat its old success rate as physically valid full cutting.
- Board execution boundary: physically tested; not a board-force model. Separation and path completion differ.
- Current three-object preparation uses free cucumber, free prism and the flat-bottom half potato. Half-potato randomized preparation passed 144 conditions; the earlier moving whole-potato scene is historical.
- Force/contact parameters are not calibrated for this project's real robot. Ground contact is a finite penalty law. All-FEM-node minima mix real and ghost geometry; the latest tofu diagnostic separates them (see GROUND_CONTACT_REVIEW). No task difficulty was created by tuning ground contact.
- Earlier scalar feedback, fixture tests, force-risk runs and prior next-step statements below are historical. The 2026-11-11 S3 rates and retired perturbation protocol are not current paper evidence.

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
  verified in the DiSECt knife state): material proxy mean 5.33 vs 5.34 N, peak 8.43 vs 8.42 N, same
  cut-through step; the per-step force series agree to about 0.1 N. Sawing had negligible effect in these tested configurations.
- Code observation (not an isolated causal experiment): the material-feedback scalar is `sum_i |f_i|` over per-spring knife contact forces, put
  on the vertical axis with zero torque (`coupling.feedback`, inherited from SliceIt's bridge;
  DiSECt's own knife force is the same scalar). Contact-force directions are discarded.
- Parameter observation: tangential friction in `knife_contact_force` scales with
  `|v_t| / sdf_kf`; with the official `sdf_kf = 50` it is negligible at cutting speeds.
- Held cucumber_003, pure press: the scalar reaches 35.4 N while the vector sum of the same
  contact forces is 1.0 N vertical and 2.7 N sideways. The scalar mostly counts the opposing
  squeeze of the two cut faces on the blade; with sawing the vector sum changes by about 0.1 N.

Interpretation: tested variants did not exhibit the intended sawing benefit.
This does not establish that all scenes, trajectories or DiSECt settings behave
identically. The material bridge contributes a force proxy and zero material
torque. `coupling.feedback` adds it to the MuJoCo sensor-body wrench before filtering.
That wrench is the tared inverse-dynamics load of the distal links and tool; DiSECt forces
are not applied to the MuJoCo robot, so it contains no material reaction. It was
0.001-0.003 N·m in the reference cuts of `runs/food_force*_20261111` (limit 5 N·m), so the
torque limit was inactive there; not measured for policy rollouts. Summed magnitudes are
not net contact wrench. DiSECt fits this scalar to measured force; that does not establish
calibration of these exported scenes.

With this bridge and `sdf_kf = 50`, insensitivity to lateral motion is structural (per
contact |f| does not depend on the sliding direction, and friction is negligible), so a
policy has no incentive to saw. DiSECt reports real mean force reductions above 40 % from
slicing, which this configuration does not reproduce.

Changing the scalar to a vector sum alone is not a validated remedy for sawing:
material/contact/friction models and mounting still need a physical basis.
These are proxy measurements under recorded holds, not minimum real cutting forces.

## 2026-11-12 — Consistency cleanup after task review

No new experiment or physics/learning change. Entry descriptions and documents
distinguish historical tasks from open decisions. S0 parameters/paths remain;
their directory documents the retired task. B1 final data is marked used.

The sawing tool's L2 summary of per-axis temporal maxima is relabeled a
component-peak envelope: axes may peak at different times. Computation is
unchanged; prior JSON labels remain historical records. Native robot wrench and
material-proxy contributions are distinguished. Validation accompanies the cleanup
commit. These changes do not authorize training.

Validation on GPU7 (configured environment, optional MPPI enabled): root 273
passed / 5 skipped; backend 29 passed. Upstream DiSECt AST deprecation warnings
remain. Changed Markdown relative links and Git whitespace pass. Executable ASTs
are unchanged after removing docstrings and accounting for the sawing output-label
changes; no scene/config JSON changed. Logs: `runs/codex_consistency_20261112/`.
These checks establish cleanup consistency, not physical task validity.


## 2026-11-12 — Codex net-wrench repair and physical rerun

This fixes the optional net sensor interface and its measurements, not task success,
friction calibration or a learned sawing policy. Historical scalar reproduction is
unchanged. User authorized these repairs and a real simulation check; no learning ran.

Confirmed defects corrected:

- The final source observation transform mixed moment into force: a pure 2 N m
  couple and 0.3 m offset produced a fictitious 0.6 N force. Net observations/limits
  now use sensor axes/about-sensor moments, matching controller input and history;
  replay restoration follows the same convention. Scalar retains source behavior.
- Knife moments used particle positions after integration with forces evaluated
  before it. The last force-evaluation positions are now cached per world and
  covered by pause/reset checks.
- The last 2 ms sample could escape the net limit check before pose success.
  It is now checked in the same control step.
- The measurement's integer world indices were passed to a boolean material mask,
  skipping world 0. The rerun deliberately uses E scales in order [2, 1].
- End-of-step vertical force and within-step maximum total force were mixed.
  Schema `simultaneous_coupling_samples_v2` records simultaneous 500 Hz vectors and
  uses a common contact-to-release interval. Contact onset remains an explicit
  0.5 N sum-of-contact-magnitudes diagnostic; release fraction is 0.98.
- Net work now includes lateral and angular power, using the moment about the
  knife origin. It no longer multiplies total force magnitude by downward speed.
- Nondefault force/friction modes record their revision and effective friction
  values; multi-object metadata retains per-group values and common signal frames.

Physical check on GPU7: fixed-bottom prism, down 3 cm/s, pure press vs
±2 cm at 1 Hz, net feedback, friction transition 0.01 m/s. Force-limit termination
disabled only for this measurement, matching the previous measurement design.
Budget cap was 20 minutes; completed in 159 + 155 seconds. All four runs reached
the diagnostic 98% release fraction; final releases were 298–304 of 304 springs.
The script's `ending=success` is still historical goal-pose completion, not a
separate validation of material separation.

| E scale | Trajectory | 98% release time (s) | Vertical mean (N) | Net-force mean / peak (N) | Filtered sensor torque max component (N m) |
|---|---|---:|---:|---:|---:|
| 2 | press | 2.332 | 3.883 | 3.884 / 6.433 | 0.847 |
| 1 | press | 2.548 | 3.134 | 3.135 / 5.665 | 0.738 |
| 2 | saw | 2.264 | 1.552 | 3.271 / 5.232 | 0.943 |
| 1 | saw | 2.328 | 1.362 | 2.737 / 4.909 | 0.861 |


For E x1, vertical mean fell about 57%, while total-force mean fell about 13%.
Do not describe this as halving total force, or compare these means with the old
mean-of-step-peaks statistic. These are prescribed trajectories in one parameterized
scene, not learned-policy results or validation against real cutting forces.

Evidence: `runs/codex_net_wrench_fix_20261112/press_saw/report.json` includes
all samples, code hashes, scene hash and command; `report_check.json` independently
recomputes summaries with NumPy and verifies finite, uniformly sampled traces and
matching source hashes. `implementation.patch` captures the pre-commit implementation.
Backend final suite: 32 passed (3.97 s). Root final suite: 286 passed / 5 skipped
(196.48 s), with CUTTING_BACKEND_ROOT set and optional MPPI environment enabled.
The five skips are legacy SAC/TensorFlow checks (missing gym, source checkpoint,
explicit bridge opt-in, and Python 3.8 tf2rl); the new GPU checks executed.
Logs: `backend_final.log` and `root_final.log` in the same run directory.
Tests include pure-couple and point-force examples, full rigid-transform power
invariance, four-world contact cache/pause/reset, net snapshot restoration, and the
last-sample torque-limit event.

Separate source finding, not silently changed here: the bundled DiSECt contact
kernel evaluates a point with endpoint weights (1-c, c), but distributes that
contact force with (c, 1-c). Consequently its nodal moment need not equal the
contact-point moment. METHOD records the exact scope: this repair reports the
wrench at the source's evaluated contact point; it does not validate angular
momentum conservation of the material discretization. Fixing that source formula
would change material dynamics and is distinct from sensor-interface correction.


## 2026-11-12 — Contact-coordinate correction and kernel confirmation

User authorized confirming and fixing the contact-transfer defect. This supersedes
the *unresolved source caveat* in the preceding interface-repair entry; its numerical
results remain valid records of the old implementation (`38b85a5`).

**Root cause correction to the earlier explanation.** The earlier audit looked at
the consumers and called it reversed nodal weights. Following the producer shows
why swapping weights alone would be incomplete: the SDF search computes distance
and normal at `(1-u)*x1+u*x2` but stores `1-u`. The force routine then gathers
position/velocity with `(1-c,c)` and scatters force with `(c,1-c)`. The search point
and nodal distribution match each other, but the force evaluation and reported
moment use the opposite point. Return `c=u` and scatter with `(1-c,c)` so all four
uses agree. The optional triangle branch also needs the latter distribution
correction; its coordinate producer already has the proper convention.

This was reproduced by launching the **production CUDA kernel in isolation**.
In a test where the allowed search interval is [0,0], the old kernel returned
coordinate 1 on both sides. Its nodal/contact moment discrepancy reached
0.164082 N m and its virtual-power discrepancy 1.449986 W. The corrected kernel
returns 0 and both residuals are zero. In four worlds with nontrivial search
coordinates, maximum residuals changed from 0.128895 N m / 0.761252 W to
2.29e-9 N m / 7.16e-9 W. An independent second SDF query at the reconstructed
point matches the original distance; both sides and triangle force scattering
are covered. These check local transfer consistency, not whole-system energy
conservation or calibration.

**Version and scope.** `contact_discretization_revision=edge_coordinate_consistent_v2`
is recorded for both scalar/net and propagated through grouped environments.
It deliberately distinguishes this material solver from old checkpoints. No
material/friction coefficient, search-iteration count, damage rule, timestep,
force threshold or reward was tuned. Historical Gazebo source patches are not
modified; old bundled behavior remains reproducible from `38b85a5` in an isolated
checkout/cache. No learning was started.

**One existing test failed initially; the failure is retained.** At 4 ms the
packed-vs-single comparison differed by about 0.495 in one spring's residual
stiffness (spring 233; 15.68 vs 16.18). Three repeats also showed the same gap
between identical single-world runs, with particle-position differences below
2.5e-7 m. This establishes that the failed equality is not specific to batching.
A separately synchronized substep trace did not reproduce the split, so the exact
trigger was not isolated. Float32 atomic accumulation and hard contact/damage
branches preclude assuming long-run per-spring identity.

The material-packing test now checks same-input **one-substep** q/stiffness
agreement at the original tolerances, then retains its 4 ms position comparison
and adds nonnegative, irreversible damage checks. No tolerance or physics threshold
was increased; the old 4 ms per-spring equality claim is explicitly withdrawn.
Logs: `backend.log` (initial failure), `repeat_diagnostic.json` and
`contact_branch_trace.json`. Final backend suite: **36 passed** (3.91 s). Root suite with GPU assets and
optional MPPI enabled: **286 passed / 5 skipped** (197.95 s). Skips are the same
legacy SAC/TensorFlow dependencies/opt-in checks, not the new CUDA checks.
Logs are `backend_final.log` and `root.log` in this run directory.

**Physical rerun**, same fixed-bottom prism and settings as the preceding entry:
E scales [2,1], down 3 cm/s, press vs ±2 cm at 1 Hz, net feedback, friction
transition 0.01 m/s. Force-limit termination disabled only in this diagnostic.
20-minute cap; completed in 158 + 155 seconds. Same scene hash, finite 500 Hz
traces, summaries independently recomputed with NumPy.

| E scale | Motion | Mean net force before → after (N) | Peak net force before → after (N) | Final released springs |
|---|---|---:|---:|---:|
| 2 | press | 3.884 → 3.889 | 6.433 → 6.439 | 300/304 |
| 1 | press | 3.135 → 3.152 | 5.665 → 5.670 | 298/304 |
| 2 | saw | 3.271 → 3.279 | 5.232 → 5.238 | 304/304 |
| 1 | saw | 2.737 → 2.748 | 4.909 → 4.892 | 301/304 |


All four reached the **98% spring-release diagnostic** and the historical pose
goal; this is not a newly validated separation-success definition. Mean net-force
changes in this single rerun are below 0.6%, with no evidence here of a large
force-performance change. A mathematically demonstrated consistency bug should be
fixed even when the aggregate change in this particular scene is small; do not
claim a general performance improvement.

All data: `runs/codex_contact_coordinate_20261112/`. `before.json` / `after.json`
contain kernel outputs; `press_saw/report.json` contains every sample and source
(including the modified kernel) and scene hashes; `comparison.json` contains the
independent summary and old/new comparison. Documentation and measurement checks
do not reopen retired S3 training.

## 2026-11-12 — Contact repair re-review and documentation reconciliation

Re-reviewed `38b85a5` / `4b229df` at the user's request. The coordinate producer,
both cut-side consumers, optional triangle scattering, force sign and pre-integration
contact-point cache agree with the corrected transfer identities. No further
physics-formula change was made. The eight source hashes in the previous physical
press/saw report still match current source; that rerun remains applicable.

Fresh GPU7 checks: **backend 36 passed** (3.96 s); **root 286 passed / 5 skipped**
(198.83 s), with GPU assets and optional MPPI enabled. The skips are the same legacy
SAC/TensorFlow dependency/checkpoint/opt-in cases; production CUDA contact tests ran.
Logs and source-hash verification:
`runs/codex_contact_recheck_20261112/{backend.log,root.log,physical_source_check.json}`.

Corrected current documentation's stale scalar-only description, temporal peak
envelope description, blanket negligible-sawing finding, unmodified-source claim,
unclamped real-specimen prescription, and obsolete P2/P3 reference. Clarified that
`38b85a5` is an immediate pre-fix reference, not every historical run's exact code.
Repaired **31 relocated archive links** without changing archived conclusions;
**219 local Markdown link targets** exist (external URLs/heading anchors were not
validated by this check). `git diff --check` passed.

Two test comments had overstated a contact-branch switch as the proven trigger of
long-run damage variation; corrected them. The exact trigger remains unresolved,
and replacing the 4 ms per-spring equality check with single-step consistency plus
longer position/damage-bound checks narrows coverage. No assertion, tolerance,
material parameter, reward or task threshold was changed in this re-review.

Review handoff: [CONTACT_FIX_REVIEW](archive/CONTACT_FIX_REVIEW.md), including derivation,
evidence, scope boundaries and the test change requiring explicit review.
No training restarted; Claude's untracked recording script was left untouched.

## 2026-11-12 — Press vs sawing on three objects (net feedback), and Claude's check of the contact repairs

**Measurement** (`runs/task_diagnosis_20261112/sawing_net/{prism_cucumber,potato}/report.json`;
`measure_sawing.py` at `9b6357a`, i.e. **before** the contact-timing and edge-coordinate repairs
of `38b85a5`/`4b229df`; Codex's prism rerun above changed mean net force by under 0.6 %).
Reference trajectory, 3 cm/s downward, bottom-held scenes, net feedback, friction transition
0.01 m/s (DiSECt paper default, not calibrated), force limit disabled. Values are end-of-control-step
samples of the net contact force between first contact and 98 % spring release (a diagnostic
window, not an approved completion rule). Young scale 1 unless noted.

| Object | Motion | Vertical mean / peak (N) | Along blade mean (N) | Net norm mean / peak (N) | Peak sensor torque (N·m) |
|---|---|---|---|---|---|
| prism | press | 3.10 / 5.61 | 0.03 | 3.10 / 5.61 | 0.74 |
| prism | saw 2 cm, 1 Hz | 1.41 / 3.63 | 2.14 | 2.70 / 4.64 | 0.86 |
| prism | saw 2 cm, 2 Hz | 0.97 / 2.97 | 2.41 | 2.64 / 4.70 | 1.09 |
| potato_003 (8e-6 s) | press | 5.81 / 11.74 | 0.05 | 5.90 / 11.97 | 1.54 |
| potato_003 | saw 2 cm, 1 Hz | 2.50 / 7.48 | 3.95 | 5.08 / 11.70 | 2.27 |
| potato_003 | saw 2 cm, 2 Hz | 1.58 / 4.94 | 4.41 | 4.84 / 11.33 | 2.07 |
| cucumber_003 | press | 11.09 / 23.41 | 0.17 | 11.10 / 23.45 | 3.20 |
| cucumber_003 | saw 2 cm, 1 Hz | 4.31 / 14.14 | 7.21 | 8.87 / 21.72 | 2.80 |
| cucumber_003, Young x2 | press | 17.04 / 30.68 | 0.86 | 17.10 / 30.72 | 4.33 |
| cucumber_003, Young x2 | saw 2 cm, 1 Hz | 6.50 / 19.87 | 11.73 | 14.05 / 33.57 | 4.07 |

- Sawing lowers the vertical component (mean -55 to -73 %, peak -35 to -58 % at Young x1) and
  moves force into the blade direction. The norm of the net force falls much less (mean -13 to
  -20 %, peak 0 to -17 %) and rises for the stiff cucumber (peak 30.7 -> 33.6 N).
- DiSECt's real-robot result is a reduction of the force **norm** (mean above 40 %). This model
  reproduces friction redirection only; the fracture-side effect of slicing is not in DiSECt's
  damage law. Which quantity a force limit applies to (per component vs norm) therefore decides
  whether sawing helps in this simulation.
- potato_003 at Young x2 released 239-240 of 245 springs in every motion (below the 98 % window).
- Illustration: `runs/videos_sawing_20261112/prism_press_vs_saw.mp4` (`record_reference_cut.py`;
  its force trace is the norm of the net force).

**Check of `38b85a5` and `4b229df`** (read against the kernel source, no new runs):
`edge_knife_sdf(u, a, b)` evaluates `a*u + b*(1-u)` and the search calls it with `(u, xj, xi)`, so
the search point is `(1-u)*xi + u*xj`; upstream stored `1-u`, and the consumers interpolated
position and velocity with `(1-c, c)` (the mirrored point) while scattering force with `(c, 1-c)`
(the search point). Storing `u` and using `(1-c, c)` everywhere is consistent. The contact points
of `knife_wrench()` at `9b6357a` used the mirrored coordinate and post-integration positions; both
are corrected. `rigid_wrench_transform` was not changed by the repair. Agreed with the review
handoff's open item: the replaced 4 ms per-spring equality test reduces long-horizon coverage and
its trigger is unexplained.

## 2026-11-12 — Free-object single-cut task: platform check on one scene

Task decision (user): the object is placed freely on the board, one cut, slight passive sliding
allowed, no active rolling or repositioning; new fixture schemes are cancelled. Success means the
slice is fully separated from the body (no remaining material connection) without a force-limit
event; a remaining connection is recorded as incomplete.

Scene: `configs/free_scenes/cucumber_003.json` (the scanned cucumber with no static vertices; ground
contact and friction 0.2 as exported) -> `backends/mujoco/runs/sweep/free_cucumber_003`. Check:
`scripts/check_free_object_scene.py`, net feedback, friction transition 0.01 m/s (not calibrated),
force limit disabled; knife motion: 1 s rest, press at 3 cm/s to the scene goal height, 0.5 s hold,
3 cm retraction, 0.5 s hold. Reports: `runs/free_object_check_20261112/cucumber_{press,saw}/report.json`.

**Severed-state check.** The tetrahedral mesh has exactly two connected components (175 and 153
particles) and all 65 cut springs join one to the other, so the cut springs are the only material
connection across the cut plane. "All cut springs released" is therefore the separation check for
this scene; the released fraction is a progress diagnostic only.

| Young scale | Motion | Released at end | Separated | Peak net force | Side displacement at the end (mm; x across the cut, y up, z along the blade) |
|---|---|---|---|---|---|
| 1 | press | 65/65 (step 63) | yes | 3.0 N | (1.3, -1.8, -2.3) and (-0.8, -2.7, -2.4) |
| 2 | press | 62/65 | **no** | 16.4 N | (-0.6, **+23.8**, -3.1) and (-4.2, +5.5, +4.5): the object is lifted with the retracting knife, 10 N still acting |
| 1 | press + saw 2 cm, 1 Hz | 65/65 (step 63) | yes | 1.9 N | (1.7, -1.8, 11.0) and (-1.4, -3.2, -2.5) |
| 2 | press + saw 2 cm, 1 Hz | 65/65 (step 64) | yes | 16.3 N | (2.0, -1.8, 12.8) and (-0.8, -2.7, 13.4) |

- **Object-board contact:** the unheld object stays on the ground (lowest particle 0.55 mm below the
  ground plane, i.e. within the 1 mm contact radius) and all states stay finite. During the 1 s rest
  it settles by about 2 mm downward and 2 mm along the blade and still moves at about 2.7 mm/s at the
  end of the rest: the exported state is not a settled state.
- **Knife contact:** finite net wrench; the knife cuts without pushing the object away across the cut
  (under 2 mm), and sawing drags the pieces 7-13 mm along the blade.
- **Completion is condition dependent:** at Young x2 the pure press does not separate the object and
  lifts it on retraction, while the same press with sawing separates it. One scene, one saw setting,
  uncalibrated friction: an observation, not a general result.

Assessment: on this scene the platform expresses the three things the task needs (free object on the
board, knife contact with a net wrench, a well-defined severed state), with two known issues to handle
in the task definition rather than by fixing objects: episodes must start from a settled object, and
ground/knife friction are uncalibrated simulation settings.

#### 2026-11-12 — Settled start; correction of the free-cucumber table above

`settle_seconds` advances the material with the knife held at its (non-touching) start before the
reset state is saved (0.75 s used). With it the 1 s rest moves the object by under 0.06 mm.
From the settled start (`cucumber_{press,saw}_settled/report.json`):

| Young scale | Motion | Released | Separated | Peak net force |
|---|---|---|---|---|
| 1 | press | 65/65 | yes | 2.4 N |
| 2 | press | 65/65 | yes | 1.5 N |
| 1 | press + saw 2 cm, 1 Hz | 65/65 | yes | 2.5 N |
| 2 | press + saw 2 cm, 1 Hz | 65/65 | yes | 1.9 N |

**Corrections to the entry above.**
- The Young x2 press that stayed incomplete at 16.4 N started from the unsettled export, in which
  that world's object had moved about 1 mm across the cut plane during the rest; from the settled
  start the same press separates the object at 1.5 N. The difference between press and saw in that
  table is therefore not evidence about sawing or about a need for an adaptive policy. The fixed
  press and the fixed saw both complete this scene and stay as reference comparisons.
- "The object stays on the board" holds for rest and cutting; in the incomplete case the object was
  lifted about 24 mm with the retracting knife.
- These runs had the force limit disabled: "separated" is not "task success under the limit".

**Platform limit seen here:** the cut plane is fixed in the object and the knife is constrained to
one plane; about 1 mm of sideways offset between them changed the peak force from about 2 N to
16 N. Sliding along the blade is represented; sliding across the cut plane is not represented
faithfully. Scope of the separation check: this scene only; new exported objects need the same
topology check during asset preparation.

## 2026-11-12 — Free cucumber re-measured: per-world settling, 2 ms sampling (supersedes the two tables above)

Review corrections accepted: (1) the two earlier free-cucumber tables used step-end samples of the net
force (one value per 0.05 s), not the 2 ms samples the limit check uses; (2) the "settled" table settled
every world with the nominal material and then changed the material, so its Young x2 rows started from
a state that does not belong to that material; (3) the explanation "about 1 mm sideways offset causes
16 N" was a guess from two runs that differed in initialization, material order and state, and is
**withdrawn**; (4) "this scene poses no force difficulty" is **withdrawn**.

Implementation: `GPUEnvironment.settle()` / `CuttingBatch.settle()` settle every world with its own
material scales and save that as the reset state; the contact check during settling covers every
material substep (`advance(track_peak=True)`), not only the last one. `check_free_object_scene.py`
samples the net contact force and the six-component observation wrench at every 2 ms coupling step.

Free cucumber, 0.75 s settling per world, 0.25 s rest, press at 3 cm/s to the scene goal height, 0.5 s
hold, 3 cm retraction; net feedback, friction transition 0.01 m/s (uncalibrated), force limit disabled;
utilisation is reported against the old S0 simulation limits [50, 50, 50 N, 5, 5, 5 N·m].
Reports: `runs/free_object_check_20261112/cucumber_{press,saw}_2ms/report.json`.

| Young / cut-spring scale | Motion | Separated | Peak net force, 2 ms (N) | Peak sensor force component (N) | Peak sensor torque (N·m) | Peak utilisation | Notes |
|---|---|---|---|---|---|---|---|
| 1 / 1 | press | yes, first hold step | 7.7 | 5.3 | 0.75 | 0.15 | pieces move under 2 mm |
| 2 / 1 | press | **no** | 18.0 | 16.3 | 3.20 | 0.64 | object lifted about 25 mm with the retracting knife |
| 2 / 2 | press | yes, hold | 19.8 | 18.0 | 2.43 | 0.49 | |
| 1 / 1 | press + saw 2 cm, 1 Hz | yes, first hold step | 2.7 | 1.5 | 0.29 | 0.06 | pieces dragged 12-16 mm along the blade |
| 2 / 1 | press + saw | yes, hold | 14.5 | 12.1 | 1.99 | 0.40 | one piece displaced 31 mm across the cut |
| 2 / 2 | press + saw | yes, hold | 23.0 | 14.0 | 2.39 | 0.48 | |

- Rest motion after per-world settling is under 0.04 mm in every world.
- The free object does not have uniformly small forces: 2 ms peaks range from 2.7 to 23 N and peak
  utilisation from 0.06 to 0.64 over these six cases; step-end sampling understated them
  (e.g. 5.2 N vs 7.7 N, 0.9 N vs 2.7 N).
- With each world settled for its own material, the Young x2 press again leaves the object attached
  (not separated) while the same press with sawing separates it. One scene, one saw setting,
  uncalibrated friction, limit disabled: an observation that the motion changes the outcome, not
  evidence that a learned policy is needed. Fixed press and fixed saw stay as reference comparisons.
- Not decomposed: the recorded knife force combines contact normal force, damping and friction;
  no reading here separates "fracture resistance" from "pinching". Earlier statements attributing about
  2 N to fracture and the rest to the hold are withdrawn; what is supported is that the hold and the
  initialization change the knife force strongly.
- Validation status: neither the free nor the held scenes are validated against this project's real
  cuts. The scanned objects carry material values copied from official DiSECt configs onto different
  meshes with our scene settings; similarity of the held forces to the paper's curve is an
  order-of-magnitude comparison only (the paper's 12-15 N is the late plateau of the curve, not a mean).


## 2026-11-12 — free_cut_v1 implementation and rejected reward candidate

User authorized the opt-in task implementation and verification, not long training.
Parent e529161; see [FREE_CUT_REVIEW](archive/FREE_CUT_REVIEW_20261112.md) and the versioned contract
[TASK_FREE_CUT](TASK_FREE_CUT.md). No historical pose task or material parameters
were replaced.

Implementation: complete-separation settlement, force/workspace failure precedence
(charged once), true finite deadline/no bootstrap, deterministic step_count in the
WM/critic, bounded supervised progress and separation-only completion head.
The same completion probability weights expected completion reward and survival.
Policy input stays 10 x 24 observations + the 8-D pose/speed goal. Material-randomized
resets re-settle selected worlds from immutable exported state, preserving live worlds.

Verification on GPU7:
- Root suite with CUTTING_BACKEND_ROOT enabled and the MPPI Python: **298 passed,
  5 skipped**, 232.39 s (final code). Skips are legacy gym/TensorFlow assets and opt-in source bridge.
- Backend suite: **36 passed**, 4.83 s.
- New tests include finite-difference BPTT, learnable force outputs at the source zero initialization, an actual small supervised WM optimizer
  update, deterministic horizon absorption, true terminal timeout records, masked
  GPU reset and both local/process grouped environments. These do not establish
  fitted WM accuracy or successful policy training.

Physical task/reward check (force limits ACTIVE): one free cucumber, nominal
material, 0.75 s settling, target 3 cm/s, net contact, friction transition 0.01 m/s,
limits [50,50,50 N,5,5,5 Nm]. Reference starts pressing immediately after reset;
it is not the earlier rest/press/hold/retract measurement protocol.
Provisional weights: progress 10, speed/load rates 1/s each, normalized jerk
rate 0.1/s, completion +100, failure -100, gamma .99, 100-step/5 s deadline.

| Reference | Ending | Steps | Discounted return | Peak net contact force (2 ms) |
|---|---|---:|---:|---:|
| Press at 3 cm/s | success | 43 | 72.77 | 3.37 N |
| Press + saw (2 cm, 1 Hz) | success | 45 | 71.44 | 1.88 N |
| Fast press (9 cm/s) | success | 17 | 93.33 | 5.88 N |
| Stationary above object | terminal timeout, incomplete | 100 | -2.41 | 0 N |

**Reward preference FAILED.** The earlier discounted completion bonus outweighs
the speed penalty. It is an objective/weight issue, not evidence of a learned
policy exploiting a WM: no learner participated. These weights must not be used
for a new long run. No coefficient, force limit or physical parameter was adjusted
to hide this negative result. After these four completed cases the checker was
stopped; retract and slow-press references were not run. Intended physical force
violation and high-load slow pressing are also not covered by this batch.
Failure precedence/once-only penalties are covered by task tests, and final
coupling-sample force termination by existing GPU regression tests.

Artifacts (GPU7, not Git or Mac): runs/free_cut_v1_review_20261112/report.json,
runs/free_cut_v1_review_20261112.log, runs/free_cut_root_final.log,
runs/free_cut_backend_suite.log; final focused tests: 11 passed (35.37 s). The report includes exact per-step reward and
progress, parent revision, dirty-diff fingerprint, and full provisional config.
It predates the checker's newly added speed/jerk/utilization row fields, which
enable offline reward rescoring in future runs. No new policy checkpoint exists.


## 2026-11-12 — Free-cut review repair and bounded physical checks

Full implementation/evidence handoff: [FREE_CUT_REPAIR](archive/FREE_CUT_REPAIR_20261113.md).
Branch codex/free-cut-v1, code 1b287d5 / 5362975. No long training.

- Reset: 256 worlds / one reset, 25.66 -> 4.43 seconds; all 256 reset, 25.47 -> 16.03.
  Material draws and settling duration unchanged. Inactive rows preserved exactly; evolved
  rows are not bitwise-repeatable (including eager/eager repeats). See the handoff's numerical
  comparison and its explicit limits.
- Gamma=1 candidate: normal press 109.819, sawing 109.806, 3x press 109.285, 20x reference
  109.735; stall -3.808, retract -1.778, slow incomplete .449. The 20x probe did not exceed
  the force limit and is not counted as a force-failure test.
- Corrected profiles, paired with a new press reference: press 109.811; mid-cut slowdown
  107.561 (94 steps, separated); mid-cut hold .029 (100 steps, incomplete). Observed ordering
  matches the intended preference. Fixed reference motions are not six-action WM training data.
- 32 actual six-action transitions followed by one WM/actor/real-critic update all produced
  finite values. This is an execution check, not learned performance. Physical-label diagnostics
  exposed the untrained completion head's false completion probability (~.513); one update
  is not used to judge its learnability.
- Root 303 passed / 5 skipped; subsequent affected tests 29 passed; backend 37 passed.
  The new graph comparison checks a full 32-substep replay. A proposed 4 ms pointwise velocity
  tolerance failed for repeated eager runs too; no existing backend test was weakened.

Current limits: no trained v2 WM/policy; completion calibration after pretraining is unverified;
no actual physical force-failure / near-limit slow-press case in this bounded batch; the
100-step horizon and EE box are review-only configuration, not a general multi-object setup.


## 2026-11-12 — Free-cut v3 second-review repair

This is an engineering reward/termination/reset check, not a trained-policy comparison.
The v2 ordering claim is superseded: its 20x-speed penalty margin was only about 0.08.
V3 uses progress-weighted speed error plus a finite time-budget cost; formulas, cost bounds,
release-consistent completion hazard and provenance are in TASK_FREE_CUT / FREE_CUT_REPAIR.
Simulation limits remain [50,50,50,5,5,5], free cucumber, nominal material, .75 s settling,
friction transition .01, H=100, gamma=1. No physical parameter was tuned to make the task hard.

### Actual reference rerun

| Trajectory | Ending | Steps | v3 return | 2 ms net-force peak |
|---|---|---:|---:|---:|
| press | success | 43 | 109.821 | 4.146 N |
| saw | success | 46 | 109.720 | 1.880 N |
| fast | success | 17 | 105.210 | 6.172 N |
| stall | timeout | 100 | -19.999 | 0.000 N |
| retract | timeout | 100 | -14.555 | 1.122 N |
| slow | timeout | 100 | -6.594 | 3.769 N |
| extreme_fast | success | 5 | 94.300 | 9.677 N |
| slow_press | success | 94 | 99.625 | 3.807 N |
| hold_in_material | timeout | 100 | -12.566 | 2.576 N |

Normal press exceeds 20x press by 15.52 return, 3x press by 4.61 and the completed slow-press
case by 10.20. This repairs the measured reward preference; it does not prove the absence
of every reward exploit or that a learned critic will be accurate. Sawing is allowed, not
rewarded merely for being sawing. Every row was recomputed by the task during physical
execution, not only rescored afterward. The run finished all nine cases, with no natural
force-limit event in this nominal condition. The force-abort wiring probe is separately
labelled and must not be pooled with this table.

Different reruns need not have identical force peaks or release times (e.g. sawing completed
at 46 steps here versus 43 in the old record). Existing long-duration floating repeatability
limits remain; no numerical rule was relaxed in this review. These are case observations,
not a statistical force comparison or evidence of real-world calibration.

Raw evidence on GPU7: runs/free_cut_second_review_20261112/. The reference report contains
configuration/hash, source diff/hash, script snapshot and per-step rows. Intermediate timing
harness attempts are retained; only throughput_final is the final implementation profile.


### Isolated physical abort probe

To test the actual 500 Hz sensor-event path, a separate TEST-ONLY config used limits
[.1,.1,.1,.01,.01,.01]; the task/reference config remains [50,50,50,5,5,5]. Normal pressing
terminated at step 18 with collision=True, success=False, outside=False, timeout=False,
rho=0. The first sensor torque component reached .0102196 Nm against .01 Nm (utilisation
1.02196); net material force peak was .116932 N. Episode return -100.26715, last-step reward
-100.05671. The one failure settlement was charged and the episode ended. This is a physical
wiring/termination test, not evidence of difficulty or risk under the nominal task limits.
Raw report: runs/free_cut_second_review_20261112/force_abort/report.json.


### Final-code throughput and memory

GPU7 RTX 4090, one free-cucumber group, 256 worlds, full-width networks, source replay
capacity 1000 rows/world. The reset stress deliberately starts worlds with staggered
remaining deadlines in a 16-step test horizon; it is not a task-performance experiment.
32 real-control iterations = 8192 physical transitions, 16 timeouts/reset rows each step,
512 ended episodes in total. Materials are redrawn before each .75 s settle. Actual WM,
actor and real-critic updates ran, with finite reported losses/gradients. All episode endings
were deadline endings; no learned cutting-success claim or completion calibration is made.

| Timing | Measured |
|---|---:|
| Physical vector step, mean | 3.025 s |
| Reset of 16 selected rows, mean | 5.473 s |
| Small-review-update loop, mean total | 8.724 s |
| Full-duration one-world reset after cache stress | 4.700 s |
| One source-sized WM update: 10 x 2048, window 8 | 103.687 s |
| One actor/real-critic update: actor 4096, h8; critic 5 x 4 | 26.134 s |

The first loop uses batch 32, one WM minibatch, one critic epoch/minibatch, updating every
four physical steps after its data warmup. The final source-sized update uses microbatch
64 and eager execution. They are intentionally reported separately. Neither the 8–9 second
small-update loop nor these eager timings are a validated compiled multi-object long-run
budget. At source cadence 0.4 updates/vector step, the full update costs must be amortized
in addition to physical/reset time; do not omit them.

At the full-update measurement: PyTorch peak allocated 1961.7 MiB,
reserved 2284.0 MiB; observed whole-device used 2891.8 MiB.
The cache stress visited all capacities 1,2,4,...,256 and never retained more than
256 scratch-world capacity. Inactive material rows stayed bitwise unchanged.
This measures one-object eager allocation. Three object workers, different meshes and
compiled graph pools can cost more and are not certified to fit by this result.

Evidence: runs/free_cut_second_review_20261112/throughput_final/report.json, per-iteration
metrics and source diff. Whole final profile took 438.2 s including initialization,
cache pressure and the full-size updates. It produced no trained-policy checkpoint.


Final regression on the repaired code: root **310 passed, 7 skipped**; backend **38 passed**;
MPPI environment **12 passed**. Root's two MPPI dependency skips were covered by the dedicated
run; five historical SAC/Gym/TensorFlow checks remain explicitly skipped. Logs and commands
are in runs/free_cut_second_review_20261112/regression_final.json and *_final.log.
Code: 0ef9718, c720cd9, abe84b2. The latter removes demonstrated float32 cancellation in
release-fraction gradients and logs emitted probabilities directly. No long training,
S3 restart, merge to main or new learned policy was produced by this review.

## 2026-11-13 — free_cut_v3 short training-path validation (free cucumber, single object)

Engineering check of the production training path; not a learning result. Config
`configs/free_cut/short_train_cucumber.json` (256 worlds, S0 material randomization, original
simulation limits, 180-step horizon, v3 reward unchanged). `train_gpu.py` with compile, critic `real`,
actor horizon 8, BC 0.25, 500 WM-only updates (unchanged). Run 1: 3.6 h cap
(`runs/free_cut_short_20261112/training`), then resume from its `last.pt` with replay for 2.3 h
(`.../training_resume`). GPU7 RTX 4090, no other job.

**Chain.** Collection, per-material re-settling resets, terminal settlement, WM pretraining, transition
to joint training, 43 actor/critic updates, periodic checkpoints (250 ... 1250, `wm_only_complete.pt`,
`first_joint_update.pt`) and resume with replay all ran. 543 update records, no non-finite value, no
traceback. 2425 episodes: success 1026, timeout 1399, force limit 0, workspace 0; every success has
complete separation and no separated episode is recorded otherwise. Peak limit utilisation 0.74;
the original limits were never reached (data coverage, not adjusted).

**Time and memory.**

| Stage | s per vector step (256 worlds) | Share in run 1 |
|---|---:|---|
| WM-only (iterations 40-1330) | 14.5 | physics 21 %, reset 16 %, WM updates 63 % |
| Joint (from about iteration 1330) | about 19.4 | adds actor/critic updates |

WM update about 24 s compiled (104 s eager in the earlier profile). GPU memory 2.8 GB in the WM-only
stage, about 7.8 GB once actor/critic were compiled. Estimate for one object, 5000 iterations: about
5.2 h to the first joint update plus about 20 h of joint training (about 1500 policy updates), i.e.
about 25 h. Three objects (three worker processes on one GPU, potato at 8e-6 s) are not measured and
must not be extrapolated linearly.

**WM progress and completion.** Fraction of sampled WM transitions that are separations: about 0.2-0.3 %.
Missed completions among them (head probability < 0.5): 100 % at WM update 61, about 80 % at the
end of WM-only training, 73 % at update 543. False completion among incomplete transitions: 0.07-0.09 %
at the end. In imagination after 43 policy updates the mean per-step completion probability was 0.11
with predicted progress 0.71 and 8-step survival 0.55, far above the physical separation rate per
transition; this indicates that the policy reaches WM states where the completion head is more
optimistic than on recorded data. Not judged from 43 updates; to be measured in the long run
(imagined vs physical completion on the same start states).

**Observations for the code owner (none blocks training).** A resume restarts the 40-iteration
update warmup (counted from the resume iteration). A WM metric ratio is logged as 1.0000026 (float
accumulation in weighted minibatch means). The W&B offline exporter logged transient "Bad Gateway"
retries. Timeouts end at a median progress of 0.85 under the initial/early policy.

#### 2026-11-13 — Deterministic evaluation of the two short-run checkpoints; corrections

Evaluation (`scripts/evaluate_multi.py`, `configs/free_cut/eval_cucumber.json`: training distribution,
seed 100, 64 worlds x 2 episodes; reports `runs/free_cut_short_20261112/eval_{first_joint,last}`):

| | `first_joint_update.pt` (no policy update) | `last.pt` (43 policy updates) |
|---|---|---|
| Separated (task success) | 24/128 (19 %) | 88/128 (69 %) |
| Timeout | 104 (81 %) | 0 |
| Workspace | 0 | 40 (31 %), every one at step 52 |
| Force limit | 0 | 0 |
| Median episode length | 180 steps | 7 steps |
| Median speed error | 20 mm/s | 131 mm/s |
| Peak limit utilisation, median / max | 0.32 / 0.52 | 0.44 / 0.77 |
| Mean undiscounted return | 9.8 | 36.6 |

For reference, the fixed press at the commanded speed returned about 110 in the physical reward check.
After 43 updates the policy separates the object in about 6 steps (far above the commanded 1-5 cm/s);
the workspace endings had released about 95 % of the springs. Inference, not verified: the sawing
output keeps one sign, and the mapper integrates it into the target offset (about 2 mm per step), so
the target leaves the 10 cm half-width of the workspace after about 50 steps. 43 updates do not
judge the method; the cause of the behaviour change is not established (WM completion/progress
prediction, critic and the integrated lateral offset are all candidates). Next: same-trajectory
prediction records (WM given the executed actions from the same start states) before any
attribution.

Corrections to the short-run entry above: (1) GPU7 was shared with a verification job between about
00:55 and 01:12 Berlin time on 2026-11-13, during the resumed run, so its timings are not exclusive-GPU
timings. (2) The comparison "11 % imagined per-step completion probability vs 0.3 % separated
transitions in replay" mixes different states, actions and statistics and is withdrawn as evidence
of model exploitation. In v3 the completion probability is the predicted release fraction times the
head probability, so "missed completions" can come from the progress prediction, the head or both.

#### 2026-11-13 — Same-trajectory prediction check of the two short-run checkpoints

`scripts/diagnose_free_cut_predictions.py`: deterministic policy, one episode per world on
`eval_cucumber.json` (the same 64 conditions for both checkpoints), every step recorded; then the
checkpoint's own WM, critic and policy are evaluated on those recorded states and executed actions.
Reports: `runs/free_cut_short_20261112/same_trajectory_{first_joint_update,last}/report.json` (+ steps.npz).

| | first_joint_update (1 update) | last (43 policy updates) |
|---|---|---|
| Episodes: success / workspace / timeout | 13 / 0 / 51 | 43 / 21 / 0 |
| 1-step WM error: progress / force component / EE position | 0.006 / 0.15 N / 0.12 mm | 0.09 / 1.15 N / 2.5 mm |
| Completion probability on separating steps: total (release fraction x head) | 0.04 (0.26 x 0.14) | 0.59 (0.61 x 0.97) |
| Completion probability on non-separating steps; share >= 0.5 | 0.0002; 0 % | 0.71; 74 % |
| 8-step cumulative completion when not separated within 8 steps | 0.001 | 0.999 |
| WM predicts the workspace exit in the last 8 steps before a real exit | (no exits) | 8.9 % (9.4 % when there is no exit) |
| Critic minus realised return-to-go, workspace episodes | - | +294 |
| Imagined actor objective vs realised return-to-go, workspace episodes | - | 97.7 vs -102.9 |
| Same, successful episodes | 4.3 vs 95.9 | 98.6 vs 100.6 |

- All 21 workspace endings: x axis, step 52, the sawing output had one sign at every step (mean
  |a4| before clipping 1.42) and the integrated target offset reached -102 mm, i.e. the lateral
  workspace half-width. The cut had stalled near 95 % release while the knife slid along the blade.
- On these trajectories the WM gives a high completion probability where the cut has not separated
  (mostly the stalled workspace episodes), does not predict the lateral exit, and the critic values
  them far above any achievable return (+294 above the realised return, above the 110 maximum).
  Inside the model these episodes therefore score like successful cuts (97.7) while they end at -103.
- With the untrained policy the same WM was accurate on its own trajectories (progress 0.006,
  0.12 mm) and conservative about completion. The large errors appear on the state-action region the
  updated policy moved to (saturated lateral motion, stalled cut), which the replay data from earlier
  policies barely cover.
- Interpretation (not yet a conclusion): this matches the pattern "model score improves, physical
  result gets worse, the error concentrates on the actions the policy now prefers". Whether more
  data from the new policy corrects it (normal model-based RL dynamics) or the policy keeps exploiting
  it is not determined by 43 updates.

## 2026-11-13 — Budgeted joint training on the free cucumber with same-trajectory checks

Resumed from the short run's `last.pt` (43 policy updates) on Codex's infra code (`b1bcb92`): same task,
reward, networks and action interface; fixed budget of 1250 more iterations (to 2691), 9 h cap, no
extension. Completed in 7.5 h (GPU shared with the diagnostics): 543 policy updates. Run:
`runs/free_cut_joint_20261113/training`; every checkpoint evaluated with
`diagnose_free_cut_predictions.py` (`12b951f`) on the same 64 development conditions
(`eval_cucumber.json`, deterministic policy, one episode each): `runs/free_cut_joint_20261113/eval/`.
Episode-level, paired across nodes; a development set, not a final test set.

| Policy updates | Success / workspace / timeout / force | Mean return | Median speed error (mm/s) | p(complete) on separating / non-separating steps | Critic - return-to-go (all) | Imagined objective / return-to-go (all) |
|---:|---|---:|---:|---|---:|---|
| 43 | 44 / 20 / 0 / 0 | 36.5 | 123 | 0.63 / 0.70 | n/a | n/a |
| 66 | 49 / 15 / 0 / 0 | 54.3 | 50 | 0.18 / 0.013 | +59 | 46.7 / -6.6 |
| 166 | 41 / 0 / 23 / 0 | 58.6 | 79 | 0.32 / 0.014 | +13 | 10.8 / -1.4 |
| 266 | 46 / 1 / 17 / 0 | 64.4 | 87 | 0.48 / 0.008 | +1 | 11.6 / 11.2 |
| 366 | 46 / 0 / 17 / 1 | 64.4 | 70 | 0.50 / 0.014 | +9 | 19.7 / 11.6 |
| 466 | 52 / 0 / 12 / 0 | 79.7 | 79 | 0.34 / 0.014 | +3 | 27.8 / 20.9 |
| 543 | 52 / 0 / 12 / 0 | 80.3 | 78 | 0.43 / 0.010 | +6 | 22.7 / 17.9 |

The 43-update row is the corrected-script rerun; simulator nondeterminism changes single-node counts by
about one episode, so only clear changes are read.

- Physical performance improved over the budget (success 69 % -> 81 %, mean return 36.5 -> 80.3) with
  a plateau between 266 and 366 updates. One force-limit ending occurred (366 updates); otherwise none.
- The false completions and the missed lateral exits of the 43-update checkpoint were corrected within
  the first 23 updates of new data (non-separating completion probability 0.70 -> 0.013; exit
  prediction before real exits 10 % -> 95 % with 0.1 % false alarms). Overall critic bias and the gap
  between imagined objective and realised return-to-go stayed small afterwards (within about 10).
  The pattern "model score rises while physical results worsen" was not observed in this budget.
- Remaining: completion recall on separating steps stays at 0.3-0.5; the critic underestimates the
  return of successful episodes by 30-50; successful cuts take about 10 steps, i.e. 2-3x the commanded
  speed; the timeouts stall at 91-98 % release; the mean sawing magnitude rose again late (median
  |a4| 0.65-0.84 at the last two nodes).
- Training episodes in this run: 6205 (success 4768, workspace 339, timeout 1098, force 0), with
  exploration noise. Time shares: WM 44 %, reset 24 %, physics 16 %, policy 15 %; 21.7 s per
  iteration with the diagnostics sharing the GPU. Resets grew to a quarter of the time because
  episodes became short.

#### 2026-11-13 — Same-condition reference comparison and reward decomposition (543-update policy)

`scripts/compare_free_cut_references.py`; report `runs/free_cut_joint_20261113/reference_comparison/report.json`.
Same 64 development conditions as the policy diagnostic (0 condition mismatches), same task settlement,
180-step deadline, limits and reward. References: press straight down at each case's commanded speed
over the scene-reference travel plus that case's start lift, then hold; and the same press with a
+-2 cm, 1 Hz lateral sinusoid. The references command knife motion directly and bypass the six-action
compliance mapping, so they are physical control references; differences are not all attributable
to the learning algorithm. The reward reconstruction from recorded transitions matches the logged
reward to below 1e-6.

| Controller | Success / timeout | Mean return | Median speed error | Median steps | Peak utilisation median / max |
|---|---|---:|---:|---:|---|
| Policy (543 updates) | 52 / 12 | 80.3 | 76 mm/s | 11 | 0.41 / 0.96 |
| Press at commanded speed | 45 / 19 | 75.1 | 5 mm/s | 66 | 0.42 / 0.78 |
| Press + saw | 38 / 26 | 61.9 | 7 mm/s | 107 | 0.40 / 0.78 |

Mean return terms (progress / speed cost per released fraction / speed cost per time / load /
terminal): policy 9.8 / -6.0 / -4.0 / -0.7 / 81.3; press 9.7 / -0.4 / -3.6 / -0.9 / 70.3;
press + saw 9.6 / -0.4 / -5.7 / -1.0 / 59.4. On the 27 cases all three complete: median return 104.2
(policy) vs 109.6 and 109.7; 10 vs 55 and 54 steps.

Case pairing: of the 12 policy failures (all timeouts; final progress 0.71-0.99), a reference completes
5 (press 3, saw 4); in 7 all three time out (mostly cut-spring stiffness above x1 or commanded speed
below 2 cm/s). The policy completes 10 cases the press does not.

Frozen 543-update critic: on the policy's own trajectories, value minus realised return-to-go has mean
+6.3 and mean absolute value 26.6. On the reference trajectories the gap between reference return and
this critic's score is -3.5 (abs 35.3) and -2.7 (abs 37.3); the critic estimates the current policy's
return, and the reference states may lie outside the training data, so these are recorded gaps, not
critic errors.

Reading: under this reward and deadline, cutting at 2-3x the commanded speed costs about 6 points more
speed penalty than the press but avoids timeouts that cost the full completion reward; the policy's
speed choice is consistent with the reward as specified, not by itself evidence of a model error.
Most remaining policy failures are cases that the simple references do not complete either.

## 2026-11-13 — Free prism and free potato: platform check (same protocol as the free cucumber)

Scenes: `configs/free_scenes/{prism,potato_003}.json` (no static vertices; otherwise the existing
scene settings; potato at its existing principal-axis orientation: shortest axis up) exported to
`backends/mujoco/runs/sweep/free_{prism,potato_003}`. Check: `check_free_object_scene.py`, per-world
material settle 0.75 s, 0.25 s rest, reference press at 3 cm/s to the scene goal, hold, retract; force
limit disabled; 2 ms force sampling; potato at 8e-6 s. Reports: `runs/free_object_check_20261113/`.

- Topology: both meshes have exactly two components joined only by their cut springs (prism 304,
  potato 245), so the separation check applies.
- Prism: rest motion 0.001 mm. No world separated: 282-288 of 304 springs released at the end (press
  and press + saw, Young/cut-spring x1/x1, x2/x1, x2/x2); 242-264 when the press ends. Peak net force
  4-12 N. The halves moved 0.4 mm apart; with sawing the object slid 23-31 mm along the blade. The
  fixed-bottom prism released 298-304 under the same reference before (2026-11-12).
- Potato (principal-axis pose): not at rest after settling (particle speeds 60-104 mm/s and about
  1 mm of motion during the 0.25 s rest). Under the press it was pushed 7-20 mm across the cut plane
  (x) and released only 4-120 of 245 springs; with sawing one world was lifted 25 mm. No world separated.

Not decided here: the potato's resting pose is a geometry choice of asset preparation (a hull-based
stable pose is the realistic placement); why the free prism keeps 16-22 springs intact is not
established. The free cucumber (2026-11-12/13) separates under the same references.

#### 2026-11-13 — Knife-edge tracking, travel and potato rest (three free objects)

`check_free_object_scene.py` now records the knife-edge height (knife body y plus its lowest vertex,
DiSECt frame) and the commanded edge height, relative to the settled object's lowest particle.
Press at 3 cm/s from the scene start (no extra lift), Young/cut-spring x1/x1, x2/x1, x2/x2, force limit
disabled. Reports: `runs/free_object_check_20261113/knife_lag/`. Object heights as placed: cucumber 39.4,
prism 50.0, potato 53.6 mm.

| Object (settle) | Rest: centroid motion in 0.25 s / max particle speed | Separated | Lowest edge vs object bottom | Mean / max tracking lag in the press | Edge within 0.5 mm of the commanded end |
|---|---|---|---|---|---|
| cucumber (0.75 s) | <= 0.024 mm / 3 mm/s | 2 of 3 (x2/x1: 62/65) | -3.5, -1.3, -3.4 mm | 4.3-4.6 / 4.7-7.0 mm | 0.40-0.50 s after the commanded end; not reached for x2/x1 |
| prism (0.75 s) | 0.001 mm / 0.4 mm/s | 0 of 3 (282-287/304) | -3.6, -3.6, -3.3 mm | 4.5-5.4 / 5.0-6.4 mm | 0.50 s; not reached for two worlds |
| potato (0.75 s) | 0.6-1.1 mm / 74-102 mm/s | 0 of 3 (8-121/245) | -3.9 to -4.0 mm | 4.3-4.7 / 5.1-6.7 mm | 0.40-0.50 s |
| potato (3.0 s) | 0.26-0.41 mm / 39-65 mm/s | 0 of 3 (55-121/245) | -3.2 to -3.6 mm | 4.3-4.7 / 5.3-6.4 mm | 0.40-0.50 s |

- Placement and simulation (reported separately from cutting): cucumber and prism are at rest and
  numerically sound; the potato in its principal-axis pose is still moving after 3 s of settling.
- Historical travel inference (superseded by the geometry repair below): these heights used
  translation plus the lowest local vertex, omitting rotation and comparing against the initial
  settled bottom. They alone do not establish coverage of the remaining connections.
- Lag (from knife motion, not from cut time): during a 3 cm/s press the edge trails the command by
  about 4.5 mm (about 0.15 s) and reaches the commanded end 0.4-0.5 s after it; on stiffer material it
  can stay short of it. Not yet measured at 1 cm/s, the speed that sets the time budget.


#### 2026-11-13 — Corrected knife geometry and an actual free-prism rerun

User-authorized measurement repair, not a physics/task change. Based on 1ed9b54;
script: `check_free_object_scene.py`, measurement revision `knife_pose_geometry_v2`.
The knife vertices are now transformed by the actual xyzw rotation **and** translation.
The edge is this asset's two lowest local vertices. Heights relative to the initial
settled bottom, the instantaneous bottom and the DiSECt world origin are separate.
The initial/current particle minima are not labelled as the board height.

Arrival now means entry into the existing symmetric +/-0.5 mm height band during
the post-command hold, excluding retraction; the old one-sided check also accepted
arbitrarily deep overshoot. This is a measurement tolerance, not a task threshold.
The commanded edge is the translated initial edge with its initial orientation;
it is a reference proxy, not a directly imposed DiSECt pose.

One actual three-world run took 238 s: free prism, E/spring x1/x1, x2/x1, x2/x2,
0.75 s per-material settle, original 3 cm/s press/hold/retract, force-limit termination
disabled. Neither material, controller, trajectory, force limits nor separation
criteria were changed. Output: `runs/knife_geometry_fix_20261113/prism_press/`;
raw geometry: `geometry.npz`; offline spatial analysis:
`runs/knife_geometry_fix_20261113/analyse_geometry.py` and
`prism_press/geometry_analysis.json`. All remain on GPU7.

| E / spring | Released / 304 | Lowest edge vs initial / current bottom (mm) | Maximum omitted-rotation error (mm) | Mean press height lag (mm) | Peak net force (N) |
|---|---|---|---|---|---|
| 1 / 1 | 285 | -3.745 / -3.924 | 0.163 | 4.528 | 4.199 |
| 2 / 1 | 288 | -3.781 / -3.846 | 0.194 | 4.589 | 4.656 |
| 2 / 2 | 282 | -3.858 / -3.932 | 0.588 | 5.164 | 12.106 |

Height extrema above are from 50 ms control-step endpoints. The raw geometry
samples knife pose, both material spring endpoints, stiffness and instantaneous
object bottom synchronously every 2 ms; endpoint interpolation follows
`eval_cut_springs`, not the knife contact-search coordinates.
At 3 cm/s, the measured arrival delays were 0.45, 0.50 and 0.50 s.
These are not estimates for 1 cm/s.

Spatial observations from the 2 ms record:
- All final residual connections were inside the blade's longitudinal span at its
  deepest pressing/holding position. At that instant, their endpoints were
  3.05-12.10 mm above the blade edge at the same longitudinal coordinate.
- For each residual spring, the minimum distance of either endpoint to the edge
  segment during press/hold was 0.031-0.068 mm.
- Residual endpoints were not universally only in the bottommost layer: at the
  deepest instant the first two worlds spanned about 0.08-8.39 mm above the current
  bottom; the x2/x2 world spanned 0.09-0.28 mm.
- All states were finite, but no world fully separated. This supports no obvious
  gross vertical-stroke or blade-length shortfall in this run. Proximity alone is
  not evidence of contact at every material substep, sufficient damage, a physics
  bug, or task impossibility. No residual springs were removed or counted as cut.

The script also records mean particle velocity and RMS speed alongside maximum
particle speed, so a single fast particle is not reported as whole-object motion.
Potato orientation, the 1 cm/s lag and multi-object training were not changed or
run in this repair.

Validation on GPU7: root suite with CUDA assets and the MPPI environment:
**324 passed, 5 skipped**; backend suite: **38 passed**. Regression tests include
a 90-degree rotated knife, asymmetric material-edge interpolation in batched
worlds, and overshoot/retraction exclusion from the arrival statistic. Independent
NumPy reconstruction of the saved geometry matched the reported step heights
within 7.5e-6 mm; final residual counts matched exactly.
Logs and source patch/hash: `runs/knife_geometry_fix_20261113/`.


#### 2026-11-13 — Why free-prism connections remain after the blade descends

User-authorized bounded debugging, based on c0d5a76. No physics, reward, material
or separation-rule change. An opt-in observer reads the exact inputs to every
`eval_cut_springs` launch and its stiffness output, before particle integration.
It is installed only in this diagnostic process and restored on exit.

Run: same free prism and 3 cm/s press/hold/retract, E/spring x1/x1, x2/x1, x2/x2,
force-limit termination disabled; 478 s wall time. Runtime material dt was
**1e-5 s** (use runtime dt, not the export JSON's nominal dt). The trace covered
435,000 material launches, each processing all three worlds. Outputs:
`runs/residual_damage_debug_20261113/prism_press_trace/`;
analysis: `runs/residual_damage_debug_20261113/summarize_trace.py`.
A first instrumentation attempt stopped before the motion loop because Tape was
imported from the wrong module; the dispatcher import was corrected and tested.

The implemented rule is, per material substep:

`ke_next = max(0, ke - nonzero(|knife_force_left| + |knife_force_right|)
                          * max(endpoint_distance - rest_length, 0) * softness)`.

Passing a geometric plane does not itself remove a material connection. The
observer recomputed this formula independently in PyTorch. Maximum per-substep
stiffness discrepancy was 0.00024414, consistent with float32 rounding at initial
stiffness 1800-4000; cumulative discrepancies were at most 0.00293 per spring.
Actual accumulated decrements also matched initial minus final stiffness.

| E / spring | Released / 304 | Residual count | Residual stiffness range | Residual contact duration range (s) |
|---|---|---|---|---|
| 1 / 1 | 284 | 20 | 0.369-301.965 | 0.107-0.203 |
| 2 / 1 | 287 | 17 | 2.873-605.772 | 0.086-0.201 |
| 2 / 2 | 282 | 22 | 38.552-643.559 | 0.222-0.409 |

Every residual spring had both knife contact and positive damage; none was simply
skipped by the update. Their stiffness had decreased by about 70-99.98%, but not
to zero. Most bottom-connection damage happened during the hold while the robot
caught up with the reference; a few continued damaging briefly during retraction.
Last damage times ranged from 2.230 to 2.926 s; after retraction no residual spring
had knife contact, so damage ceased. Some retain substantial stiffness: this is
not solely a tiny-positive-number issue. Compared with the uninstrumented rerun,
the first two worlds differ by one released spring; this is not a bitwise
deterministic replication claim.

Measured edge minima were 3.85-3.93 mm below the current particle minimum. Together
with the earlier geometry trace, no gross vertical stroke shortfall was found.
The observed mechanism is incomplete damage accumulation over this reference
contact history, not an observed missed stiffness update. This does not certify
the physical fidelity of the damage/contact law or prove the task impossible.

Separate source limitation, not a proposed repair: the active analytic knife SDF
uses its x-y cross-section; `vec3.h::sdf_knife` does not use the depth parameter,
and `sim.py::knife_sdf` leaves longitudinal clipping commented out. Thus the
finite mesh length is not a longitudinal SDF collision boundary. Mesh and SDF
tip heights for this asset agree (both -29.54 mm locally). Residual connections in
this run lie within the displayed blade length, so the source observation alone
does not explain their survival. No contact-model redesign was made.


Follow-up chosen *after* the trace: one reference at the already planned minimum
speed, 1 cm/s. Same scene/material/settle/hold settings and separation criterion;
no damage observer in this follow-up. The existing `--down-speed` option also
sets retraction speed, so the full sequence changes from 4.35 to 10.55 simulated
seconds. First-separation checks include all phases and remained false.
Run took 560 s; output: `runs/residual_damage_debug_20261113/prism_press_1cms/`.

| E / spring | Released / 304 | Mean press height lag (mm) | Arrival delay after commanded end (s) | Lowest edge vs current bottom (mm) | Peak net force (N) |
|---|---|---|---|---|---|
| 1 / 1 | 283 | 1.639 | 0.25 | -4.261 | 2.486 |
| 2 / 1 | 281 | 1.682 | 0.20 | -4.284 | 2.896 |
| 2 / 2 | 281 | 1.768 | 0.25 | -4.459 | 5.679 |

Tracking improved and force decreased, but slowing this reference did not fully
separate the prism. No simple speed remedy was demonstrated. No missed stiffness
updates were found in the traced 3 cm/s run; incomplete accumulation under that
contact history is established, while physical fidelity and reachability under
other controls remain separate questions. This bounded diagnostic ends here:
no automatic speed/friction/softness sweep and no weakening of the success rule.
The prism need not block preparation of the other freely placed objects.

Validation on GPU7: root suite with CUDA assets/MPPI environment **328 passed,
5 skipped**; backend **38 passed**. Tests cover contact/extension gating,
non-mutating inputs, observer restoration on exception, and an actual DiSECt
kernel with observation enabled/disabled producing identical stiffness and force
outputs for the same synthetic inputs. Tests and run logs are in
`runs/residual_damage_debug_20261113/`.

## 2026-11-13 — Board-stop v4 and paired frozen-policy audit

User-authorized task change: effective finite knife-path coverage plus board arrival, no force/workspace failure or knife penetration. Complete material separation is retained as a separate outcome. No training or material/fracture modifications. Full implementation, approximation and reproduction command: [BOARD_CUT_REVIEW](archive/BOARD_CUT_REVIEW.md).

GPU7, latest `free_cut_joint_20261113/training/last.pt` (543 policy updates), same 64 seed-100 development conditions from `eval_cucumber.json`; all material/start/target-speed fields match the historical evaluation and the paired run. Frozen policy weights, force limits and 180-step horizon are unchanged. This is not a learning improvement experiment, and rewards are not compared across the different task definitions.

| Measurement | Legacy v3 replay | Board-stop v4 transfer |
|---|---:|---:|
| Task successes, different definitions | 53/64 spring separation | 38/64 geometric path + board arrival |
| Episodes with knife below the board | 52/64 | 0/64 |
| Legacy successes involving board penetration | 41/53 | Not applicable |
| Complete material separation at episode end | 53/64 | 19/64 |
| Incomplete endings | 11 timeouts | 26 board arrival without complete coverage |
| Minimum actual knife clearance | -9.119 mm | +0.000825 mm |

All 11 timeouts in this legacy replay remained incomplete under v4; no replay timeout became a path success. Fifteen legacy successes became incomplete board arrivals. Among the 38 v4 path successes, 19 still retained material connections (release fraction 0.9385–0.9846). Thus path coverage/board arrival and physical separation are empirically different, not interchangeable labels. The old missing board constraint admitted invalid successful trajectories; these results do not support blaming the lower old success rate primarily on an overly strict separation label.

Historical evaluation was 52/64; the new legacy replay is 53/64, with three case outcomes different. Conditions are identical, but the physical trajectories are not bitwise repeatable. Against historical labels one old timeout becomes v4 path success; this does not override the paired replay's zero conversions or establish a criterion-only causal effect. The stop changes dynamics, and v4 keeps going after a v3 separation ending until board arrival.

All 26 incomplete v4 cases have coverage 62/65. Each also has three geometric interface midpoints (indices 32,33,35) more than 0.5 mm below the board at termination; the lowest midpoint is -1.036 to -1.327 mm. At the first recorded control frame the minimum was only -0.073 to +0.026 mm. This is direct evidence of material/ground overlap during pressing, consistent with incomplete geometric coverage at a nonpenetrating knife stop. It does not establish the spring damage mechanism. No coverage tolerance was relaxed, no bottom witness discarded, and no knife penetration allowed to turn these cases into successes.

The 64 v4 episodes ended in 9–11 control steps, maximum six-axis force-limit utilisation 0.543. This is still the old fast policy, not evidence of target-speed tracking. Finished-world knife pose, robot joints, material midpoint positions and release fraction remain exactly unchanged while the remaining worlds advance (checked from raw transitions). All recorded v4 knife minimum gaps are positive; geometry tests additionally exercise high-speed translation and rotation between sample endpoints.

The stop is an execution interlock: no invented board wrench, no claim to simulate compliant board contact. v4 WM construction explicitly rejects reuse of v3 completion/critic labels. This delivery is a corrected physical task plus frozen-policy evaluation, not a new trained v4 system.

Artifacts: `/dataHDD/dong/Experiments/Cutting_DEV/runs/board_cut_audit_20261113/`: `legacy64/`, `board64/` contain manifests, conditions, per-episode records and raw transitions; `comparison.json`, `integration_assertions.json`, `below_board_material_witnesses.json` retain the paired and geometry checks. Each manifest records exact source/checkpoint hashes. The conservative stop-contact refinement was made during the legacy replay; the legacy observer's behavior was unchanged, and the board evaluation used the final stop implementation. Regression results are recorded in BOARD_CUT_REVIEW.

Regression validation on GPU7 for this change: root **337 passed / 5 skipped**, backend **38 passed**. Runtime v4 source hashes match the final code.

## 2026-11-14 — Task simplification, code/doc cleanup and object readiness

The user rejected making the RL task depend on both a geometric coverage gate and complete separation. The selected direction is the existing SliceIt target-pose completion interface plus a nonpenetrating board execution boundary. Coverage and actual separation remain diagnostics. The v4 audit is retained as historical evidence, not promoted into a more complex learned task. No new reward weight, WM head, action dimension, training run or material parameter was added.

Code review found that the old-position coverage test used the current object's transverse footprint. The diagnostic now stores and uses the previous footprint separately. Stationary proximity no longer earns a downward-sweep credit. The diagnostic revision is swept_stop_path_v2. Invalid penetration beyond 2 micrometres aborts with FloatingPointError before a learning reward can be emitted; it is not a new RL failure penalty. The normal board stop and force/workspace penalties remain distinct. A briefly proposed penetration-penalty edit was withdrawn before commit after the user's numerical-error question.

Physical checks: GPU7, one frozen 543-update policy transfer per free scene, four worlds each, seed 100, existing material/start/speed sampling, 180-step deadline, five-minute wall budget per scene. Existing poses and material properties unchanged; potato substep 8e-6 s. The output names success/board_incomplete below retain **v4 diagnostic semantics**, not the newly selected target-pose task. They are not generalization estimates.

| Existing free scene | Initial centre-of-mass speed (mm/s) | v4 path completion / force ending / incomplete board arrival | Full separation | Minimum knife height (mm) |
|---|---:|---:|---:|---:|
| Cucumber | 0.071–1.222 | 2 / 0 / 2 | 2/4 | +0.000894 |
| Prism | 0.092–0.135 | 2 / 2 / 0 | 0/4 | +0.001108 |
| Potato, unchanged principal-axis pose | 16.857–88.409 | 4 / 0 / 0 | 0/4 | +0.000795 |

All 12 rollouts completed with finite data and no knife penetration. The reference 0.75 s per-material settling does not make the current potato stationary: COM speed confirms whole-object motion, not merely one fast deforming particle. Prism can supply development interaction data; the old reference's incomplete spring damage is not itself a reason to forbid collection. Potato needs a naturally stable resting preparation before using it as a stationary-start training condition. No pose was silently changed in this cleanup.

Minimum particle height during pressing was -1.800 mm (cucumber), -2.094 mm (prism), -1.273 mm (potato). The upstream particle-ground eval_contacts law is a soft penalty contact (penetration-dependent elastic term plus damping/friction), not a geometric hard clamp. Correction 2026-11-14: these all-node minima did not distinguish ghost FEM nodes from occupied contact geometry. They cannot establish actual surface overlap; retain them as raw diagnostics only. The knife tolerance is a separate quantity, and no ground stiffness was tuned to improve outcomes.

Saved under runs/board_review_20261114/: per-scene source/checkpoint manifests, initial_rest.json, episode rows, and raw transitions including coverage masks and minimum particle heights. Endings above do not certify model calibration or readiness of the simplified training interface.

Documentation cleanup: TASK_FREE_CUT now contains only the current direction and implementation status; v1-v3 equations, FREE_CUT_REVIEW and FREE_CUT_REPAIR are archived with links corrected. README, PLAN, METHOD, DATA, REAL2SIM, config/script indices and the current-results index distinguish historical from current status. The false current bottom-held prescription and no-trained-policy statement were removed. PAPER's description of SliceIt as SAC per material was corrected against the primary text: it randomizes the calibrated foods during training.

Tests are regression evidence, not success-rate certification. The five optional root-suite skips are: two Gym-dependent SAC bridge cases (Gym absent in this environment), source TensorFlow checkpoint, explicit SAC physical bridge check, and source tf2rl/Python 3.8 learner. Existing source/v1-v3 tests remain useful while those code paths and checkpoints are retained. No test was deleted to make the suite pass.

Long training is not launched: the simplified pose task still needs a board-aligned goal wired into the actual training entry with consistent task identity. No coverage/progress head expansion is required for that direction. This is a finite implementation task, not a requirement to explain all residual springs first.

Validation for the 2026-11-14 cleanup: root **339 passed / 5 skipped**, backend **38 passed**. The separate optional-SAC check printed all five skip reasons (optional_test_reasons.log). Physical checks above are separate evidence; a passing suite is not a long-training readiness claim.

Nominal goal audit: with the existing robot/material mapping and unchanged initial
orientation, the translation-only target blade gap is -4.390 mm for all three
assets; the actual initial/target EE orientation difference is 0.122 degrees.
This is a kinematic proxy, not another physical cut result. It shows why the
existing lower target must be explicitly aligned to the board before reusing
pose completion. The target was not silently shifted in the paired runs.
Data: nominal_targets.json. A first inspection script exited on an unsupported
GPUEnvironment.close() call; it was corrected, and the report above was produced
by the completed second inspection. Neither inspection ran a cutting trajectory.

## 2026-11-14 — Implemented pose completion and board boundary

Supersedes the earlier Nov-14 implementation-pending note. The user explicitly
requested changing the +100 event to prescribed target-pose arrival without
failure, retaining the remaining v3 reward terms and adding the board requirement.

Implementation: `free_cut_pose_v1`; source/v1-v4 identities remain unchanged.
The goal uses finite knife geometry and the tool offset, translated to the board.
Force/workspace failures override arrival and are penalized once. Timeout is a
terminal unsuccessful episode. The accumulated source arrival counter is retained.
No spring-release or geometric-coverage gate, and no ordinary board-contact penalty.
Release progress remains in running reward shaping and the WM, not in the success
predicate. The learned terminal head predicts task termination and cannot itself
award a probabilistic +100 bonus. Invalid physical penetration is still an error.

The WM board endpoint projection has an implicit derivative checked against finite
differences; detaching the bisection fraction would give the wrong BPTT gradient.
Projection runs eagerly, with the rest of the model on the existing compilation
path. Compiled forward/backward matched eager execution on a board-crossing case.

### Physical task settlement

Two nominal-material worlds per object, prescribed reference controller, active
simulation force/workspace limits, new pose task. These are engineering checks,
not learned-policy success rates:

| Object | Pose success | Final arrival count | Material released | Minimum knife/board gap |
|---|---:|---:|---|---:|
| Free cucumber | 2/2, step 36 | 11 | 64/65 and 65/65 | +0.823 micrometres |
| Free prism | 2/2, step 36 | 11 | 266/304 in both worlds | +0.885 micrometres |

The +100 settlement is distinct from total final-step reward (~99.889 after the
retained running costs). No crossing below the board occurred. The maximum
difference between the rigid WM geometry proxy and integrated physical knife gap
was 0.100 mm (cucumber), 0.066 mm (prism). These differences are reported, not hidden
by describing the two kinematic paths as exact equivalents.

### Actual optimizer and restart checks

Collected 24 real controller steps in two cucumber worlds (48 transitions).
A small WM (64x64, explicitly for engineering validation) performed an actual WM
update and BPTT actor/real-target critic update. Saved model, optimizer states and
replay; reloaded them, verified model outputs exactly, and executed another WM and
actor/critic update. Both update counters advanced correctly. No terminal samples
were in this small replay; physical terminal settlement was tested separately above.
Losses/gradients were finite; these checks do not establish model accuracy or
learning performance. Four-world cucumber/prism group construction and one real
step also passed with shared task metadata.

An earlier zero-action, full-episode engineering probe exhausted its 600-second
wall budget during concurrent GPU tests. It is not a completed performance
evaluation and is not included in success-rate claims. The final bounded update
check and complete prescribed reference episodes replace it for their respective
engineering purposes. Timing from these concurrent checks is not dedicated-GPU
training throughput.

Evidence: `runs/pose_task_20261114/{reference,reference_prism,update_final}/report.json`,
`compile_final.json`, `multi.json`, test logs. The historical separation diagnostic
now rejects pose checkpoints instead of silently applying spring-completion labels.

### Potato preparation

Prepared a gravity-supported resting orientation using the uncut tetrahedral
mesh, volume-weighted centre of mass and convex-hull stable-pose candidates.
Selected the supported candidate with lowest COM height. No bottom clipping,
remeshing, fixed particles, altered material/friction parameters or velocity erasure.
Re-exporting the cutting plane produced 247 cutting springs (old orientation: 245).

Three material pairs (E/spring: 1/1, 2/1, 2/2), fixed nominal density, 8e-6 s material
substeps. A continuous passive-rest check measured:
- at 0.75 s: substantial vertical motion remained;
- at 3 s: vertical COM speed 4.3–10.6 mm/s, horizontal components <=0.15 mm/s;
- at 5 s: COM speed ~0.117–0.247 mm/s;
- at 8 s: COM speed ~0.045–0.072 mm/s; particle RMS ~0.54–0.56 mm/s.
Material/ground penalty-contact overlap was approximately 0.493 mm. This is not
knife penetration and is not evidence of calibrated ground-contact fidelity.
These observations support the new preparation with longer settling, not the
claim that orientation alone solved all motion. Production per-material reset and
a subsequent passive interval are checked separately in
`potato_production_reset.json`.

The potato development configuration now uses this asset and 8 s settling.
That has real reset cost and is not yet a measured long-training throughput or
a completed multi-object replay dataset. No long training has been launched.


Production reset follow-up: `env.settle(8)` with the three material pairs above,
then 0.25 s passive physics. Reset COM speed was 0.065–0.143 mm/s; subsequent COM
displacement was 0.0019–0.0092 mm. Fixed-particle count was zero. The 8 s state is
produced by dynamics, with its residual velocities preserved. This is a small
preparation check, not validation of the entire randomized material range.

Validation: full root suite 345 passed / 5 skipped, backend 38 passed. The
subsequently added guard against using the historical separation diagnostic with
pose checkpoints passed its focused suite (9 passed). Board-projection gradient
finite differences, compiled/eager forward/backward agreement, real physical
settlement, optimizer/replay restart, and two-object environment integration are
separate evidence above. No claim of training convergence follows from test counts.

## 2026-11-14 — Whole versus flat-bottom half potato preparation

**Scope and authorization.** One bounded preparation comparison, not policy training:
two freely placed geometries, three E/cut-spring pairs (1/1, 2/1, 2/2; density x1),
up to 8 s of settling, then one nominal-material reference press per geometry.
The user approved this scope. Wall cap was 45 minutes for the preparation comparison;
the physical runner had a 1700 s internal budget and completed both shapes. No
material/contact tuning, fixing of particles, artificial velocity reset, reward
change or new long run.

**Assets.** Whole: `free_potato_stable_20261114_v3`. Half:
`free_potato_half_20261114`, generated from the upper geometric half of the same
uncut mesh, capped horizontally and placed with this cap down. Whole/half heights
are 52.35/26.03 mm, uncut mesh volumes 156.03/75.10 cm3, and tetrahedra 2795/1671.
Half remeshing used the existing fTetWild settings; flattening its support cap
changed remeshed volume by 0.00335%, with no inverted/zero-volume tets. The COM
projection is inside the cap support polygon. Both exported scenes have zero fixed
particles. Since mass, height and discretization change together, this is a
preparation comparison, not an isolated support-shape causal experiment. The half
is our prepared asset, not a claim of using the exact official potato mesh.

**Controls.** The source scene JSON differs only in the mesh filename. Runtime
integration is 8 microseconds; net knife wrench, friction transition 0.01 m/s,
board boundary, force limits [50,50,50,5,5,5] and `free_cut_pose_v1` reward are
unchanged. Knife start clearance is 10 mm for both. Reference feed is 30 mm/s,
ending its commanded vertical displacement at the board stop gap and then holding.
The original 180-step (9 s) task limit remains. Reference cuts restore the nominal
material's complete 8-second settled state, including residual velocity; they
do not test the proposed shorter settling duration.

**Settling measurements.** Samples every 50 ms; an operational stable window
requires six consecutive samples spanning 0.25 s with COM speed <=0.2 mm/s,
mass-weighted particle RMS speed <=1 mm/s, and COM displacement within the window
<=0.01 mm. These are common diagnostic thresholds, not a task reward or a guarantee
about unsampled extrema. Report the first such window that stays accepted through
the remainder of the trace; one run per material pair.

| E / cutting spring | Whole | Flat-bottom half |
|---|---:|---:|
| x1 / x1 | 4.65 s | 0.55 s |
| x2 / x1 | 5.35 s | 0.55 s |
| x2 / x2 | No continuously accepted window by 8 s | 0.55 s |

At 1 s the half's COM speeds are all below 0.006 mm/s and mass-weighted particle RMS
speeds are 0.075–0.081 mm/s. There was no knife contact or spring release during
settling. All three worlds together took 73.28 s (whole) / 68.27 s (half) wall time
to integrate 8 s. The first 1 s of the half trace took 8.73 s wall time, including
cold graph setup; these are three-world measurements, not per-environment costs
or full-training throughput. Earlier potato checks sampled fewer time points and
reported mostly endpoint/short follow-up stability; the stricter sampled window
here exposes intervening residual motion.

**One nominal reference per geometry.**

| Geometry | Pose-task outcome | Steps | Released springs | Peak material resultant | Max six-axis limit utilisation | Minimum knife gap |
|---|---|---:|---:|---:|---:|---:|
| Whole | Timeout | 180 | 73/247 | 25.78 N | 0.602 | +0.182 mm |
| Half | Pose success | 39 | 75/144 | 5.92 N | 0.147 | +0.553 mm |

Neither cut triggered a force/workspace failure or knife-board penetration.
Neither material fully separated. Whole final TCP height error is 3.38 mm against
the 2 mm target tolerance; the blade's lowest point being near the board does not
make the full target pose satisfied. Return sums were -14.46 / 102.84 respectively.
These two reference episodes do not establish learned-policy performance or prove
the whole object impossible to cut.

**Correction from inspecting the exported model.** The effective ground contact
distance is **0.01 mm**, not the 1 mm previously inferred from the JSON
`ground_radius`. The exporter assigns that setting to `contact_closeness`, while
the running particle contact kernel reads `contact_distance` (default 1e-5 m).
Normal ground damping is zero; material damping is not zero. Exported bottom
height is 0.8 mm, so neither mesh starts with the previously suspected 0.2 mm
contact-layer compression. This is an existing configuration/kernel field mismatch;
it was recorded, not changed mid-comparison. Static/trajectory evidence does not
isolate the cause of all oscillation.

**Decision supported.** Prefer the unheld flat-bottom half as the next potato
preparation candidate. A 1 s settling duration is supported by these three traces,
but the two reference cuts above used 8 s, and the full continuously randomized
material/density distribution has not been validated at 1 s. Existing active
`pose_potato.json` remains the whole/8 s development config; no silent task or
training-distribution replacement. Full separation remains a reported outcome,
not the pose-task success gate.

**Reproduction and evidence.** New entries:
`scripts/prepare_half_food_mesh.py`, `scripts/compare_potato_preparation.py`,
`configs/free_scenes/potato_003_half.json`.
Server outputs: `runs/potato_preparation_20261114/`: protocol, asset quality,
initial-ground calculation, `physics/summary.json`, per-shape traces and full
geometry recordings, reference goal audit, comparison PNG/PDF and test logs.
The runner's final source adds effective-parameter/goal-error logging to the
executed experiment without changing its physics or decision rules.

Validation: targeted stability-window tests 5 passed; root GPU suite 351 passed /
5 skipped; backend suite 38 passed. Logs: root_tests_commit.log and backend_tests_commit.log.


### Follow-up: force provenance, separation and display

Nominal potato parameters match the local SliceIt DiSECt fork's
`examples/config/osx/ansys_potato.json`: E=2e6 Pa, nu=0.45, density=1101 kg/m3,
interior/surface cutting stiffness=200. Nominal means scale x1, not calibrated
properties for our scanned whole/half geometries. No new force decomposition was
measured. The 25.78/5.92 N values are total material resultant peaks for the two
reference runs, not individual sensor-axis thresholds.

The pose task pays +100 for the prescribed arrival condition with failure
precedence and the finite-blade board boundary; it does not require separation.
The fixed reference ends on that condition. Source inspection of
`eval_cut_springs` confirms damage reduces stiffness only with knife contact and
positive spring extension (`max(0, ke - incontact * extension * softness)`).
Geometric blade passage is not an unconditional connectivity deletion. This
explains how the two metrics can disagree; it does not identify the cause of the
69 residual half-potato springs or establish an implementation defect. Do not
attribute the residual connections to a policy: these runs use prescribed motion.

Actual scan textures were retrieved from the same HF mirror's
`OmniObject3D_objects_with_texture.zip`, by range extraction of three individual
objects, not downloading the archive. For cucumber_002, cucumber_003 and potato_003,
mesh points, faces and USD transforms equal the existing untextured copies exactly.
`scan_identity.json` records the check. The original renderer used one flat amber
color for all foods, losing these appearance assets.

New optional display pipeline: rigid scan registration, texture sampling onto
barycentrically subdivided recorded exterior triangles, and a Blender Cycles still
renderer. Registration RMS to the coarse settled surface is 1.075 mm (cucumber_003)
and 0.763 mm (potato_003); this is approximate appearance registration, not a
physics alignment or calibration. Recorded positions, cut topology and measured
quantities are preserved. Cut-face colors are schematic; the pre-existing visible
cut seam is NOT evidence of completed separation. No high-resolution scan is
substituted for the deforming physical geometry. Appearance binding checks on the
actual two recordings report zero deviation from barycentric interpolation and
reject a mapping applied to a different recording. Blender 2.82 legacy vertex-color
encoding was checked with an emission-only neutral 0.5 sample (saved RGB 128),
avoiding double gamma conversion. Both actual scene stills were rendered and
visually inspected. These are initial-state previews, not a new policy evaluation.
Evidence: `appearance_checks.json`, `color_encoding.log` and
`runs/multi_scene_preview_20261114/textured_comparison.png`.

### BladeMaster forward path located (read-only, not newly integrated)

GPU7 checkout:
`/dataHDD/dong/Projects/BladeMaster-Warp-forward-check-20261107`,
clean at `4b2d983`, remote `Ju6276/BladeMaster-Lab`. This is the project's
reproduction, not the paper authors' code. Existing
`runs/forward_port_20261107/report.json` records a completed external-tool test:
1452 particles, 3.8 simulated seconds, 40.25 wall seconds including compilation,
two 726-particle components, force peak 0.813 N and torque peak 0.0151 N m.
These are the old banana-preset test's results, not potato/cucumber measurements
or current UR5e integration evidence.

`ExternalToolPort.advance(ToolState, interval)` accepts pose/linear/angular
velocity and returns `force_n`, `torque_nm`. The existing robot driver is Franka.
An UR5e backend needs frame/time/reset/batch adapters and removal or replacement
of DiSECt spring-based progress interfaces. WM+BPTT needs only forward simulation;
it does not require the differentiable BladeMaster branch. Old WM weights cannot
be claimed compatible with the new physical dynamics. Historical force convergence
issues in BladeMaster RESULTS remain relevant; migration alone does not validate
cutting forces. No migration or new training was launched here.


### Render follow-up — remove prepared-face highlighting (2026-11-14)

At the user's request, both video and Blender still exporters now default to
unhighlighted prepared faces. Textured recordings use nearby exterior texture
colors; untextured recordings use the same exterior color. The optional
`--highlight-cut-faces` restores the old schematic band for diagnosis. Geometry,
recorded forces, connectivity and policy inputs are unchanged.

Re-rendered the recorded cucumber and potato initial states; outputs are
`runs/multi_scene_preview_20261114/{cucumber,potato}/no_highlight.png`.
Existing mesh openings remain visible; these are not painted cracks and were not
filled to make the material appear intact. Tool mounting/camera inspection is
documented in METHOD. Validation logs are in
`runs/potato_preparation_20261114/no_highlight_{root,backend}_tests.log`;
actual-scene appearance checks are in `no_highlight_checks.json`.


### Single mounted tool, whole-arm view and 256-world initialization preview

Default renders now show the URDF mounted knife/adapter only, in steel gray.
The blue DiSECt material-contact proxy is still used for physics and is visible
only with `--show-contact-blade`. The previously visible blue edge was the
second representation protruding from the exterior mesh; this change removes
the duplicate display and does not establish geometric identity of the two tools.

Reviewed both Blender outputs:
`runs/multi_scene_preview_20261114/cucumber/single_knife_close.png` and
`single_knife_wide.png`, with corresponding scene, blend and provenance files.
The pale prepared-face highlight is disabled. Existing geometric seams remain;
no material vertices or openings were altered to conceal them.

Initialized and settled **256 actual cucumber worlds** using
`configs/free_cut/pose_cucumber.json`, overriding only world count and disabling
the unnecessary single-group subprocess wrapper. No policy was loaded, no
physical control actions were issued and no optimizer updates occurred.
`runs/batch_preview_256_verified_20261114/` contains config, task, world states,
world-0 display template, completion report, capture checks, full-grid PNG and
render provenance. World-0 particle, knife, joint and body arrays agree with
the template; per-world material and lift differ; local surface indices are valid.
The grid is a display layout of isolated worlds, not their physical positions.
The preview explicitly says NO TRAINING; coarse material colors at this distance
are display colors. It is not a screenshot of the previous free_cut_v3 run.

At inspection there was no active Cutting training on GPU7 or A6000. The earlier
cucumber training used 256 environments; three-object long training has not
started. A6000 GPU0 was running a separate HOAP2/SuperMimic job, left untouched.
Only the requested preview PNGs were copied to Mac temporary storage; project,
data and rendered scene files remain on GPU7.

During preview development, an incorrect provenance call and the use of packed
all-world topology with world-0 vertices were caught before producing the final
image. The capture script now supplies the repo path and extracts the first local
topology block. Failed intermediate directories have no completed report and
are not final evidence.


Final validation: root **351 passed, 5 skipped**; backend **38 passed**.
Logs: `runs/potato_preparation_20261114/final_render_{root,backend}_tests.log`.
The actual-scene display check `single_tool_checks.json` confirms that enabling
the debug blade adds exactly one visible actor per viewport and leaves material
and knife coordinates unchanged; exterior scan colors remain identical.
`capture_checks.json` verifies the 256-world snapshot and local topology.

## 2026-11-14 — Half-potato selection and completed integration work

The user selected the free flat-bottom half as the default potato. Only the active scene/name and per-material settle duration (1 s) changed; the whole asset remains for historical comparisons. Production material sampling covered 128 draws plus all 16 E/density/spring/clearance corners. All **144/144** met the existing stability criteria throughout a 0.5 s follow-up: 0.25 s windows, COM speed <=0.2 mm/s, mass-weighted particle RMS <=1 mm/s, displacement <=0.01 mm. No initial tool contact, cutting damage or fixed particles. This is finite coverage of the declared range, not a proof for every possible draw. Evidence: `runs/half_potato_randomized_20261114/`.

An actual multi-object construction exposed a metadata bug: different per-object settling durations were treated as different reward/task definitions. The metadata now records heterogeneous reset settings per object; shared force, reward and robot definitions must still agree. Serial/process group regression checks passed. A three-object batch (cucumber, half potato, prism) then collected 36 real transitions and performed one WM plus one actor/critic update with finite losses and gradients. Evidence: `runs/half_potato_multi_update_20261114/`. This validates the shared update path, not policy convergence.

Whole/half force differences are observations under changed geometry, mass, height and discretization together. They do not isolate shape from size, nor constitute real-food calibration.

### Complete pose-task reference audit (30 episodes)

Three objects, ten reference commands each. Same nominal material multiplier, 10 mm initial clearance, 30 mm/s target speed, 180-step horizon, gamma=1 and existing six-axis limits. Labels fast/extreme-fast describe commanded feed multipliers (3x/20x), not measured robot-speed multipliers. References bypass the learned six-action mapping and are not replay training actions.

| Object | Reference | Outcome | Steps | Return | Released / total springs |
|---|---|---|---:|---:|---:|
| cucumber | press | success | 46 | 107.620 | 65/65 |
| cucumber | saw | success | 46 | 108.274 | 63/65 |
| cucumber | fast | success | 28 | 104.743 | 62/65 |
| cucumber | extreme_fast | success | 22 | 103.419 | 62/65 |
| cucumber | slow | success | 140 | 96.310 | 62/65 |
| cucumber | stall | timeout | 180 | -20.000 | 0/65 |
| cucumber | retract | timeout | 180 | -17.276 | 18/65 |
| cucumber | slow_press | success | 89 | 101.761 | 62/65 |
| cucumber | hold_in_material | timeout | 180 | -15.136 | 21/65 |
| cucumber | force_attempt | timeout | 180 | -14.128 | 64/65 |
| potato | press | success | 39 | 102.848 | 75/144 |
| potato | saw | success | 39 | 102.876 | 75/144 |
| potato | fast | success | 27 | 101.830 | 75/144 |
| potato | extreme_fast | success | 23 | 101.751 | 75/144 |
| potato | slow | success | 106 | 95.512 | 75/144 |
| potato | stall | timeout | 180 | -20.000 | 0/144 |
| potato | retract | timeout | 180 | -18.671 | 1/144 |
| potato | slow_press | success | 70 | 98.339 | 75/144 |
| potato | hold_in_material | timeout | 180 | -18.293 | 25/144 |
| potato | force_attempt | timeout | 180 | -16.497 | 86/144 |
| prism | press | success | 54 | 107.469 | 262/304 |
| prism | saw | success | 54 | 107.475 | 264/304 |
| prism | fast | success | 31 | 104.528 | 262/304 |
| prism | extreme_fast | success | 25 | 102.685 | 261/304 |
| prism | slow | success | 168 | 94.776 | 259/304 |
| prism | stall | timeout | 180 | -20.000 | 0/304 |
| prism | retract | timeout | 180 | -15.362 | 85/304 |
| prism | slow_press | success | 108 | 100.867 | 258/304 |
| prism | hold_in_material | timeout | 180 | -14.327 | 114/304 |
| prism | force_attempt | timeout | 180 | -14.663 | 256/304 |

The success settlement is the user-selected pose/counter rule without failure. Spring release is a separate measurement; pose success is not a claim of complete physical separation. All terminal cases were retained, including timeouts. No task threshold or reward weight was changed to obtain these results.

| Object, normal press | Peak material force (N) | Maximum sensor-limit utilization | Minimum blade-board gap (mm) |
|---|---:|---:|---:|
| cucumber | 3.540 | 0.068 | 0.183 |
| potato | 7.827 | 0.147 | 0.546 |
| prism | 4.204 | 0.109 | 0.160 |

Maximum reward decomposition error: 3.72e-06; WM reward-equation error supplied with the actual physical next-state fields: 3.72e-06. This checks equations and time alignment, not learned-model prediction accuracy. Physically observed force-terminal episodes: **0/30**. The aggressive force-attempt command is not claimed to trigger a force event if it did not. Unit coverage of force settlement must not be described as a physical force-limit episode.

Raw per-step rewards, components, poses, forces, terminal flags and packed transitions are in `runs/pose_reward_audit_20261114/`. Manifests explicitly record nominal-material and clearance/speed overrides of the source development configs. Shared GPU timings are not throughput benchmarks. These are finite reference comparisons, not policy performance or reward-optimality proofs.

### Two additional material paths

- Four BladeMaster reproduction shapes were converted to DiSECt prescribed interfaces: banana, cake, tofu and luncheon. All passed topology checks and completed nominal pose references, none fully separated. E/nu/rho transfer is explicit; fracture/plasticity is not mapped to cut springs. Cake/tofu defaults now settle for 3 s and have actual reset/step checks. **Ground interpretation corrected below**: the earlier minima of about -4.7 mm initially and -10.4 mm at the end included ghost FEM nodes, so they do not establish occupied-material penetration. The later physical-contact diagnostic supersedes that interpretation. Knife nonpenetration is a distinct condition. All four remain candidate assets. Details: [asset note](BLADEMASTER_DISECT_ASSETS.md).

- Independent BladeMaster/UR5e backend: actual 37-step reference completion, 48 mapped-action transitions, finite and changing WM/actor/critic weights, optimizer/replay restore followed by another update, and two-world masked reset preserving the other world exactly. The real knife now has a swept-board guard at each 1 ms robot substep. Separate `blademaster_pose_v1` uses analytical depth progress, not DiSECt spring labels; do not present it as a matched-reward backend comparison. Details: [backend review](BLADEMASTER_BACKEND_REVIEW.md).

### Integrated validation

Final root suite: **366 passed, 5 skipped**; backend: **38 passed**. Logs: `runs/half_potato_integration_verified_root_20261114.log` and `runs/half_potato_integration_backend_20261114.log`. The added legacy-reward-override rejection test initially had a script-import-path error; the test harness was corrected, its focused suite passed 15 tests, and the final full suite above passed. No long training was launched and no source/weights/data were copied to Mac.

## 2026-11-14 — Corrected tofu ground diagnostic and authorized pose training

The prior candidate report called all-node minimum y physical penetration.
This was incorrect: DiSECt cut tets contain extrapolated ghost nodes on an
empty subdomain side. Their coordinates are not occupied material surface.
The checker now records contact-enabled nodes and cut-edge interpolated
points separately from ghosts. No ground kernel, material or reward changed.

In a new nominal tofu reference with 3 s settling, 70 steps reached the pose,
344/436 springs released, peak reaction was 12.113 N and max force utilization
was 0.305. At 50 ms recording intervals, the lowest actual contact geometry
was -0.309 mm and the lowest ghost was -10.396 mm. No sampled actual contact point
was below -1 mm. The earlier all-node numbers remain raw FEM observations but
must not be interpreted as 10 mm physical penetration. Small genuine penalty
contact overlap remains, and these samples do not bound every material
substep. The 0.75 s historical reference is not exactly reproduced by this 3 s run.
See [ground review](archive/GROUND_CONTACT_REVIEW.md) for source tracing and tests.

The user authorized one A6000 three-object pose-task long run after preflight,
plus separate tofu and conditional BladeMaster work. The planned main run is
192 worlds (64 each), fresh networks, 500 WM-only updates then real critic+h8 + BC 0.25,
with 5000 vector-step / 24 h limits. Preflight/actual launch status is in
[TRAINING_20261114](archive/TRAINING_20261114.md); authorization is not a claim that a
long run has already started.

The active board branch did not contain b1bcb92's schedule persistence change.
Its warmup-anchor/config/accumulator repair and resume tests were restored;
fresh-run cadence and physical/task behavior are unchanged. Historical old
snapshots lacking these fields cannot promise exact cadence recovery.

## 2026-11-14 — Full-batch launch checks and new runs

A6000 GPU1:192 worlds,64 iterations,12288 transitions,10 WM and8 policy updates,
finite optimizer execution.189 unselected worlds unchanged during a three-world
partial reset; actual reset cost26.92 s. Restore advanced 64 -> 70 with updates at 67/69,
preserving replay and cadence. Final microbatches 128/8192 reduce cached WM update
time from~29 to 14.56 s and policy/critic time from~9 to 4.68 s; reset costs remain.
Full details and limits: [launch record](archive/TRAINING_20261114.md).

Fresh main job launched 06:21+01:00 on 2026-11-14, source 905afb9,192 worlds,
at most5000 iterations/24h. Separate GPU7 BladeMaster job uses cf20ef0 and
external solver db06c1a,3 intact-object worlds and nominal parameters. Both start
with500 WM-only updates. These are starts, not completed training results.
The shared integration passed392 root tests/5 skips and39 backend tests; those
include the logging and BM selected-world placement corrections.


## 2026-11-14 — Stop initial multi-object runs for scale/interface correction

The 192-world DiSECt and three-world serial BladeMaster jobs were interrupted
with SIGTERM at the user's request to correct long-training scale/configuration.
DiSECt stopped at iteration802,153984 physical transitions,304 completed WM rounds;
BladeMaster at iteration552,1656 transitions,204 WM rounds. Both had zero policy
updates. Final checkpoints and replay were preserved; they are engineering
history, not completed learning results. The source vector-step cadence generated
disproportionate fitting relative to the small BladeMaster data stream.
The three-component policy wrench interface was a retained design, not a missing
WM force output. The next run explicitly versions the six-component actor input.
Restart details and capacity measurements: [TRAINING_SCALE_20261114](archive/TRAINING_SCALE_20261114.md).


## 2026-11-14 — Full-wrench collection-scaled restart preflight

The old192-world DiSECt and3-world BM runs were stopped before policy training.
New interface:10x34 actor observation including all six wrench components,
same6 controls; transition-scaled updates, real critic32, actor8, BC0.25.
No claim of a new learning algorithm or calibrated real-robot forces.

DiSECt GPU7:3072 worlds(1024/object),40 steps/122880 transitions; full WM, actor,
critic updates, selected reset, actual geometry capture and checkpoint/replay
restore plus continued updates passed. Effective real critic targets131072;
final optimizer counts20/2/40. Mean physics-step23.924s. Free memory3.449GiB after
learning and4.701GiB after restoration. This is capacity validation, not a
success-rate result; no episode ended during this40-step initial-policy sample.
Preserve separate earlier reference/reward/terminal evidence.

BM GPU7:256 true batched worlds;40 steps/10240 transitions, full WM/actor/critic32
updates and replay restore passed. Root420/5 skipped and backend39 at its own
commit; external new forward regressions4 passed, full external2164 not completed.
BM long training is deferred by user decision.

After integration, root425 passed/5 skipped; backend39 passed. Fixed named
checkpoint replay retention: each periodic checkpoint now has its own paired
replay; last.pt still uses replay.pt. Existing metadata/iteration rejection and
legacy matching fallback remain. Mid-gradient emergency snapshots are resumable,
not identical to uninterrupted execution. See TRAINING_SCALE_20261114 and
BLADEMASTER_BATCHING_REVIEW for exact scope, code identity and paths.

The bounded main DiSECt run has now launched on GPU7 PID1048242, frozen02203b1,
3072 worlds. Runtime manifest agrees with the validated recipe. First-step
training geometry is captured and rendered; initial collection is not a policy
performance result. Exact paths and synchronization record:TRAINING_SCALE_20261114.

## 2026-11-14 — Native BladeMaster cuts at three plane offsets

User requested changing the single-cut location without preparing a new cut
interface, and comparing the reaction. GPU9 (134.100.13.219, RTX 4090) was idle
before this bounded test. GPU7's active learning job was not changed.

Evidence: `runs/bm_offset_native_20261114/` on GPU7 contains the exact
`check_offsets.py`, log, report and nine raw trajectories. The original test
directory on GPU9 is
`/dataHDD/dong/Projects/BladeMaster-offset-check-20261114`.
Source was exported from BladeMaster-Warp-batched commit `e2e5221`.
GPU9 used Warp 1.18.0. This is a native scalar ExternalToolPort test, not the
UR5e/FDCC task, learned policy, batched training, or a force-limited success test.

Protocol: identical intact asset bytes within each object, nominal transferred
elasticity/density and paper-banana remaining parameters; settle 0.2 s, blade
normal offset -10/0/+10 mm, identity orientation, prescribed 3 cm/s downstroke,
2 ms coupling/sample interval, stop edge at +1 micrometre above board, then hold
0.2 s. Every trial starts from a fresh connected undamaged asset. No remeshing,
pre-cut interface, or offset-specific material change. Budget 20 minutes; all
nine trials completed within that budget.

| Object | Plane offset (mm) | Final component sizes | Peak force (N) | Mean force (N) |
|---|---:|---|---:|---:|
| cucumber | -10 | 2261 / 3997 | 10.483 | 2.151 |
| cucumber | 0 | 2762 / 3496 | 5.757 | 2.086 |
| cucumber | +10 | 3133 / 3125 | 6.463 | 2.172 |
| half potato | -10 | 812 / 1965 | 6.972 | 2.925 |
| half potato | 0 | 1368 / 1409 | 8.396 | 3.221 |
| half potato | +10 | 1774 / 1003 | 9.371 | 2.988 |
| prism | -10 | 7293 / 11220 | 1.722 | 0.627 |
| prism | 0 | 9537 / 8976 | 1.680 | 0.572 |
| prism | +10 | 11220 / 7293 | 0.844 | 0.444 |

All trials start with one component and finish with two according to native
reference-grid/side-code connectivity. Force is the norm of the **interval-average
material-on-tool force** at 2 ms, summarized over the prescribed downstroke,
excluding the final hold. It is not an instantaneous material-substep peak,
filtered UR sensor wrench, or board reaction. No force-limit abort was enabled.

This supports changing the cut plane location on the same intact asset in the
native backend. It does not establish arbitrary orientation/multi-cut capability,
UR5e reachability, policy generalization, or real force accuracy. Force differences
are observed, not solely attributed to geometry: only one trial per setting,
with discretization/contact/free-object motion all contributing. In particular
the prism's asymmetric offset peaks should not be interpreted as a calibrated
physical prediction. The training configuration still uses fixed cut locations.

FDCC terminology clarification: current tensor controller ports the SliceIt
cambel/cartesian_controllers FZI implementation (f4359b9). It is forward-dynamics
Cartesian compliance/admittance control, producing joint **position** commands
in this integration. It blends pose-derived virtual spring and wrench error;
it is not UR's built-in force_mode nor a direct joint-torque policy.

## 2026-11-14 — A6000 shared-device BladeMaster training capacity

Stopped the user's HOAP-2 job on A6000 GPU0 by SIGINT to its verified process
group (1057890); supervisor exited with code 130. Existing HOAP outputs and
checkpoints were retained. Another user's openpi remained active, consuming
26,224 MiB on GPU0 and 44,110 MiB on GPU1. GPU0 had 22,265 MiB free before probes.

Same intact cucumber / half potato / prism assets and nominal material presets
as the 256-world BladeMaster configuration. Full six-wrench observations (34),
state 980, source-sized networks, WM batch 2048/window 8/10 minibatches, policy
batch 4096/actor h8/BC 0.25, real critic 4096 windows of 32 steps/131,072 samples.
Microbatch 512 and critic microbatch 8192; replay capacity 1000 per world.
Each probe collects 40 fresh physical steps, then updates, resets one world per
object, saves/restores learner+replay, and updates again. WM warmup is skipped
only to exercise the learning lifecycle in this bounded preflight.

Cutting_DEV code: 3738271 (clean); external BladeMaster source: e2e5221 (clean).
The run-local probe adapts scripts/check_training_capacity.py by selecting
BladeMasterEnvironment and overriding both group counts and total worlds.
GPU0 free-memory reserve criterion: 3 GiB after learning and restoration.
Total active probe time: about 63 minutes, within the two-hour budget.

| Worlds | Cucumber / potato / prism | Mean physical step (s) | Observed peak process memory (MiB) | Free after restore (GiB) | Result |
|---:|---|---:|---:|---:|---|
| 256 | 86 / 85 / 85 | 27.69 | 22004 | N/A | Updates passed; robot reset allocation failed |
| 192 | 64 / 64 / 64 | 22.33 | 17302 | 4.84 | Full lifecycle passed |
| 208 | 70 / 69 / 69 | 23.50 | 18354 | 3.81 | Full lifecycle passed |

For 256 worlds the complete initial WM, actor and critic updates were finite.
Only 0.312 GiB remained after learning; the MuJoCo Warp collision temporaries
during selected reset exhausted device memory. No successful restore is claimed
for that size. The memory watchdog stopped that probe and resources were released.

192 and 208 worlds both passed all phases. Unselected worlds were unchanged by
reset, and each completed 20 WM optimizer steps, 2 policy steps and 40 critic
steps including restoration. 208 worlds left 3.814 GiB after restore and
completed in 1,173 s; 192 left 4.841 GiB and completed in 1,123 s.
These short collections had no completed episodes; physical reference task
completion remains the separate previously recorded check.

208 is the largest fully validated candidate in this shared-device probe.
224 was estimated below the chosen reserve using measured scaling and was not
run; neither the exact shared-device maximum nor an exclusive 48 GB maximum was
measured. The earlier GPU7 256-world preflight used microbatch 64 and a different
restore allocation pattern; these are not matched hardware-throughput tests.
The timings here include contention with openpi.

Evidence on A6000:
 /dataHDD/dong/Experiments/Cutting_DEV/runs/bm_a6000_capacity_20261114/
contains configs, probe/supervisors, logs, reports, and successful learner/replay
checkpoints. GPU7 has the same records excluding checkpoint tensors under the
same runs name. No new BladeMaster long training was launched; GPU7's active
3072-world DiSECt-backed WM+BPTT run was unchanged.

GPU7 status at this review: iteration 160, 491,520 transitions, WM updates 715,
policy updates 215, running. Its latest 1,000 completed training episodes were
838 pose successes and 162 force-limit failures (all failures from prism).
Earlier windows differed substantially; this is online completed-episode data,
not a frozen success-rate evaluation. W&B run k48zc1lh remains offline with no
cloud URL or sync marker.

## 2026-11-15 — GPU9 BladeMaster online long-run deployment

User authorized moving the BladeMaster run from occupied A6000 to idle GPU9
(RTX4090), through tams61. No other user's workload was stopped.
Shared training source931e2b5; native material sourcee2e5221.
Source changes since GPU7's frozen02203b1 are documentation and the optional
cache guard; the guard is enabled only for this new launch. Existing GPU7
training was not restarted. Task/config/assets/robot reset/controller hashes
are recorded in the run plan and matched the tested A6000 deployment.

### Capacity and memory correction

All 256-world cases used86 cucumbers,85 half potatoes and85 prisms. First
40-step probe collected10240 real transitions, completed WM,actor,real critic,
masked reset and checkpoint/replay restoration with finite updates. It left
only1.90GiB free, below the3GiB pre-physics reserve. Expandable allocation
left2.14GiB after restore; smaller storage chunks plus allocator GC also did
not meet the reserve. These are headroom failures, not algorithm/physics
failures. Reports are retained separately; none was labelled a complete pass.

Measurement showed about2.4–2.6GiB of inactive Torch cache retained while Warp
also needs memory. The optional --cuda-cache-reserve-gib3 releases unused
Torch blocks before physics. Live tensors, replay, effective batches, schedules
and reward are unchanged. Restore uses the original storage microbatch512.

Final probe reused the recorded10240 transitions for learning, performed four
new physical steps, masked reset with bitwise-unchanged unselected worlds,
full-world reset plus four more physical steps, checkpoint/replay restore and
another full update. These eight physical steps are additional validation, not
a new10240-transition collection. Optimizer totals:20 WM,2 actor,40 critic.
All finite; full-reset continuation passed. Cache release raised free memory
from2.19 to4.32GiB before reset, and from2.12 to4.28GiB after restore.
256 is the validated configuration, not an exact maximum-capacity claim.

### Regression and launch

New guard tests:7 passed. Full root suite onGPU9:430 passed,7 skipped;
backend:39 passed. Initial suite failures were missing regression scene assets
and unset DiSECt extension environment, fixed before the successful reruns.
The optional MPPI dependencies are absent onGPU9; those tests stay skipped.
No claim of full external BladeMaster differentiation-suite coverage.

Frozen run checkout:/dataHDD/dong/Projects/Cutting_DEV_bm_long_20261115.
Run:/dataHDD/dong/Experiments/Cutting_DEV/runs/bm_gpu9_20261115.
Started training PID138581, seed0,fresh networks, online W&B required.
Budget24hours/1280000 new transitions; actor8,real critic32,BC.25;
500 WM-only rounds,10240-transition warmup,640 transitions/WM round,
WM batch2048x10,actor4096,critic4096x32,microbatch512/8192.
Nominal native BladeMaster materials, fixed clearance and target speed;
this does not reproduce DiSECt's material randomization distribution.

Run records include plan.json,launch_command.json,launch_status.json,
preflight*/report.json,root_tests_ready.log,backend_tests_complete.log and
training/{manifest,progress,wandb}.json. Full replay and weights stay onGPU9.
Initialization and the first online records are checked separately below.

Runtime manifest verified: real critic32,actor8,BC.25,gamma1,256 worlds,
six-wrench interface,cache reserve3,online tracking and fixed budget all match
the launch plan. The source tree was clean. First cloud read confirmed running,
iteration4,1024 physical transitions,policy updates0 (initial collection).
W&B run name:bm_gpu9_three_object_wrench6_256_20261115.
URL: https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/aqoyihk2

## 2026-11-15 — Storage and document maintenance

Removed 17 inactive worktrees only after preserving source snapshots, staged and
unstaged patches and Git refs. Deleted 28 completed-preflight tensor files and
one exact historical replay duplicate; replaced two identical GPU9 rolling
diagnostic copies with relative links to retained numbered files.
Tensor bytes reclaimed: GPU7 6,269,467,188; A6000 1,511,716,284; GPU9 2,295,040,722.
Live training checkouts/checkpoints/replay, dependency environments, assets and
unique research evidence were preserved. Exact paths, hashes and recovery
instructions are in each server's operations/cleanup_20261115 ledger; see DATA.

No source/config/physics/training changes. Six completed launch/review notes
moved to archive; six superseded plan/paper drafts now use immutable Git links.
Corrected stale README/DATA/machine status. All documentation relative links
resolve and git diff --check passes.
CPU-only regression on GPU7 (CUDA hidden to avoid the live training allocation):
root 415 passed / 22 skipped; backend 27 passed / 12 skipped.
This is not a new GPU physics validation. Active GPU7/GPU9 training advanced
during cleanup; 2026-11-15 01:07 UTC snapshots were policy421 and WM20 respectively.

## 2026-11-15 — Completed frozen policy evaluations and GPU9 SIGILL

These are bounded development evaluations requested by the user, not new
training or sealed tests. No long run was restarted.

### Training exit status

GPU7 completed its 24-hour budget: iteration346, 1,062,912 transitions,
policy1109 / WM1609, elapsed86,414.9 seconds. The760 episodes completed after
iteration192 were all timeouts (347 cucumber,124 half-potato,289 prism).
An earlier success rate over completed episodes does not describe the final
policy or imply that no episodes ended later.

GPU9 stopped before budget completion: launcher exit -4 (SIGILL), last progress
iteration3785 / 968,960 transitions / policy998 / WM1498. At2026-11-15
21:34:18 Europe/Berlin, the kernel recorded pt_main_thread PID138581 trapping
an invalid opcode inside libcuda.so.595.80. The launcher and training process
are gone; progress.json's running state is stale. No Python traceback, usable
core dump, or nearby visible Xid/OOM-kill record was found. The fault location
does not prove a driver defect; the triggering cause remains unresolved.
Torch2.4.1+cu121 uses driver595.80 on the RTX4090. Ten CUDA matmul/backward,
synchronize and cache-release smoke iterations passed; the frozen evaluation
below also completed. Neither establishes that the long-run crash is fixed.
W&B reports finished, which must not be read as budget completion.

### Protocol and results

One deterministic first episode per world,16 worlds/object,48 total, seed20261115;
unchanged task/reward/limits; every case runs to its own termination. Maximum
one hour per evaluation, no learning. Exact task/interface/backend metadata are
checked. Manifests record checkpoint/config/script hashes and source revisions.
Raw transitions, goals, actions, rewards, termination and conditions are saved.

| Frozen checkpoint | Pose success | Other outcomes | Mean return | Mean steps |
|---|---:|---|---:|---:|
| GPU7 last, policy1109 | 0/48 | 48 timeouts | -10.9737 | 180 |
| GPU7 iteration192, policy373 (evaluated on A6000 GPU0) | 0/48 | 48 workspace failures | -99.8906 | 55 |
| GPU9 iteration3456, policy866 | 48/48 | no force/workspace/timeout failures | 106.9391 | 34 |

The saved GPU9 checkpoint is policy866, not policy998 at the crash. All three
objects scored16/16 pose successes: cucumber mean return106.5494 /33 steps,
half-potato107.2362 /35 steps, prism107.0317 /34 steps. Mean episode peak material
forces were6.010,9.627,3.066 N respectively. Maximum sensor-limit utilization
over the run was0.18353. Every final material component count was2; this is
a diagnostic, not an independently validated intended-slice quality criterion.

GPU7 final per-object mean returns were cucumber-9.9089, half-potato-15.0810,
prism-7.9312; all48 timed out without full material separation. The earlier
checkpoint also had no fully separated episodes. Both frozen checkpoints fail
in different ways, so these data do not isolate degradation of the final
checkpoint from a previously verified successful frozen policy. Historical
training success has not been reproduced by these two deterministic evaluations.

GPU7 conditions are randomized within the training material/clearance/speed
distribution. GPU9 uses nominal replicas, not48 independent randomized-material
generalization cases. Backends and distributions differ: do not rank algorithms
from this table. Pose success remains distinct from material separation.
The two DiSECt evaluations have exactly matching48 case records and matching
scene/config hashes. They ran on different GPUs; identical physical floating
point trajectories are not asserted.

### Code, checks and artifacts

Added scripts/evaluate_pose_policy.py with separate pose-success and separation
metrics, first-episode accounting, finite-value checks, raw trajectory storage,
explicit incomplete/error summaries and finally-based cleanup. The historical
evaluate_multi.py now rejects pose-task versions rather than mislabeling pose
success as separation. No training objective, reward, physics or action mapping
was changed. Root tests:418 passed /22 skipped; backend:27 passed /12 skipped,
CUDA hidden on GPU7 to avoid competing with the evaluations. These are CPU
regressions, not GPU unit-test coverage; the three physical evaluations above
are separate execution evidence. Both evaluation scripts compile.

Remote artifact root on each host:
/dataHDD/dong/Experiments/Cutting_DEV/runs/frozen_pose_eval_20261115/
- GPU7: disect_last/, comparison.json, plan.json, root_tests_final.log,
  backend_tests.log.
- A6000: disect_192_ready/ (completed); disect_192/ retains the initial failed
  environment attempt (Ninja missing from PATH), not counted as an evaluation.
- GPU9: blademaster_3456/.
- GPU9 exit evidence: runs/bm_gpu9_20261115/exit_investigation_20261115.json.

All successful evaluations used script SHA256
44d935e21825c3dd468bdcd4765879eeb4d8f898930be66053d14cffb412c449.
Full checkpoint identities and hashes are in comparison.json and manifests.
At final occupancy check, GPU7/GPU9 and A6000 GPU0 were idle; A6000 GPU1
remained occupied by user tu's feeding training (~29 GB), which was untouched.
No evidence connects that separate-host workload to the GPU9 exit.

Next work is a targeted investigation of the saved main-policy failure traces
and the GPU9 runtime fault before proposing another long run, not another
automatic training restart.

## 2026-11-15 — Video evidence, inference audit and tool appearance correction

At the user's request, repeated the48-case frozen evaluations with physical
geometry recording (at most one hour each; no training). Selected case0,16,32
before execution: the first cucumber, half-potato and prism world. Main run
again had0/48 successes and48 timeouts, mean return-11.0201. BladeMaster again
had48/48 pose successes, mean return106.9389. Numerical force values differ
slightly across executions; these clips are labelled reruns, not footage from
the original training.

Main saved trajectories show selected cases ending about69.5–73.7 mm sideways
from the target and1.46–2.96 mm above its target height. None records board
contact. Thus this is actual failure to meet the pose condition, not just a
missing success label. It does not identify why the learner chose those actions.
Reconstructing policy histories from recorded observations for steps1,10,50,180
on CPU reproduces recorded GPU actions within2.87e-6; train/eval modes agree
exactly. The environment/policy/physics source did not change between frozen
training02203b1 and evaluation6ddda87; the intervening training-code change was
an optional cache guard. This excludes neither learning bugs nor batch-size
effects and is not a full correctness proof.

The first BladeMaster video previews incorrectly displayed the inherited URDF
knife_and_adapter appearance as if it were the physical BladeMaster blade.
This is a visualization error; those previews are superseded. BladeMaster
training uses its configured wedge (half-size0.0005,0.06,0.04 m; bevel0.005 m).
The adapter keeps the original knife-center position at initialization but
sets a different initial blade orientation, then drives it by a fixed EE
transform. It does not simulate frictional gripping. Do not describe the URDF
appearance blade as its material-contact geometry or its mount appearance as
validated hardware. No training physics was changed to match the picture.

The common current frozen-evaluation renderer now displays one actual backend
collision blade, omitting the mismatched combined URDF appearance knife/adapter.
Robot visuals share the existing visual-mesh loader; material colors are display
colors. BladeMaster's display surface is reconstructed from measured particles,
never used for contact or success. No artificial crack overlay is added.
The visible lack of corresponding mounting hardware is labelled rather than
concealed by a fabricated adapter. This is a geometry/behavior diagnostic, not
a claim of finished real-robot fixture design.

BladeMaster displayed vertices are transformed from recorded EE poses and the
same vertices_in_ee used by board checks, through the MuJoCo base frame.
Selected final physical knife gaps above the board are0.00650,0.00522,0.00693 mm.
They must not be inferred from the old URDF visual knife. The DiSECt-backed main
video uses its recorded material knife vertices directly.

Current commands: evaluate_pose_policy.py --record-geometry, then
render_pose_evaluation.py for either backend. Historical record_episode.py /
render_episode.py remain for their old recording formats and shared visual
helpers; render_training_snapshot.py serves the all-environment static view.
They are not interchangeable entry points. The method is the user's WM+BPTT;
SliceIt supplies the adapted robot/controller/task conventions, while DiSECt
and BladeMaster name material backends, not different policy algorithms.

Artifacts under runs/frozen_pose_eval_20261115:
disect_video/, blademaster_video_ready/, videos_physics/ (current six videos).
videos/, videos_checked/, videos_verified/ are superseded previews.
Main policy_execution_audit.json and selected_case_motion.json are under
disect_last/. BladeMaster blade_render_alignment.json and physical_blades.npz
are under blademaster_video_ready/. The exact recorder used by these reruns is
archived as evaluate_pose_policy_geometry_used.py; later recorder revision also
stores EE poses directly in snapshot metadata for self-contained rendering.
The user-requested videos and preview frames alone were copied to the Mac,
not checkpoints or project source.

Regression:418 root passed /22 skipped;27 backend passed /12 skipped, CUDA
hidden. Both physical recording reruns completed. This validates recording
execution, not a repair of main-policy learning or GPU9's earlier SIGILL.


### 2026-11-15 — Correct the missing handle and restore requested appearance

User review correctly rejected the previous videos_physics render: removing the
entire knife_and_adapter.stl also removed the handle and mount, leaving the
physical blade visually floating. The caption was not a substitute for the
requested mounted tool view. Those videos are superseded by videos_mounted/.

Retained the original URDF grip/handle and clipped only the old blade beyond
its shoulder. The physical blade geometry and trajectory remain unchanged.
BladeMaster's differently oriented wedge uses a clearly labelled display-only
shank/collar to represent its already implemented fixed kinematic connection;
it adds no physics, inertia, contacts or reward changes and does not claim
validated mounting hardware. Relative physical blade motion in its recorded
gripper frame is below1e-12 m; the renderer rejects nonrigid motion above1e-5 m.
The selected main proxy varies by0.488 mm relative to the gripper across the
recording; that recorded variation is preserved.

Restored a single silver-gray knife, the original UR5e visual meshes, shared
green cucumber/tan potato/muted prism display colors, no blue proxy overlay and
no artificial crack marking. Colors now come from render_episode.MATERIAL_STYLES
for both current backends and the historical renderer. BladeMaster particle
display surfaces use summed Gaussian density and surface smoothing to reduce
white rims and voxel-like appearance; they never determine physical connectivity.
The whole-arm camera was widened, and --preview now permits start/middle/end
inspection before encoding.

No policy execution or physical re-evaluation was repeated: these are rerenders
of the same saved trajectories and their results are unchanged. The previous
videos_physics/ and related Mac previews are not the current deliverable.

Validation for this display correction: root418 passed /22 skipped; backend27
passed /12 skipped (CUDA hidden). All six MP4s encoded, with first/middle/last
frames saved and reviewed. Both camera views include the original handle; the
close view is fitted to the material, blade and mount over the full trajectory.
Each BladeMaster render checks the mount rigidity before encoding.


## 2026-11-15 — Restore the established studio visualization

Historical reference: d74ad44/f9abb84 and
runs/multi_scene_preview_20261114/cucumber/single_knife_close.png.
The new frozen-evaluation renderer had bypassed the scan appearance and studio
lighting, causing a visible regression even after knife/handle corrections.

The duplicate render_pose_evaluation.py was removed. Its valid recorded-geometry
adaptation is now export_evaluation_scene.py, feeding the existing
render_episode_still.py, which supports stills and videos using one style.
The retained historical VTK force-chart renderer is not the presentation default.

Actual geometry is unchanged: one physical blade, original handle, no blue
proxy or artificial crack band. BladeMaster's visible fixed-joint shank remains
display-only; it is not proof that a hardware mount has been validated.
The URDF camera-cable envelope box is hidden in studio output only.

Cucumber FEM reuses the hash-verified reference appearance after topology and
rigid correspondence checks. Other food surfaces use approximate scan-color
transfer, disclosed in sidecars; they do not inherit the original scan UV.
Blender 2.82's system BLAS crashed in a tiny camera normalization operation;
elementwise arithmetic avoids that renderer-only compatibility issue.
View-layer denoising is explicitly enabled (the older scene-only switch did
not actually denoise these frames).

Validation so far: root suite 418 passed / 22 skipped; backend suite 27 passed /
12 skipped, with CUDA hidden during concurrent CPU rendering. Optional GPU
paths were not rerun for this visualization-only change. Both suites must use
their own working directories. Studio start/middle/final previews have been
inspected for the two cucumber recordings. Full-clip QA follows below.


Full-clip QA completed: six MP4s at 1100 x 780, 10 fps, 1x playback, plus a
one-second final-state hold. Main clips are 10.0 s each; BladeMaster clips are
2.7 / 2.8 / 2.7 s. All decode fully; frame counts match sidecars. Start, middle
and final frames of every clip were visually inspected. The camera-cable box is
absent, robot meshes and metal blades are retained, and food appearance no
longer uses the superseded flat-color layout. Half-potato display subdivision
and per-frame shading normals preserve the underlying FEM surface.

Final code re-export reproduced 137 prism geometry arrays exactly. Root tests:
418 passed / 22 skipped; backend: 27 passed / 12 skipped (CUDA hidden).
Outputs: runs/frozen_pose_eval_20261115/studio/, with qa.json and contact sheet.
The five obsolete videos* preview directories were deleted after validating
replacement checksums; their JSON provenance is retained in
studio/superseded_render_manifests.json. Raw evaluation recordings are retained.
Only videos, preview images and small manifests are delivered to the Mac.

## 2026-11-16 — Shared geometric depth reward (pose_depth_v2)

User request: remove dependence on DiSECt spring release in progress reward and
use BladeMaster-style downward depth. Implemented as a new opt-in task for both
backends. Pose +100, failure precedence, the arrival counter, board interlock,
force limits and action interface are unchanged. The speed penalty's progress
factor was changed as well; setting only the positive progress weight would
have left spring-dependent reward in place.

Per-world progress uses maximum depth from the actual reset blade-proxy gap to
the positive board stopping clearance. The new span is stored in replay/state
history, reset per world, and preserved on exact physical-frame restoration.
The shared WM computes this progress from predicted joint state analytically.
It has no learned release-progress output and no spring-progress dynamics input.
Old WM/replay/critic shapes and task metadata are incompatible; no old checkpoint
is silently resumed or its published rewards reinterpreted.

Verification on GPU7: three objects, one world each per backend, 16 reference
steps (48 transitions each), with a masked reset of world 0 after step 8.
This is interface verification, not a complete-episode or trained-policy test.

| Backend | Maximum physical/WM equation reward difference | Wall time |
|---|---:|---:|
| DiSECt | 1.46e-08 | 157.5 s |
| BladeMaster | 1.43e-08 | 45.2 s |

Unselected worlds' frame tensors were exactly unchanged by reset for both.
Progress was monotone within each episode and zero after reset. All measured
wrenches remained finite. Physical and model termination flags agreed on these
short segments. Unit tests additionally cover pose settlement, force/workspace
precedence (retained tests), bounded/re-entry progress, independent separation
labels, equal equations across backend identities, finite BPTT gradients,
a finite WM fitting backward pass, and rejection of old model tensor shapes.

Raw reports: runs/depth_reward_20261116/disect_env.json and blademaster_env.json.
Commands: source runs/sac_env.sh, then use scripts/check_pose_depth.py with
configs/free_cut/depth_three_object_check.json (base .venv) or
configs/blademaster/depth_three_object_check.json (cutting-mppi-20261108 Python,
which includes SciPy). Use PYTHONPATH=.:backends/mujoco.
The first invocations omitted the DiSECt cache environment / used a BM Python
without SciPy; they failed before collecting transitions and remain recorded
separately. Successful checks used the documented environments.

Physics interpretation: eval_cut_springs lowers stiffness only when the spring's
knife contact is nonzero and its extension is positive. Its update is
max(0, ke - incontact * max(length-rest,0) * softness).
There is no unconditional severing when a blade passes a geometric height.
Thus arrival with residual connections is allowed by the implemented mechanism;
this does not identify the cause of any particular residual spring or exclude
contact/geometry bugs. No damage rule or residual spring was changed here.

Limitations: depth reward is motion shaping, including free-space descent; it
does not establish material separation. The 10-reference full-horizon preference
audit was adapted to the new task but not rerun in this bounded check. No new
long training or performance claim. Physical knife geometry/mounting still
differs between existing backends and remains a separate unification task.

Final CPU regression on GPU7: root 426 passed / 22 skipped; backend 27 passed / 12 skipped. GPU execution was checked separately by the two three-object probes above; these counts do not mean every optional GPU test ran.


## 2026-11-16 — Shared physical knife and frozen-policy mechanics

Implementation parent: fe36fe8; new revision disect_knife_rigid_mount_v1,
paired with pose_depth_v2. External forward source parent e2e5221, worktree
BladeMaster-Warp-shared-knife-20261116.

The physical blade has the DiSECt ten-vertex geometry (150.000006 x 59.040001 mm,
3.500000 mm spine thickness, 0.080 mm shoulder). Both backends use a canonical
rigid EE mount. BladeMaster uses two-slope contact/constraint kernels; DiSECt
synchronizes the tool to rigid FK through substep interpolation. Material laws
and backend-specific integration remain separate.

Three-object reference probe, GPU7:
- DiSECt: cucumber/half potato/prism pose success at 41/33/40 steps; maximum
  physical-tool/FK vertex discrepancy 1.02e-8 m; minimum board clearance
  1.004e-6 m. All three still had material connections.
- BladeMaster: pose success at 50/50/50 steps; discrepancy 6.43e-9 m;
  minimum clearance 1.108e-5 m. Each native material graph had two components.
- Sixteen-step reward and partial-reset checks passed on both backends; maximum
  physical/WM reward discrepancy 1.47e-8 (BM), 1.46e-8 (DiSECt).
Reports: runs/tool_analysis_20261116/disect_full_tool_probe.json,
bm_full_physical_tool_probe.json, shared_disect_check_v2.json and
shared_bm_check_final.json (final configuration). The earlier BM FK-only probe is superseded by the
physical-solver-pose probe.

Existing-policy analysis uses the recorded original physics: main 1109 updates
(48 first episodes, all pose timeouts), BM 866 updates (48 first episodes,
all pose completions). Old tools, task rewards and conditions differ.
Checkpoint identities and reconstructed actions match (max 6.68e-6/1.67e-6).
Re-executing the analysis from raw traces produces identical output arrays.

- Sensor force/moment, policy outputs, mapped FDCC gains/increments, feed,
  target errors and board clearance are plotted against time.
- Both checkpoints disable pitch target increments with max_twist=[1,0].
- PCA includes reset, pre-contact-proxy, contact+3/+5 and goal-only controls.
  BM reset features overlap and later separate by object; other motion/history
  factors vary at the same time. A material-identification claim is unresolved.
- Replacing older force history by current force leaves the main mapped gains
  unchanged and changes its x-target increment by 6.45 micrometres/step on
  average; BM changes stiffness and x increments. Main action clipping is
  about 84%, masking some raw-network sensitivity.

Figures, arrays and reproduction check:
runs/tool_analysis_20261116/{main,blademaster}/ and analysis_reproduction.json.
Detailed interpretation: POLICY_MECHANICS_20261116.md. No new policy training.


Final root/backend GPU regression on GPU7: **446 passed, 5 skipped** at root,
**39 passed** in backends/mujoco. The GPU runs include the actual material/tape
and packed-world tests; DeprecationWarnings come from the upstream dflex AST
compiler. Shared-profile external tests use the external repository's pinned
requirements in an isolated dependency directory.


External BladeMaster full regression: **2163 passed, 8 skipped, 18 subtests
passed**, 2124.99 s, using the isolated pinned-requirement environment.
Recorded separation-event audit: the main-policy trajectories contain zero
complete-separation events; BM traces have final component counts but no
per-step separation timestamp. Figures preserve this distinction.


## 2026-11-16 — Shared-knife scene and reward audit

Captured one settled nominal world for cucumber/half potato/prism per backend;
no policy, control steps or optimizer updates. Shared pose_depth_v2 reward,
physical blade and rigid mount; actual start clearance remains unmatched:
DiSECt 5.52/4.47/4.71 mm, BM 17.45/29.45/5.59 mm above highest material sample.
BM currently starts all groups using the maximum object height. The scene
comparison preserves this difference.

Eight studio PNGs (six close views, two wide views) use the existing studio
renderer. Four corresponding camera transforms/scales compare exactly equal.
The robot-base transform is now recorded explicitly for BM: the old export
assumed the base link had visible geometry and raised ValueError otherwise.
The shared optional knife uses the same display connector for both backends.
Report: runs/shared_scene_views_20261116/{scene_geometry_audit,render_audit}.json.
The wood-board rectangle is a display footprint, not the support-plane extent
or robot EE workspace. TASK_FREE_CUT now states the exact speed-penalty formula,
six action meanings and distinction between EE limits and object sliding.

Validation after the capture/export repair: root GPU suite 446 passed / 5 skipped
(239.70 s); backend 39 passed (6.22 s). The six exported scenes and eight
rendered images completed; camera identity and snapshot provenance checks passed.
No physics, reward, action parameters or policy weights were changed in this audit.


## 2026-11-16 — Optional material-motion envelope (shared task)

User selected small free motion, an excess-motion soft cost and a larger failure
boundary. Added opt-in task.object_slip, revision planar_mass_rms_v1, in both
shared_motion_check configurations. Physics, friction, original configs and old
checkpoints are unchanged. No new long training.

Existing first-episode geometry traces (16 cases/object/backend) were summarized
using mass-weighted RMS horizontal particle displacement from episode start.
This measure includes deformation and fragment opening; it avoids cancellation
between oppositely moving halves. Different tools, materials, starts and episode
durations make these old traces descriptive only.

| Backend / object | Min / median / max peak RMS (mm) |
|---|---|
| DiSECt cucumber | 33.51 / 50.32 / 59.87 |
| DiSECt half potato | 29.62 / 44.28 / 76.50 |
| DiSECt prism | 2.21 / 36.02 / 62.94 |
| BladeMaster cucumber | 20.23 / 20.28 / 20.33 |
| BladeMaster half potato | 19.94 / 19.95 / 19.95 |
| BladeMaster prism | 18.00 / 18.09 / 18.21 |

Source: runs/object_slip_20261116/old_policy_motion.json, derived from
runs/frozen_pose_eval_20261115/{disect_video,blademaster_video_ready}.
Development values: 25 mm free, >60 mm failure; capped quadratic excess cost
20/horizon per step. They are not calibrated physical limits or benchmark gains.
This is not a retrospective change of the recorded successes/failures.

Checked actual ground-friction settings: DiSECt contact_mu=0.2 for all three
scenes; BladeMaster paper-banana-based object configs load static=dynamic=0.2,
board_half_extent=(0,0). Equal coefficients do not imply equal contact dynamics.
No friction values were changed.

Implementation checks include partial reference reset, opposing fragment motion,
free-zone and soft-cost gradients, analytical WM/physical task equivalence,
simultaneous failure precedence, checkpoint layout rejection and one finite WM
optimizer update. Policy input dimension is unchanged; WM/critic gain one
supervised displacement scalar and require new task/state metadata.

Three-object physical checks used 16 reference control steps per backend, with
one world's reset forced at step 8. Other worlds' frames remained bitwise equal
through that partial reset. In each transition, the *measured next physical state*
was passed into the analytic WM task calculation. This tests task equations, not
learned dynamics accuracy. Short traces did not reach the displacement failure
boundary; that boundary and its gradients/settlement are covered by constructed
state tests, not a natural large-slip episode in this probe.

- disect: 169.3 s, maximum absolute task-reward difference 1.46e-08; reward/termination and masked reset checks passed.

- blademaster: 44.5 s, maximum absolute task-reward difference 1.43e-08; reward/termination and masked reset checks passed.


Artifacts: runs/object_slip_20261116/{disect_physical,blademaster_physical}.json
and the corresponding logs. No claim that these changes already reduce sliding
in learned policies; no new trained policy was produced.

Regression after the changes: root **463 passed / 5 skipped**, with
CUTTING_BACKEND_ROOT set and the MPPI Python environment; backend **39 passed**.
Logs: runs/object_slip_20261116/{root_tests,backend_tests}.log.
The DiSECt scene-friction read succeeded in the base Python environment; the
first BladeMaster config read lacked SciPy there and was rerun successfully in
the documented MPPI environment. Both physical probes passed.

Reporting correction: describing the old runs as merely two physics backends
was incomplete. Their actual physical blades and mounting/coupling differed,
as did task shaping and condition distributions. Shared-tool revision
ffede36 and scene audit ba347e9 address the new configurations, not the provenance
of old policies. See SHARED_KNIFE for remaining reset and 2 ms coupling differences.

## 2026-11-16 — Shared per-object start cases and optional bounded pitch

User decision: prepare simulation starts like the intended real-robot sequence,
using each placed object's height and variable clearance. No separate fixed
5 mm debug task. Implemented source: configs/shared/scene_v1.json; full
settings, remaining mismatches and commands are in SHARED_SCENE.

Each object has 3 worlds in this resource-bounded probe. The 9 sampled gaps (mm)
are cucumber [24.448, 10.463, 13.140], half potato [19.864, 26.958, 9.601],
prism [17.545, 13.648, 24.904], identical between backends. Actual heights are
measured after settling; identical clearance does not imply identical absolute
knife height or material shape. The camera/local-height estimator is not implemented.

| Measurement | DiSECt | BladeMaster |
|---|---:|---:|
| Physical steps / worlds | 16 / 9 | 16 / 9 |
| Maximum actual gap error | 1.171e-07 m | 1.936e-07 m |
| Maximum physical/WM reward-equation difference | 1.485e-08 | 1.473e-08 |
| Partial reset: non-reset frame fields identical | yes | yes |
| Wall time (includes setup/settling) | 184.8 s | 55.7 s |

Reports: runs/shared_scene_20261116/disect.json and blademaster.json.
Both preserve requested per-world gaps on partial reset. The reward check supplies
actual next-state fields to the WM task equations; it is not a learned-model
prediction accuracy result. Short reference probes do not measure trained success,
active-pitch contact, randomized-material throughput or real-robot performance.

During implementation, the first new-scene export serialized CPU tensors, which
the packed CUDA path could not consume. Export now preserves CUDA tensors and a
loader guard rejects a mismatched device before kernel launch. A partial BM reset
also exposed an empty-group tool-array assignment shape error in the new alignment
helper; it now skips empty masks and indexes the singleton tool dimension explicitly.
The successful reports above were generated after both fixes.

Optional pitch defaults off. Enabled mode has angle/rate clamps and matching
differentiable physical-action/WM mappings; physical tilted-contact validation is
still pending. Shared reference controls now use the same target speed. Nominal
knife friction is explicitly .2 on both newly prepared backends, with old assets
unchanged. At that revision, per-episode BladeMaster material randomization was outstanding; the following entry records its implementation.

Regression on GPU7: root 487 passed / 5 skipped (241.52 s), backend 41 passed
(5.61 s). Root used the MPPI-capable environment and CUTTING_BACKEND_ROOT,
so available GPU tests were enabled. Changed-document relative links and git
diff whitespace checks passed. No long training started.


## 2026-11-16 — Paired per-episode materials in both backends

Both physical backends now share deterministic per-world episode streams:
log-uniform E .5–2, density .8–1.25, cutting-resistance .5–2; uniform target
speed 1–5 cm/s. Geometry groups remain fixed. Cutting resistance scales DiSECt
spring stiffness and BladeMaster cutting energy respectively. Selected worlds
restore raw state, apply their new coefficients and settle for 1 s. BladeMaster
uses persistent per-world GPU coefficient arrays and a timestep selected at
maximum E / minimum density. Saved streams resume from each world's next episode;
physical episodes restart on trainer resume.

GPU7 checks, three objects x three worlds:
- Both 16-step physical/reward/reset checks passed; inactive worlds unchanged.
- Corresponding initial/reset material, speed and gap cases agree to 5.83e-8.
- Physical versus WM task-equation reward differences: DiSECt 1.46e-8,
  BladeMaster 1.52e-8 (using actual next-state inputs).
- BladeMaster collected 40 steps / 360 transitions; real critic window 32,
  actor horizon 8, BC .25, observation 34. WM, policy and critic updates,
  masked reset, replay/checkpoint and sampler restore all passed; optimizer
  steps 20 / 2 / 40. Forty collection steps contained no naturally ended episode;
  explicit partial resets exercised episode resampling.
- Root regression: 496 passed / 5 skipped, 241.40 s. Robot/DiSECt backend:
  41 passed, 12.51 s. Native BladeMaster: 2165 passed / 8 skipped / 18 subtests passed, 2140.56 s; source c51ef73 (SOURCES).

Reports: runs/paired_material_20261116/{disect,blademaster,paired_conditions}.json
and bm_learning/report.json. Final prepared configs are in assets_final;
the obsolete unused fixed reference-speed field was removed after the probes,
with semantic equivalence checked against their inputs. Commands are in
SHARED_SCENE and RUNNING. The learning probe overlapped regression workers;
its wall time is not an exclusive-device capacity result. No long run started.

## 2026-11-16 — Optional final/path robot position constraints and interface cleanup

The user replaced particle-based sliding constraints with two selectable robot
position modes. Current shared default is final only; path plus final is selected
by common.task.tool_motion.enabled. Both require EE robot-base XY distance <40 mm
from the prescribed goal, extending the old 40 mm single-axis test, together with
the existing height/pitch tolerances and >10 qualifying arrival samples. Samples
remain cumulative. Final success is +100 only without a simultaneous failure.

The optional path mode leaves <=25 mm free, applies a quadratic cost up to
20/horizon per step, and treats >60 mm as failure (-100 once). Final-only mode
does not add this running cost or failure. Force, workspace and board rules
remain in both modes. Fragment separation/opening carries no penalty in the
canonical configuration. Distances are fixed engineering values, not calibrated
hardware limits or size-adaptive thresholds. No policy has been retrained.

Implementation: completion_xy_error is derived from joint FK and the known goal,
not independently predicted by WM. Physical reward/termination, imagined reward/
absorption, arrival counter, critic history and logging share the same rule.
Policy dimensions remain 34 per frame / six outputs (pitch disabled). Task and
state metadata change; historical checkpoint/replay is not silently resumed.

Other issues found and repaired:
- Evaluation resizing had left explicit clearance arrays at their original size;
  evaluation seed did not reseed the new episode-condition sampler. Both now
  regenerate from the shared case protocol.
- Evaluation validates physical knife, board geometry, progress semantics and
  contact discretization, and reports optional motion termination causes.
- Scene capture had still forced old nominal materials, fixed 5 mm starts and
  30 mm/s speed. It now requires the shared contract plus matching prepared
  manifest, preserving all sampled conditions.
- Six superseded depth/knife/motion check JSONs moved to
  configs/archive/scene_checks_20261116. Historical trained-task inputs remain.
- Real deployment docs now distinguish policy I/O from timestamping, wrench
  compensation, coordinate transforms, action mapping, FDCC and hardware stopping.
  This change does not implement or validate a real-robot execution runner.

Boundary bug found by the physical-transition audit: requesting exactly 60 mm
can yield 0.06000000000000005 m after robot-frame subtraction. A strict raw
comparison incorrectly failed 6/9 DiSECt and 2/9 BladeMaster cases in the initial
audit. Comparison now allows eight floating-point epsilons at metre scale.
The 60 mm path boundary remains included; 40 mm final boundary remains excluded.
Translated-coordinate tests cover float32 and float64. This numerical tolerance
is not a new task safety margin.

Evidence directory: runs/interface_motion_review_20261116/ (server only).
tool_assets/ contains current prepared inputs. capture/ contains completed
nine-world settled-state geometry from each backend with actual condition logs;
no policy was loaded. Both use the same paired material/speed/gap distributions.
Older disect.json / blademaster.json in this directory tested the now-retired
particle-RMS interface. tool_disect.json / tool_blademaster.json preceded the
roundoff fix; only the *_final.json reports should be used for current boundaries.

The tool audit holds measured physical kinematics fixed and changes the goal XY
offset to test both reward switches. It is not a naturally occurring 60 mm robot
motion, a trained-policy evaluation or a learned-model accuracy measurement.
Unit tests also cover final arrival, failure precedence, gradients, WM fitting/
BPTT and checkpoint restoration under the new state layout.

Validation completed on GPU7:
- Final physical checks: three objects x three worlds/backend, 16 reference
  steps, masked reset preserves all other worlds exactly. Physical-versus-WM
  reward-equation max error 1.369e-8 (DiSECt), 1.545e-8 (BladeMaster).
- For both backends, the counterfactual 60 mm endpoint fails 0/9 worlds and
  61 mm fails 9/9 only in path mode. Final-only adds no distance running cost.
  At 35/50/60 mm, path costs are .00907029/.05668934/.11111111 per step;
  at 61 mm the reward change is -100.11111111 including failure settlement.
- Root with explicit CUTTING_BACKEND_ROOT: 546 passed, 5 skipped in 241.12 s.
  Skips are legacy SAC gym/TF checkpoint/separate bridge prerequisites, not the
  new position-mode or physical adapter tests. The earlier no-backend-env pass
  was 531/20 and is not the GPU validation result.
- Bundled backend: 41 passed in 5.62 s. Native BladeMaster source is unchanged;
  its prior 2165 passed /8 skipped suite was not repeated.
- Both scene captures completed with nine worlds and the sampled shared
  contract. Relative Markdown links resolve and git diff --check is clean.
- Current artifacts: tool_disect_final.json, tool_blademaster_final.json,
  their raw .transition.pt fixtures, root_gpu_tests.log, backend_tests.log,
  capture/{disect,blademaster}/manifest.json in the evidence directory above.

These checks establish the optional rules and interface consistency. They do
not establish reduced drift after learning or successful real deployment.

## 2026-11-16 — Actual motion demonstration of the two position modes

Evidence: runs/tool_position_demo_20261116/recording_final/comparison.json.
One paired-contract cucumber case (seed 0, first world), with its sampled
material, clearance and target speed. A scripted reference controller requests
a 75 mm lateral excursion over 1.5 s followed by return over 1.5 s, alongside
the commanded downward reference. No policy weights, task thresholds, friction
or material properties were adjusted for this demonstration.

Only the final-only task executes physically. Both task rules are evaluated on
each identical measured transition; the path-rule episode absorbs at its first
termination and never accrues later rewards. The video holds its terminal frame.
It does not simulate hardware braking or compare two newly trained policies.

Results:
- Path mode ends at step 25 (1.25 s) for tool_motion_failure.
- Final-only mode returns to the goal region and succeeds at step 58 (2.90 s).
- Maximum actual planar deviation is about 73.65 mm; final deviation about
  11.7 mm. Maximum wrench-limit utilization is about 0.318, so the path-mode
  difference is not caused by force failure.
- Path running position cost before termination is about 0.625; the terminal
  failure charge is 100. Total returns additionally include ordinary progress/
  speed/load/settlement terms; use the JSON for their exact totals.
- The recorder verifies the replayed final-only reward and termination against
  the actual environment on every transition, including success settlement.

Interpretation of the configured weight: at 35/50/60 mm, each 50 ms control
endpoint costs .00907/.05669/.11111. Ten steps at 50 mm cost .5669; the maximum
position running cost over the full 180-step horizon is 20. Thus it is a mild
preference inside the allowed region, with a hard episode boundary outside
60 mm. It does not force all movement to stay inside 25 mm, and the demonstration
cannot establish trained-policy behavior. No reward weights changed.

A real recording bug was discovered by the exporter's rigid-mount guard.
GeometryRecorder read DiSECt body_X_sc after reset alignment; that cache is
refreshed on the next material integration, while alignment already updates the
authoritative material pose. The initial captured blade was displaced about
2.047 mm relative to the following frames. Recording now reads frame.knife_pose.
No material dynamics, control, task or reward code was changed. A regression
test uses an intentionally unavailable/stale link cache and checks the captured
transformed vertices. The final re-recording passes the exporter's all-frame
rigidity check (error below 1e-8 m), using the same physical knife and mount.

Rendering uses the shared studio exporter/renderer with verified cucumber
appearance, one metal physical blade and no blue proxy/crack highlighting.
The exporter now preserves explicit scripted-capture provenance instead of
mislabeling every episode as a frozen learned-policy rollout.

Regression after the recorder correction: root 547 passed /5 skipped (244.24 s),
bundled backend 41 passed (5.85 s). The focused recorder/position tests pass 41/41.
Video: tool_position_modes.mp4, 1800x1110, 10 fps, 6.9 s, 0.5x playback.
Reviewed initial, terminal-boundary and final frames; ffprobe confirms the stream.
Only the requested finished video/previews are copied to the Mac Downloads folder.

The 60 mm path threshold is a radius in robot-base XY around the prescribed
end-effector goal, not a square, object-centre radius or distance from an object's
surface. It constrains one robot reference point, not the whole blade volume.
The studio exporter contains a display-only cylindrical connector/collar between
the retained URDF grip and the physical blade; its shape is schematic, not a
measured installation part and not collision geometry. The comparison overlay
now states this explicitly. No material/tool geometry changed for this video.

Remaining limits: path checks use 50 ms control endpoints, not a continuous
physical stop; post-training effectiveness is untested. The previously recorded
GPU9 SIGILL inside libcuda remains unresolved (PLAN); this recording does not
establish long-runtime reliability on that host.


## 2026-11-16 — Framework provenance audit and latest-task learner preflight

See [FRAMEWORK_AUDIT_20261116](archive/FRAMEWORK_AUDIT_20261116.md) for upstream/local
choices and remaining training-readiness limits. Removed the invented renderer
shank and collar; corrected stale shared-knife/reset and integration-patch docs.
No learner, reward, material parameter or physical mount change in this audit.
Re-exporting 59 saved frames preserves physical blade/material geometry exactly
and removes exactly two display meshes; rigidity residual 5.674e-9 m. Historical
videos have not been overwritten or claimed to show this correction.

Both current shared-contract backends passed real-data learning/restore probes:
three objects x three worlds; 40 collection steps, 360 transitions; real critic 32,
actor 8, BC .25, 34-dimensional observations. Each ran 20 WM, 2 actor and 40 critic
optimizer steps, with finite metrics/parameters. Masked reset left other worlds
exactly unchanged; replay and episode-sampler restoration passed. Total times:
DiSECt 624.57 s, BladeMaster 323.51 s, including first-update compilation.
Neither 40-step random-policy capture completed an episode; these are not
performance/success measurements or full-scale long-run capacity tests.

Evidence: `runs/framework_audit_20261116/` on GPU7. Learner source: a327260;
native BladeMaster: c51ef733364aa9cbbbceb6ba5b422cabf67ddefc. Reports store config
and script hashes, phases, losses and source status. No new long training started.

Regression after the renderer/documentation correction: root **547 passed,
5 skipped** (242.25 s); bundled robot/DiSECt backend **41 passed** (5.66 s).
Root skips belong to the legacy SAC Gym/TensorFlow/explicit bridge paths, not
the two current learner probes. Native BladeMaster source was unchanged;
its prior full regression is recorded at the same revision in SOURCES.
Logs: `runs/framework_audit_20261116/{root_tests,backend_tests}.log`.
Edited Markdown links resolve and `git diff --check` passes.


## 2026-11-16 — Controller/scene/recording interface recheck

Controller source and UR10e-port limits are now explicit in RUNNING. Current
simulation uses the SliceIt/FZI ROS-free FDCC position-controller port (pinned
cambel source f4359b9), not a running unmodified ROS2 controller. UR10e remains
an unimplemented robot-asset/frames/control/data port; no hardware test occurred.

Found and corrected a remaining visualization mismatch: the all-world overview
still drew the obsolete full URDF knife even though studio movies displayed the
physical blade. Snapshots now record every world's physical blade vertices and
triangles. Overview and studio share robot/handle clipping, hide the cable-envelope
box and add no connector. Old snapshots without per-world blade data are rejected.
DiSECt episode files now explicitly save base/world transforms, as BladeMaster
already did. The existing main-run 24x16 cm display board is shared by both new
rendering paths; original studio recording corners are retained. This changes no
physical support boundary (still an infinite plane).

Actual image inspection also found white particle clouds in the Gaussian overview
path despite the requested palette. The overview now displays low-resolution
sphere instances at recorded material-particle centres/radii with explicit display
colors; studio surfaces/scan appearance are unchanged. This is a display correction,
not a material-law change or new learned result.

Fresh scenes: three objects x three worlds per backend. All 18 physical blades
match recorded EE kinematics: maximum set-distance 3.163e-9 m. Requested and actual
per-world starting clearances differ by less than 1.88e-7 m. Task, force limits,
force-observation revision and episode-condition configurations match between the
two prepared backend inputs. These numbers check software coordinates, not real
hardware accuracy or equality of the two material models.

Evidence under `runs/interface_recheck_20261116/`:
- `final_scenes/`: actual settled snapshots and condition manifests;
- `geometry_check.json`: per-object/world physical blade and start-clearance checks;
- `final_renders/`: six studio stills, two nine-world overviews, `visual_review.jpg`
  and `render_audit.json`; shared display dimensions and no fabricated connector;
- `video_export_check.json`: all 59 frames from the prior demo re-export correctly;
  original blade/material/robot arrays unchanged, board corners numerically equal.
- `root_tests.log`: 549 passed / 5 skipped (242.14 s); `backend_tests.log`: 41 passed
  (5.57 s). The skips remain legacy SAC environment checks. Two new tests cover
  distinct physical tool poses, base/world rotation, no state mutation and invalid
  inputs. Additional final renderer checks executed on actual saved geometry.

The whole-potato appearance map was correctly rejected for the half-potato mesh
(topology mismatch); its failed attempt is retained in `renders/`. Final half-potato
and particle-backend surfaces use disclosed approximate color transfer, not recovered
original UV. Overviews use illustrative colors/LOD. These are validation snapshots,
not evidence that a new long training run has started. Native material laws, reward,
policy/WM training code and formal long-training resource settings were unchanged.


## 2026-11-16 — Current shared-scene long-training readiness and live GPU status

Question: is the current unified task ready for a bounded engineering long run,
and are the historical GPU7/GPU9 runs still active? No long run was started.
No learning, reward, material or renderer code was changed by this audit.

Current-code capacity check on GPU7, code `7db74f2`:
- Three objects x 1024 worlds = 3072, serial object-group execution as generated
  by the shared preparation. Same `pose_depth_v2`, final-only position mode,
  disabled pitch, per-episode material/speed sampling and per-world 5–30 mm gap.
- 40 actual vector steps / 122880 physical transitions; mean step 27.472 s.
  Initialization including per-material settling took 434.925 s.
- Real critic window32, actor horizon8, BC .25, observation34. Effective batches
  WM2048 / actor4096, memory chunks512 / critic8192. No shortened critic window.
- WM20, actor2 and critic40 optimizer steps across initial/restored updates;
  metrics and parameters finite. Non-reset worlds unchanged exactly; episode
  sampler and checkpoint/replay restoration passed, followed by another update.
- Free GPU memory: 5.268 GiB after learning, 6.135 GiB after restored learning;
  peak Torch allocation 17420.37 MiB. Selected-reset test took 27.696 s.
- Final status `passed`, 1758.954 s total (29.32 minutes). The random-policy
  40-step capture ended no episodes; this is not a success-rate evaluation.

The first probe had an undersized 1500-second allowance. Once measured timing
showed initialization plus 40 collection steps alone needed about 1534 seconds,
we interrupted it explicitly after 993.654 seconds (KeyboardInterrupt), then
repeated the unchanged probe/config with a 2400-second ceiling. It was not an
OOM or numerical failure. The aborted run is separate evidence, not counted as
passing. No process timer or simulation settings were altered during either run.

Evidence: runs/long_readiness_20261116/{contract.json,prepared/,
disect3072/report.json,disect3072_complete/report.json,capacity_complete.log,
status.json,proposed_launch.json}. The launch draft is not executed: fresh
networks, 24h or 1280000 new transitions (first bound), 500 WM-only rounds,
30720-transition warmup, 640 transitions/update, online W&B and paired replay
checkpoints every32 vector steps. The current main configuration is eligible for
bounded engineering training; passing this check does not guarantee convergence.

Live check at 18:48:35 CET: GPU7 RTX4090 40 MiB, GPU9 RTX4090 39 MiB,
A6000 GPU0 43 MiB and GPU1 17 MiB; all at 0% utilization and no compute processes.
Availability is a snapshot, not a reservation. A6000's development checkout is
still 2b73009 and needs an isolated current-code/assets deployment before use.
GPU9 driver remains595.80; its historical SIGILL cause remains unresolved.

Historical job status reverified from local logs and cloud API:
- GPU7: budget_complete, 24h, 1062912 transitions, policy1109. Saved frozen
  evaluation remains 0/48 prescribed-pose successes, all timeouts, return−10.97.
- GPU9: native SIGILL(-4), last logged policy998 / 968960 transitions (~19.74h).
  Saved policy866 evaluation remains48/48 nominal-replica pose successes,
  return106.94. This is not an evaluation of the unsaved policy998.
- Both old W&B runs are remotely readable and show finished. GPU9's cloud badge
  does not establish normal exit or completed budget. These two old evaluations
  differ in material distribution, tool/task settings and are not a matched
  algorithm or backend comparison. Pose success is not a separation guarantee.

BladeMaster's current nine-world learner/reset/restore check remains valid, but
this audit did not certify its new randomized/shared-tool configuration at large
scale on A6000/GPU9. Prefer a current isolated A6000 deployment for that next
capacity check rather than treating the old GPU9 launch as already repaired.
Regression remains the unchanged-code result at7db74f2: root549 passed/5 skipped,
bundled backend41 passed. Only docs changed after this probe; no redundant suite
rerun. Updated PLAN and server-local MACHINES distinguish completed old jobs
from the current readiness check.


## 2026-11-16 — Long-run entry audit: pending updates and shutdown status

Follow-up to the current-scene capacity check, at parent a102afe. Scope: active
`train_gpu.py` launch/config path, physical-task/WM regression, checkpoint/replay
resume and shutdown status. No new long training or task/network/reward change.

Two confirmed control-flow defects were corrected:

1. A vector step can earn multiple learning rounds, but the checkpoint saved
   only collection credits, not the remaining unstarted rounds. For example,
   3072 transitions with a640-transition cadence earn four rounds plus residual
   credit; a milestone after the first round lost the other three on resume.
   `UpdateQueue` now checkpoints that tail and its LR clocks; the active GPU
   runner drains it before collecting further physical data. Completed-round
   milestone callbacks run after acknowledgement. Budget checks before beginning
   a round preserve all pending work if no round can start.
2. Environment shutdown previously happened in the outer guard after tracker
   finalization. A close error could therefore accompany a completed progress
   record/cloud finish. Shutdown is now attempted before tracker finalization;
   errors enter failed status, and the setup guard prevents a duplicate attempt.
   This addresses Python cleanup reporting, not GPU9's native SIGILL.

Limits are explicit: an interrupted round may already contain optimizer steps.
Those saved parameters/optimizer states are retained; its unfinished remainder
is recorded and the whole round is not silently replayed. Only unstarted rounds
are recovered. Old checkpoints without the queue cannot reconstruct omitted
work and warn; intentional empty-replay resume records discarded queued work
and follows the existing refill warmup. Physical episodes still restart at
resume. This is not a claim of exact arbitrary-mid-minibatch continuation.

Verification:
- Targeted tests:35 passed, including actual runner control flow with CPU doubles
  for physics/learners. A saved WM milestone with two pending rounds resumes
  both before any new physical step; its completed update counts match the
  uninterrupted runner. A forced close failure reaches progress and tracker.
- Full root regression, including GPU asset paths and optional MPPI environment:
  **562 passed,5 skipped**,242.10 s. The five skips are legacy SAC dependency /
  explicit bridge/checkpoint opt-ins, not current-task failures.
- Bundled robot/DiSECt backend: **41 passed**,5.57 s. Native BladeMaster source
  was unchanged; this is not a rerun of its standalone native suite.
- The first full-root invocation stopped during collection: the MPPI Python did
  not have the bundled backend on its import path. Root conftest now supplies
  that path independently of test collection order; RUNNING shows the explicit
  PYTHONPATH too. Isolated pitch-test collection with PYTHONPATH=. then collected
  all9 cases. The initial collection error is preserved, not counted as passing.

Evidence: runs/runner_audit_20261116/{root_tests.log,
root_tests_with_backend_path.log,backend_tests.log,isolated_collection.log}.
Tests added in test_update_queue.py and test_gpu_runner_resume.py cover queued
round recovery, partial-round disclosure, budget-before-start, empty replay,
legacy/invalid queue records and actual runner lifecycle. No physical simulation
or learner equation changed, so the preceding3072-world capacity result remains
separate evidence; it was not silently rerun or relabeled as this test.

No other new blocking issue was found on the reviewed main launch path. GPU9's
native fault and the pending large-scale current-config BladeMaster deployment
remain open; neither is declared fixed by these changes. Main fresh-run proposal
and online-W&B requirement remain unchanged.


## 2026-11-16 — Shared-task deployment and offline wrench panels

User authorization: inspect docs/visualization, synchronize code, then launch
GPU7 main and A6000 BladeMaster training after checks. The old no-launch plan was
archived; PLAN and TRAINING_SHARED_20261116 now separate current task/budgets
from historical GPU7/GPU9 runs. DATA's stale0.75s current-settling and old native
source entries were corrected. SAC remains the historical single-world bridge;
current shared multi-object SAC and MPPI physical evaluation are not completed.
MPPI package/import test success is not a new-task rollout result.

The two isolated training deployments are at5ec9f85, with native BladeMaster
c51ef73. Robot MJB, source task/reset, FDCC parameters and all three intact assets
match SHA256 on both hosts. Task/reward, six-axis observation, force limits and
episode distributions match; first170 worlds per object have identical sampled
clearances. Different vector counts/hardware and native fracture laws remain
explicit, so this is not a claim of identical execution or matched compute.
Evidence:runs/shared_long_20261116/deployment_audit.json on GPU7 development.

A stale video-caption sentence still described the removed display rod. It was
removed without changing geometry. The new compose_wrench_video helper appends
two unit-separated sensor panels below an existing shared-studio movie. It checks
case/episode identity, observation continuity, reset counters, timing and encoded
frame counts. Signals are signed filtered sensor-frame values reconstructed
from full-wrench normalized observations and recorded per-axis limits.
They are20Hz endpoints, not500Hz peaks or material resultants. Source video is
preserved in the top panel, with ordinary H264 reencoding differences.

Validation used historical GPU7/GPU9 cucumber movies only, not new training
results:100/27 encoded frames preserved; upper-panel first-frame pixel MAE
1.197/1.225 on a0–255 scale. Time/unit/invalid-trace tests pass. The initial
composition invocation used the lean training Python and failed because Pillow
was absent; rerunning in the existing optional MPPI/render Python passed for
both recordings. No packages were added to the live training environment.
Panel source/data/provenance:runs/wrench_panel_validation_20261116.
The historical full movies are retained as validation artifacts, not presented
as current shared-knife training videos.

Full root regression after the visualization changes:569 passed,5 skipped,
294.78s; bundled backend41 passed,6.23s. Tests ran on GPU7 while the already
validated frozen-code training collected/fitted data; those overlapping wall
times are not exclusive-device throughput measurements. Only offline rendering
code and docs changed after the frozen launch. Native solver/learner code did
not change. Current launch state/URLs and A6000 capacity result are maintained in
TRAINING_SHARED_20261116 and the server runtime records.


## 2026-11-16 — Shared long runs launched and cloud verified

A6000 GPU1's510-world full capacity probe passed in2044.1s. It collected
20400 physical transitions, used critic32/actor8 and effective batches2048/4096,
then exercised WM/policy/critic updates, masked reset, physical snapshot and
replay restoration.20 WM,2 actor and40 critic optimizer steps were executed
including restored updates. All results finite; untouched worlds exact;
snapshot state unchanged; sampler restored. Memory headroom7.912GiB after
learning and7.904GiB after restore. This40-step probe ended no episodes and
does not itself validate physical task success.

The guarded launcher then started the510-world BladeMaster long run on A6000
GPU1. GPU7's3072-world DiSECt-backed run is also active. Both use clean5ec9f85,
same algorithm/shared-task distribution, online W&B and independently bounded
24h/1280000-transition budgets. W&B API verification at20:12 CET returned
running state and nonempty physical-transition history for scr1v5np (GPU7)
and pvord4sx (A6000). Exact URLs/PIDs/paths are in TRAINING_SHARED_20261116.
A6000's first510 transitions were recorded after about563s; GPU7 was past80k
transitions in WM pretraining. Neither policy had updated at this launch check.
No training-performance conclusion is made.

GPU7's actual3072-world first-step snapshot was rendered using the common
physical geometry entry point; grid offsets are display only. Correct VTK and
ROS asset-root invocation is now documented. Final documentation audit corrected
historical/current status labels and ten relative links in the moved plan archive;
all scanned local Markdown targets now exist. No active training code was edited.

## 2026-11-16 — Current-task SAC and two MPC adapters

See [CURRENT_BASELINES](CURRENT_BASELINES.md) for the SAC source search, explicit
SB3 fallback, commands and limitations. GPU9 completed identical three-object
first-episode cases: smoke-trained SAC1/3 pose success, frozen early WM+MPPI3/3,
nonlearned FDCC+MPPI2/3; all other outcomes were timeout, and all9 episodes
retained material connections. These are interface checks, not a ranking of
algorithms or proof of material separation. Root outputs remain on GPU9 at
`Cutting_DEV_baselines_20261116/runs/baseline_validation_20261116/`.

SAC resumed264total physical transitions/240updates including its restored
replay. Terminal observations precede masked reset, timeout does not bootstrap,
and update credits count transitions instead of vector steps. Periodic saves
were moved out of SB3's pre-insertion callback to an update boundary. The new
8D-goal MPPI test found and fixed a pose/goal slicing error in optional terminal
critic evaluation; old7D goals retain behavior. Running long jobs were untouched.

A wall-budget regression also fixed the new SAC runner's stop boundary: changing
SB3's schedule horizon alone did not end its local loop. The corrected check
runs before the next rollout; physical expired-budget restore retained the
exact240transitions/216updates/replay240 without a dropped transition.

Final verification: root577passed/3optional original-TF checks skipped;
bundled backend41passed on GPU9. New SAC and planner tests ran, not skipped.

Checkpoint payload hashes now reject mixed model/replay generations after an
interrupted save; temporary payload writes precede the final metadata update.
The mixed-generation rejection test is included. This does not promise rollback.


## 2026-11-16 — GPU9 current-task SAC capacity and launch
The3072-world SAC capacity/update preflight completed3072 physical transitions,
3072 optimizer updates and checkpoint/replay saving, with finite parameters and
normal cleanup. Total475.37s including439.94s initialization; this is not a
measurement of complete-episode or sustained training throughput.
The authorized fresh SAC run then started from5be7d90 on GPU9, same current
three-object distribution and1,280,000-transition/24h cap as documented in
[TRAINING_SAC_20261116](TRAINING_SAC_20261116.md). Online W&B i00yr0fg authenticated
and its first3072-transition history row was independently read through the API.
Local6144 transitions and zero updates are expected during30720 warmup.
GPU7/A6000 running checkouts were not changed. No new SAC success-rate claim.

## 2026-11-17 — Prescribed-cut depth perception prototype

User requested route 1 first, retaining the current prescribed cut. Added a
calibrated depth-to-start/goal module and shared-scene camera capture. Camera:
front above board, offset [0, 0.32, 0.55] m in base coordinates, looking down
approximately 30 degrees from vertical; 640x480, focal length700 px.

Recorded settled cucumber / half-potato / prism surfaces gave visible-top errors
approximately0.34 /0.14 /0.03 mm. These are ideal-depth, single-frame measurements,
not real-camera accuracy. Known robot projections mask self-occlusion; no food
mesh or material labels enter the estimator. All three produced an 8D base-frame
goal compatible with frozen checkpoint128 and its actual mounted blade.

No policy inference or physical cutting episode was executed from these plans.
No RL retraining, changed reward, new cut plane or modified running job.
The end-to-end physical handoff remains unverified. See
[VISION_ROUTE1](VISION_ROUTE1.md) for inputs, commands and limitations.
Evidence is on GPU7 under runs/vision_route1_20261117; no Mac project copy.
GPU9 regression: root589 passed/3 skipped; bundled backend41 passed.

## 2026-11-17 — A6000 throughput and WM-only schedule check

Read-only inspection of the unchanged shared-run checkout (5ec9f85), GPU1,
W&B pvord4sx. At iteration333:169,830 physical transitions,217 completed WM
rounds,0 policy updates. This agrees exactly with
floor((169830-30720)/640)=217. A WM round contains10 optimizer minibatches.
The launch requests500 WM-only rounds, not500 optimizer steps.

Over the most recent30 vector steps:50.44 s/step, comprising28.67 s physics,
10.93 s reset/settling and10.83 s WM fitting. Cumulative physics and reset
accounted for approximately84% of elapsed time. The512-sized network microbatch
is unrelated to the510 physical worlds. Material randomization resets selected
worlds, advances1 s of settling in500 exchanges, and reads each exchange back to
CPU for contact/finite checks; this is measurable overhead and a possible
optimization target, not evidence of skipped learning or deadlock.

At this rate joint learning starts around vector iteration688 (roughly5 hours
after the inspection), subject to subsequent throughput. The24h wall cap remains;
1.28M transitions are an alternative stop condition, not a promise within24h.
No schedule, physical timestep, reset conditions or running job was changed.


## 2026-11-17 — Depth-selected plane and frozen physical feedback trial

The user clarified that perception must choose the cut plane, rather than check
a supplied plane. Added visible-major-axis midpoint selection with no preset XY
input. Translated and rotated depth fixtures verify that selection moves with
the cloud; near-round axes reject. Policy goal quaternion sign is anchored to
the source policy representation, not merely checked as an equivalent rotation.

Seven independently scanned, textured OmniObject3D assets were rendered at native
metric scale: cucumber001/002/003, potato001/002/003 and apple002. Three cucumbers
and two elongated potato views passed; potato003 and apple002 rejected as
ambiguous. These are static ideal-depth geometry examples, not seven physical
cutting tests or calibrated material models. GPU7 evidence:
runs/vision_variants_20261117 (including per-object visual plans and gallery).

A single native BladeMaster cucumber trial used frozen main-run checkpoint128,
policy66. The A6000 native actor had zero updates, so this is explicitly a
cross-backend inference demonstration. The selected plane shifted the EE goal
by X -0.355 mm / Y +8.986 mm relative to the old preset, with yaw about0.46 deg.
Only calibrated depth selected its location. Pre-cut IK/reset changed the robot
pose while verifying unchanged material positions; no physical approach motion
was claimed. Thereafter actual force/motion feedback drove180 policy steps.

Final trial06: timeout, return -6.754, no force-limit or workspace failure;
peak material force2.720 N, peak limit utilization0.0850, depth progress0.95869.
Blade-board gap ended at2.476 mm, EE target height error2.491 mm, XY error14.103 mm.
Material component diagnostic remained1. The chain executes, but this policy did
not complete the selected task. This is neither native-policy performance nor
proof of arbitrary cut generalization. No counterfactual separates visual goal
change from cross-backend transfer; no policy/reward/material parameter was tuned
to make the trial succeed.

GPU9 evidence: runs/vision_closedloop_20261117/trial_06 contains all raw transitions,
per-step geometry, visual plan, conditions, source/checkpoint provenance and
handoff_audit.json. Shared studio rendering uses the actual recorded blade and
original handle; no adapter rod or blue blade is added. Camera display coloring
was corrected on the source polydata before its normal filter; depth and plan
were independently verified unchanged (max depth difference0).

Earlier attempts are retained rather than counted as episodes: trial01 startup
exit139, cause unconfirmed; trial02 overly strict raw quaternion IK comparison;
trial03/04 renderer dependency / snapshot identity failures. Trial05 ran but used
the opposite raw goal quaternion sign from the source policy, so its timeout is
an interface diagnostic. Trial06 corrected the input convention; it still timed
out. Long training checkouts and algorithms were not changed. All evaluation and
regression time shared GPU9 with SAC; no exclusive throughput claim.

Final regression on GPU9: root596 passed/3 skipped; backend41 passed.
19 visual geometry/interface tests are included.


## 2026-11-17 — Documentation consolidation and current cut-plane audit

This change touches documentation and Git organization only. No Python source,
task/config JSON, asset, model weight or running job is changed. README/PLAN/
PAPER now separate current implementation, method2 fixed-heading center-cut
pilot, and historical evidence. METHOD explicitly labels its old24D/7D source
mapping as historical. Three dated review/launch files move to archive with
links rebased. DATA records retired branches and preserved heads.

The GPU7 active prepared DiSECt assets reconstruct interfaces at material x=0.
Fractions along their initial x bounds: cucumber49.8683%, half potato49.7168%,
prism50.0000%. SHARED_SCENE records file hashes, frames and midpoint offsets.
These are approximately centered geometric cuts, not equal-volume claims.

Method2 is scoped, not trained. Visual physical execution currently supports
only the frozen WM+BPTT policy; SAC, WM+MPPI and analytic MPC visual adapters
remain to be implemented.

Validation on GPU7 RTX4090 with the prepared environment:
root588 passed/6 skipped (298.70 s), bundled backend41 passed (6.70 s);
relative Markdown links and git diff whitespace checks pass.
Root used the optional MPPI Python; backend used .venv. Existing deprecation
warnings remain. The first invocation omitted runs/sac_env.sh and failed on
TORCH_EXTENSIONS_DIR; RUNNING now includes that prerequisite.
These test counts describe this environment, not certification of untested
hardware deployment or all historical physics paths.


## 2026-11-17 — Fixed-heading visual learning and two physical handoffs

Authorized scope: train a midpoint predictor on three existing geometries with
fixed headings and small XY translations, then execute with the frozen policy.
This did not retrain the WM, actor or critic, or alter either active long run.
Source data: recorded settled DiSECt studio surfaces of cucumber003, half potato,
prism. Static rendering translations are not900 independently simulated settles.

900 cases (300 per shape), split600/150/150 by unique placement ID, +-15mm XY.
Ideal640x480 depth, fixed calibrated synthetic camera; no sensor noise or new
shapes. Visible cloud centroid normalization excludes absolute XY and full mesh
center from input. Offline labels are full surface-bounds midpoints along +base Y.
Checkpoint selection used validation only; test read once. CPU120 epochs took
42.58s; best epoch117, validation MAE0.2199mm. Online W&B abn7qktr is finished;
remote history verifies epoch120. Dataset SHA256:
8dac0c38e933dc29a4ef8e7575045c1ef63872d7e05d54fadc48c972bfb62da7.

| Held-out150 placement cases | Midpoint MAE (mm) | P95 (mm) |
|---|---:|---:|
| Learned local-cloud residual |0.2099|0.5653|
| Fixed-heading visible2-98% extent midpoint |5.4995|9.1961|
| Constant world coordinate fitted on training labels |7.6851|15.2314|
| Visible point-cloud centroid |15.1286|22.3001|

The network can exploit the known shapes/view and occlusion bias. This is not
new-object generalization, real camera accuracy or evidence of novel geometry
learning. The geometric fixed-heading variant is compared here; original route1
PCA-heading selection remains available separately.

Physical evaluation on GPU9 used the same explicitly cross-backend transferred
policy66 (source checkpoint128, SHA256
3bf6d7017bcfe31b5e38da68d61372156c2c85a77e8328b84989e126d138a8fd).
One native-BladeMaster nominal cucumber episode per selector,3cm/s,20mm
clearance,180 steps. Policy identity and recorded material/command conditions
match. Native numerical initialization is not claimed bit-identical.
One-shot depth generates the goal; motion and wrench provide execution feedback.
Start uses simulation IK/reset, not a validated robot approach trajectory.

| Quantity | Geometric fixed heading | Learned fixed heading |
|---|---:|---:|
| Absolute midpoint error against initial full rendered surface (mm) |8.8515|0.6880|
| Final target-height error (mm) |2.4964|2.8897|
| Final blade-board gap (mm) |2.4798|2.7957|
| Height-qualified samples (<2mm) |0|0|
| Peak material-force norm (N) |2.7469|8.6046|
| Maximum six-axis limit utilization |0.0854|0.2262|
| Return |−6.7568|−6.9893|
| Ending |timeout180|timeout180|
| Connected material components |1|1|

No force/workspace failure; neither pose completion nor complete material
separation occurred. Learned localization improved in this single physical view,
but control did not complete. The direct missing completion condition is target
height; this alone does not identify the learning/transfer failure's cause.
Do not turn these two cases into success-rate or force-superiority claims.

Evidence: GPU7 runs/visual_center_20261117/{dataset,learning};
GPU9 runs/visual_center_closedloop_20261117/{fixed-midpoint,learned-midpoint},
comparison.json and queue.json. Both complete transitions and geometry recorded.
Same shared studio renderer, single physical blade, original gripper, no invented
connector or blue overlay. Visual results are not physical-separation labels.

Validation on GPU7: targeted visual tests26 passed; root595 passed/6 skipped,
backend41 passed. Root skips cover unavailable SB3/gym and explicit historical
TensorFlow/physical bridge cases; no newly added visual test is skipped.
Data IDs, placement splits, relative/base label consistency and checksum verified.
New scripts compile; complete root run used the prepared CUDA/MPPI environment.

## 2026-11-17 — Official DP3 point encoder vs local midpoint network

User-authorized replacement trial; upstream MIT PointNetEncoderXYZ pinned at
47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97, class AST verified identical. Only its
colored diagnostic printing is suppressed. Official XYZ/64D/LayerNorm settings;
our bounded midpoint head, not the DP3 diffusion policy. All weights trained
from scratch. Original input normalization/target/goal conversion unchanged.

Paired600/150/150 data,120epochs,Adam.001,batch32. Independent shuffle generator
ensures the same data order across architectures for each seed. Local27073
parameters; DP363297. Same epochs do not imply equal capacity or compute.
Validation selects each checkpoint; test was already seen in the previous
pilot, so this is a reused development benchmark, not a new final evaluation.

| Seed | Local MAE mm | DP3 MAE mm |
|---|---:|---:|
| 20261117 | 0.2067 | 0.1519 |
| 20261118 | 0.1945 | 0.1361 |
| 20261119 | 0.2027 | 0.1230 |

Mean MAE0.2013 ->0.1370mm (about32% lower); pooled95th percentile0.5014 ->
0.3533mm; maximum0.9083 ->0.4869mm. Each paired seed favors DP3 on MAE.
Case bootstrap of the three-fit mean difference DP3-local: -0.0643mm,
95%[-0.0872,-0.0433]. This conditions on these fits, does not include a
population of training seeds, unseen objects or actual camera errors.
Use DP3 as the new visual encoder; retain local_v1 for historical checkpoints
and comparison. No test-driven tuning or threshold changes were performed.

Frozen physical handoff used preselected seed20261117 (not lowest test error),
same policy66 SHA3bf6d7017bcfe31b5e38da68d61372156c2c85a77e8328b84989e126d138a8fd,
nominal native-BladeMaster cucumber and explicit cross-backend inference.
DP3 plane error -0.8453mm (previous local -0.6880mm),180-step timeout,
return -7.0070, peak material force10.8438N, peak limit ratio0.2513, one material
component. Final EE height error2.9135mm exceeds2mm tolerance; zero
height-qualified samples. No force/workspace failure. One execution per version
cannot establish a force difference; smaller dataset error did not improve
task completion, and did not improve this particular physical localization.
This uses old policy66 deliberately; it is unrelated to the newer policy527
video below. No retraining or edits to either running controller.

Code validation: root598 passed/6 skipped, backend41 passed; center10 passed.
Optional SAC/Gym/historical TF skips remain. GPU7 training/evaluation/tests
overlapped: timing is not exclusive-machine throughput.

## 2026-11-17 — Requested GPU7 policy527 studio videos

Fixed checkpoint224 (policy527,WM1027), SHA
d542414c9f783ed9055b08bfd391f84ca1e1b9ada70af8ebf704fa861a7a4549.
One first episode per shape, seed20261117, generated from the actual training
conditions, unchanged task/reward, deterministic frozen policy, no updates.
All three reached pose-task success in16 steps, none hit force/workspace limits.
Material complete separation was false for all three.

| Object | Return | Peak material force N | Peak limit ratio |
|---|---:|---:|---:|
| Cucumber |98.34|7.90|0.0958|
| Half potato |99.96|15.85|0.5389|
| Prism |97.63|17.99|0.3936|

These three selected-by-seed cases are illustrations, not the last100 training
episodes or a reliable success-rate estimate. Initial attempted evaluation
combined the newer evaluator with frozen source lacking shared_baselines and
failed before scoring; failed output retained. Successful rerun used the
current validated interface, without changing the active training checkout.

Shared studio renderer, verified cucumber scan map, potato appearance projection,
same existing cameras/palette, physical blade/original handle only, no blue proxy
or invented connector. Full clips0.5x,26frames10fps, plus synchronized signed
sensor force/torque panels (20Hz filtered endpoints, not500Hz limit peaks).
FFmpeg full decode and image inspection passed; downloaded MP4 SHA256 matches
server. Local destination: Downloads/cutting_policy527_20261117.

DP3 comparison W&B: all six remote runs verified finished at120epochs:
- [0cboc47l](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/Cutting_DEV/runs/0cboc47l)
- [usx01450](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/Cutting_DEV/runs/usx01450)
- [bn7bjlv8](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/Cutting_DEV/runs/bn7bjlv8)
- [57gbfk7p](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/Cutting_DEV/runs/57gbfk7p)
- [8duewbr9](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/Cutting_DEV/runs/8duewbr9)
- [cj4mrcsu](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/Cutting_DEV/runs/cj4mrcsu)

## 2026-11-17 — DP3 goals to RL, paired WM+MPPI and nonlearned MPC

User requested videos for all three. Kept native-BladeMaster nominal cucumber,
force limits/task/reward/horizon and DP3 seed20261117 weights unchanged.
The RL episode is the existing old policy66 execution (not policy527).
WM checkpoint128 contains exactly its actor tensors; WM566, source5ec9f85.
Both learned controllers are explicit DiSECt-to-BladeMaster inference transfers.
The nonlearned FDCC approximation holds measured wrench over its horizon;
it has no learned model, critic, material parameters or fracture predictor.
Both planners use pinned pytorch_mppi, horizon8,samples512,one refinement,
gamma1,no terminal critic; analytic5substeps. No retuning after results.

The three condition dictionaries and DP3 model hashes match. Each selector
observes its separately settled initial scene; the largest target spread is
0.02435mm in Y (X0.00115mm,Z0.000009mm). Therefore these are matched nominal
cases, not bit-identical physical resets or a common cached target.

| Controller | Outcome | Steps | Return | Peak material N | Max force-limit ratio | Material components |
|---|---|---:|---:|---:|---:|---:|
| DP3 + old RL66 | timeout |180|-7.007|10.844|0.2513|1|
| DP3 + paired WM566 + MPPI | timeout |180|-6.329|10.622|0.2543|1|
| DP3 + nonlearned FDCC MPC | pose success |105|101.452|10.808|0.2749|2|

No force/workspace failure. Final target-height errors2.913/1.771/1.527mm.
WM+MPPI first entered the2mm height band atstep176: only5qualifying samples
before timeout, insufficient for the task's >10 arrival counter. Analytic MPC
entered at95 and succeeded at105 (11samples). Do not infer that extending
the horizon necessarily fixes WM, or that analytic control is generally better.
Two material components can include fragments; no new intended-slice integrity
claim. One trial per method, reused RL run, no success-rate significance.

Planning time WM mean0.4691s,p950.6281s; analytic mean0.4597s,p950.5052s.
These exceed50ms control period and were measured alongside CPU rendering.
They are offline receding-horizon demonstrations, not real-time execution.

Execution found and fixed target_speed shape N vs N-by1 before any scored
planner step. First failed runs preserved as *.failed_speed_shape. A separate
render-watcher accidentally pre-created the analytic output directory; that
attempt did not start physics and is preserved separately. Corrected watcher
creates its staging directory only after evaluation. Restart changed no task
or planner settings. WM and analytic completed with frozen-weight checks.

Verification: root605passed/6skipped, backend41passed; GPU9 optional planner/SAC
plus transfer-contract tests14passed; visual/transfer targeted36passed.
Active long training source/checkpoints were unchanged.

Recording-label correction after execution: planner snapshot policy_updates66
was inherited from the reference checkpoint, not a trained MPC actor.
New captures put null there and record reference_policy_updates66 separately.
The scored trajectories remain unchanged; video titles identify actual methods.

Approved shared studio render uses recorded blade/original handle, scan-based
color transfer, no blue proxy/cut-face highlighting/fabricated connector.
Delivery movies start with the actual pre-cut camera image and DP3 selection,
then the full1x episode plus synchronized20Hz filtered force/torque endpoint
charts. Those charts are not the500Hz peak-limit measurement. Images were
inspected for all three controllers. Delivery filenames are dp3_rl.mp4,
dp3_wm_mppi.mp4 and dp3_analytic_mpc.mp4 under the DATA paths.

## 2026-11-17 — Current-entry cleanup and evaluation persistence repair

The repository audit found stale current-facing instructions, not a new
training-method change:
- PLAN still said A6000 had no trained actor and repeated completed visual
  budgets as future work. It now points to dated evidence and current actions.
- RUNNING still described the old blue contact-blade overlay, an obsolete
  development branch and old pose-v1 evaluation configs. The current path is
  frozen pose evaluation -> shared physical-scene export -> studio rendering
  -> optional synchronized sensor-wrench panels.
- The script index listed TensorFlow single-world SAC as the active unified
  learner and contradicted the now-implemented visual MPC dispatcher.
  It now identifies train_sac_multi.py and all three visual controller choices.
- Every configs/free_cut/pose_*.json still declares free_cut_pose_v1; their
  README incorrectly called them current. They are retained as historical
  fixtures. Current pose_depth_v2 inputs are generated from the shared contract.
- DP3's existing MIT license and pinned provenance are now indexed in NOTICE.

A real lifecycle defect was fixed in evaluate_visual_cut.py: planner/trajectory/
summary persistence errors could skip env.close(). Cleanup now always runs,
and failed persistence updates progress when writing remains possible.
The pose evaluator already closed unconditionally; it now also marks result
persistence failures instead of leaving completed status behind.
Fault-injection tests cover all three save stages, successful saving and an
unwritable error log. These do not establish new physical cutting performance.

No task/reward threshold, model architecture, physical scene, active training
checkout, replay or weight was changed. Historical code imported by shared
exporters and tests remains, explicitly labeled; it is not deleted merely
because it is old. Current-facing local links and first-party Python syntax
were checked. The previous long PLAN and script instructions remain at
Git revision e7b9e69 rather than another duplicated archive file.

Runtime snapshot during this audit (2026-11-17, around15:00 CET): GPU7
policy784,854016 transitions; A6000 policy113,423300 transitions; both joint.
GPU9 SAC progress records completed at1281024 transitions / transition_budget,
23956 seconds. These are status observations, not frozen evaluation scores.
Validation: initial GPU7 suite609 passed /6 skipped /1 failed; the failing
parallel-group initialization raised CUDA OOM while the long trainer occupied
about22GiB. An isolated retry reproduced the resource failure. Tests were not
relaxed and the trainer was not stopped. On idle GPU9 (RTX4090), the same
changed code passed the complete root suite619 passed /3 skipped in155.5s,
including the parallel reset test, and the backend suite41 passed in3.3s.
GPU9's optional baseline environment includes SB3/Gym coverage unavailable in
the GPU7 optional-MPPI environment, so collection/pass counts differ.
First-party syntax:157 files; current-facing local-link check:30 documents,
zero broken paths. Existing upstream AST deprecation warnings remain; no
physics, learning efficacy or real-robot calibration claim follows from tests.
