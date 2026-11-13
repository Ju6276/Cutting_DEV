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
| Default scalar feedback is a contact-magnitude proxy; optional net feedback now includes the contact moment, corrected frame/timing and final-sample checks | 2026-11-12 net-wrench repair |
| Early scalar/high-transition-velocity sawing measurements had negligible effect; later net/0.01 m/s measurements reduce vertical force, with a smaller net-force reduction on the prism | 2026-11-12 press/saw reruns |
| Contact search, interpolation and force scattering now agree; four rerun mean net forces changed by <0.6%, not a demonstrated large performance gain | 2026-11-12 contact-coordinate correction |

**Superseded:** the 2026-11-11 "force needed to cut the held scanned objects" table (hold artefact
for cucumber_003) and every success/force-limit rate of the S3 runs (defective task definition).
Fixed-prism numbers (8/8 goal-pose completions, archive) retain that definition; they do not independently establish separation or wrist-force fidelity.

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

Review handoff: [CONTACT_FIX_REVIEW](CONTACT_FIX_REVIEW.md), including derivation,
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
Parent e529161; see [FREE_CUT_REVIEW](FREE_CUT_REVIEW.md) and the versioned contract
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

Full implementation/evidence handoff: [FREE_CUT_REPAIR](FREE_CUT_REPAIR.md).
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
- Travel: in every case the knife edge reaches 1.3-4.0 mm below the object's lowest particle, so the
  springs left intact on the free prism are not explained by insufficient knife travel.
- Lag (from knife motion, not from cut time): during a 3 cm/s press the edge trails the command by
  about 4.5 mm (about 0.15 s) and reaches the commanded end 0.4-0.5 s after it; on stiffer material it
  can stay short of it. Not yet measured at 1 cm/s, the speed that sets the time budget.
