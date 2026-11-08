# Current plan

The single current task list. Completed experiments, numbers and failed results
are in [RESULTS](RESULTS.md); task and metric definitions in
[BENCHMARK](BENCHMARK.md); implementation and named variants in
[METHOD](METHOD.md); literature in [NOVELTY](NOVELTY.md).

## Research question and scope

Robot-methods paper (RA-L/ICRA/IROS). In force-limited compliant cutting, how do
a world model's errors in predicting cutting force and failure events affect the
policy learned by BPTT through it, and how can policy improvement inside the
model carry over to physical execution — faster cuts without more force-limit
risk?

- Scope now: the single-cut task. Multi-material/shape evaluation follows as an
  experimental condition (not a contribution; SliceIt already trains on several
  calibrated foods).
- The method follows from the evidence of the next experiment; no new network or
  loss family is fixed in advance.
- Own evidence at its scope: in the fixed task, from the same start policy
  (policy62) and one seed, adding the actor objective through the WM reduced
  completion from 20–21 to 17–19 steps while most checkpoints raised the peak
  material force (RESULTS 2026-11-08). This is a behaviour change in this setting.
  Faster execution, better sample efficiency and accurate WM prediction are
  separate claims, each needing its own test. Seeds come after the method is
  fixed.
- Not pursued now: loss reweighting/GradNorm (B), terminal-feature rescaling (C),
  multi-cut tasks, additional algorithms, real-robot extension.

## Decisions in force

- **Benchmark bottom stays free** (`public-prism-ur5e-v1`); the fixed-bottom
  contract is a diagnostic only (RESULTS 2026-11-08: mean material force
  +16–25% with a static bottom).
- **Names.** Methods are named as in BENCHMARK ("Method names") and never pooled:
  SliceIt SAC in Gazebo; SAC in the unified environment (step 85000 reported);
  a future parallel SAC configuration; `source`; the current working baseline
  (`real` + h8 + BC 0.25 from policy62/WM562); V5 (`real_wm` + h8 + BC 0.25).
- **Definitions.** Success, force-limit termination, material force and terminal
  reasons as in BENCHMARK ("Task definitions"). The 50 N limit is a check on the
  filtered sensor wrench, not on the material force.

## Next experiment — WM reliability check (running)

No policy training. Each step's outcome selects the next.

Status 2026-11-08 (RESULTS, "WM reliability check, Steps 1 and 3"): supplement
collected (dev 202/23/46 success/force-limit/workspace; final 102/13/14, stored
unopened); Step 1 route 1 reproduces every label; Step 3 outcome **mixed** (state
prediction much better with A at both checkpoints; terminal-head workspace
misses and force underestimation worse; force-limit onset spikes missed by every
WM; the fixed-fit training split has no force-limit and 3 workspace episodes).
No joint training with A. Step 2 done (RESULTS, "Step 2"): the WM predicts the
higher force of the faster policies about as large as physics, but rates the
faster branch far better than physics (+11.9 vs +1.5) through the critic value
of imagined end states. Next decision is the user's (see below).

**Data.**
- Development data: `wm_frozen_fit_20261105` (train 96 episodes: 93 success,
  3 workspace; held-out 32, all success) and WM562/612/662. No existing data has a
  force-limit failure during cutting (run 3's 337 force-limit failures all end in
  the first 1–2 steps), and the held-out split has no failures.
- Supplement (GPU7, frozen policies, complete physical records: states, actions,
  sensor wrench, material force): ablation-(a) checkpoints (the faster,
  higher-force policies), policy62, and action perturbations of policy62 to reach
  the force limit mid-cut and the workspace bound. Target about 30 episodes per
  terminal reason. Budget: at most 400 episodes and 2 h wall in total. If a
  reason stays below 10 episodes, report the count and mark its metrics as
  insufficient; do not change thresholds, material or perturbation design to
  force the count without a new plan.
- Use boundary: the supplement is stratified by terminal reason and serves only
  diagnosis. Its proportions are not policy success or exceedance rates; those
  come only from unperturbed frozen evaluations.
- Perturbations, not seeds alone, create distinct trajectories: the same initial
  state and a deterministic policy give the same trajectory under any seed. The
  collection script fixes and records the perturbation distribution, the steps at
  which it is applied and the random stream per episode.
- Development vs final test: two disjoint random streams. A complete episode
  belongs to exactly one set. The development set is used in this plan. The final
  test set is stored unopened and evaluated only after the method is fixed, so
  that model changes made after looking at held-out results are not reported as
  independent tests.
- The 2 h wall budget includes the Step-2 prefix replays and branches. At the
  budget the collection stops and reports its coverage; no extra runs are added
  to fill a category.

**Step 1 — Three-way terminal check (labels before models).** On the same
development trajectories, for WM562 and later for both arms of Step 3:
(1) true state + task predicates (with the previous step's success counter);
(2) predicted state + task predicates; (3) predicted state + learned terminal
head (one end/continue output). Predictions at 1 step (teacher-forced) and
8 steps (open loop on recorded actions). Metrics per true terminal reason:
misses, false alarms, timing offset in steps.
- (1) must reproduce every recorded label; otherwise fix labels, alignment or
  predicate first.
- (1) right, (2) wrong → the error is in the dynamics prediction.
- (2) right, (3) wrong → the terminal head adds the error.

**Step 2 — Force prediction and its relation to policy choices.** On the same
physical trajectories and actions: WM-predicted vs simulated sensor wrench and
`peak_wrench` at 1 and 8 steps; signed peak error with underestimation reported
separately; peak timing; force-limit misses. `peak_wrench` is the per-step
maximum (BENCHMARK), so its targets and the persistence reference (previous
step's measured value) are per control step. Persistence is a reference point,
not an acceptance criterion. Main case: the ablation-(a)
trajectories, with the WM checkpoint those policies were trained with.
To relate errors to choices: at the same recorded states, branch on the first
action (ablation-(a) action vs policy62 action). After the first action each
branch continues closed loop with its own policy for 7 more steps; the WM
rollout and the physical execution use the same rule. Reported separately per
branch: the 8-step sum of rewards, the terminal critic value at the end, peak
sensor wrench, material peak force and terminal reason. If a branch terminates
early, both WM and physics stop that branch at its terminal (absorbing, as in
`imagine()`); the WM's predicted terminal and the physical terminal are compared
as in Step 1. Physical branches restore the recorded episode by replaying its
action prefix with fixed inputs; FP32 atomic accumulation makes repeated runs
differ slightly, so the replayed prefix is checked on the complete physical
state, controller memory and observation history against the spread of actual
repeated runs (measured first), not for bitwise identity.
- Record the model errors and whether the policy's preferred actions are rated
  better by the WM than by physics. Neither "WM worse than persistence → the
  policy exploits model error" nor "WM better than persistence → only the reward
  is responsible" is concluded from prediction error alone.
- If preferred actions are rated better in the model but are worse or more
  force-risky physically, model exploitation is supported; a further causal
  check (does the preference change after the model is corrected?) is planned
  then. If model and physics agree on the higher force, the speed/force
  trade-off of the objective is the next topic.

**Step 3 — Original vs A on fixed data.** A: during WM fitting only, the
terminal BCE reads detached predicted features and updates only the terminal
head; the state loss alone updates the dynamics. Policy imagination keeps the
full path action → predicted state → reward and terminal probability; the shared
`WorldModel.step()` is not changed, and a test checks that the actor still
receives gradients through the terminal probability. Same start (WM562),
initialization, optimizer state, data order, learning rate and 100 rounds as
`wm_frozen_fit_20261105`; the only difference is the BCE-to-dynamics gradient.
- Primary metric: held-out 8-step knife-tip position error and joint error.
  Secondary: peak-force error (underestimation), Step-1 terminal metrics per
  reason, state error vs BCE over rounds, state/BCE gradient ratio on the
  dynamics. Overall `term_acc` is not a criterion.
- Statistics: per development episode, the paired difference A − original of
  each metric (windows of one episode are averaged first, never resampled as
  independent samples); 95% bootstrap interval over episodes. Each metric is
  classified as improved (interval below zero for an error), worsened (above
  zero) or inconclusive (spans zero). Inconclusive is not "A has no effect".
- Adopt for a joint-training plan: primary metric improved at both +50 and +100,
  and neither peak-force underestimation nor any per-reason terminal metric
  worsened.
- Reject A: primary metric worsened, or inconclusive at both checkpoints with
  no secondary improvement. The next topic is the source of the prediction error
  (coverage, history inputs, integration audit; RESULTS 2026-11-04 open item).
  The report states whether the evidence was inconclusive or negative.
- Mixed — no joint training: primary improved but peak-force underestimation or
  a terminal metric worsened, or improvement at only one checkpoint. Report, then
  examine force/terminal representation and data coverage. No B/C stacking.

**Budget.** Steps 1–2 are evaluation only. Step 3 is two fixed-data fits of
about 20 min each (WM562→662 took 1166 s). One GPU (GPU7 or A6000).

## Conditional and later (not current tasks)

- **Joint training with A**, only if Step 3 adopts it; separate plan.
- **Comparisons:** SAC (parallel configuration, P1 below), the same WM with MPC,
  WM + BPTT, fixed compliance control. Other candidates in NOVELTY (Lagrangian
  constraint, other MPC variants, ablations) depend on the final claim and need
  their own plans.
- **Multi-object stage** (after the WM check). One `BatchedDiSECt` per object,
  worlds grouped by object; continuous parameters sampled per reset (precedents
  in RESULTS/NOVELTY).
  - P0 per-world materials: passed (RESULTS 2026-11-08).
  - Scenes exported and stable placement checked for free prism, jello cylinder
    and apple sphere (no rolling ≤ 0.18°, no sliding ≤ 1.8 mm). The reference
    control completes the task for prism and jello; for the free apple it ends in
    force-limit termination at step 18 (peak 51.29 N, 3/244 springs released).
    This does not show that no controller can cut the apple. Material and force
    limit are not changed to admit it; how the apple enters is an open decision.
  - P1 parallel SAC on the single prism (1024 worlds, lr 3e-4, step budget
    matched); P2 randomized prism with held-out ranges; P3 objects.
  - Open: material ranges (from literature or calibration), whether the policy
    observes an object identifier.
- **Real robot:** [REAL2SIM](REAL2SIM.md), draft awaiting the user's review.
- **Parked:** DiSECt trajectory optimization (pipeline test, RESULTS 2026-11-08;
  its adjoint needs a static bottom).

## Run and machine status (2026-11-08)

- A6000 GPU 0/1 and GPU7: idle.
- Run 1 (SliceIt SAC, Gazebo, tamsGPU9): GPU9 is unreachable ("no route to host"
  on 2026-11-08), so its live state is unknown. Last recorded snapshot: 46044
  environment steps, 43747 updates, 2297 episodes; its 72 h execution deadline
  from 2026-11-03 has passed.
- Machine routes: `runs/operations/MACHINES.md` on the Mac (outside Git).
