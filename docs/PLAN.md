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

## Next experiment — WM reliability check (awaiting approval)

No policy training. Each step's outcome selects the next.

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
- Development vs final test: the supplement is collected with two disjoint seed
  sets. The development set is used in this plan. The final test set is stored
  unopened and evaluated only after the method is fixed, so that model changes
  made after looking at held-out results are not reported as independent tests.

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
separately; peak timing; force-limit misses. Persistence (last measured force)
is a reference point, not an acceptance criterion. Main case: the ablation-(a)
trajectories, with the WM checkpoint those policies were trained with.
To relate errors to choices: at the same recorded states, compare the ablation-(a)
action with the policy62 action by WM-predicted H8 return/force and by the
physically executed return/force (replay the recorded action prefix — frozen
episodes are deterministic — then execute the alternative actions; verify the
replayed prefix reproduces the recorded states before using it).
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
  dynamics. Overall `term_acc` is not a criterion. Uncertainty: 95% bootstrap
  intervals over development episodes.
- Adopt for a joint-training plan: primary metric better at both +50 and +100
  with intervals excluding zero, and no secondary terminal metric worse beyond
  its interval.
- Reject A: primary metric not better at either checkpoint. The next topic is
  the source of the prediction error (coverage, history inputs, integration
  audit; RESULTS 2026-11-04 open item).
- Mixed (primary better, a terminal metric worse beyond its interval, or only one
  checkpoint better): no joint training; report, then examine terminal
  representation and data coverage. No B/C stacking.

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
