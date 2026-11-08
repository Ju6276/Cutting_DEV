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

## WM reliability check — closed 2026-11-08

Answers (RESULTS 2026-11-08, Steps 1–3); this round of diagnosis is finished.
- Labels, timing and predicate are aligned (route 1 reproduces every label).
- The terminal BCE disturbs the dynamics: with A (BCE isolated from the dynamics
  in WM fitting) state prediction is much better (held-out H8 tip error
  7.88 → 0.28 mm); force-limit and workspace prediction stay poor, confounded by
  a training split without force-limit and with 3 workspace episodes.
- "The faster policies cut harder because the WM under-predicts force" is not
  supported on the five checked trajectories: the WM predicts the higher force,
  and the SliceIt objective accepts the trade-off.
- "Terminal-prediction errors affect policy learning" remains untested (missed
  terminals put critic evaluations on out-of-support imagined states; the actor
  objective was not compared).

## Next stage — does A improve the actual cutting control? (awaiting approval)

One question, one route. No new networks, losses or collection; no further
prediction diagnostics as preconditions. Multi-cut, multi-object, seeds and
extra algorithms are not preconditions.

**1. WM fit with the existing coverage (informational, not a gate).** Original vs
A from WM562, identical initialization, seed and 100 rounds, on the fixed-fit
training split plus two thirds of the dev supplement (stratified by terminal
reason); evaluated on the fixed-fit held-out split and the remaining third of
dev. The final supplement set stays sealed. GPU7, about 30 min. Reported, not
used to stop step 2.

**2. Joint-training comparison (the decision run).** Current working baseline vs
working baseline + A (`--wm-terminal-gradient head`), both resumed from
policy62/WM562 with critic reset and 100 pre-fit rounds, `real` + h8 + BC 0.25,
1024 worlds, `--max-iterations 860` (iteration 2421, as ablation (a) and
bc_only); rewards, force limit and critic settings unchanged; one seed.
A6000 GPU 0 and 1 in parallel, 12 h wall each. Primary outcomes, physical
execution: frozen 8-episode evaluation at updates 150/200/250/300/last (success,
steps, force-limit failures, material peak and mean force, safety margin) and the
noisy training episodes (success, force-limit and workspace failure rates per
20 vector steps). WM prediction errors explain the result; they do not decide it.

**3. Decision.**
- Keep A: across the frozen checkpoints A is at least as successful as the
  baseline and has fewer force-limit failures in training episodes or lower
  peak/mean force at no more steps, at 3 or more of the 5 checkpoints, without a
  collapse (0/8) at any checkpoint. Then: mechanism ablations and the SAC /
  same-WM + MPC comparisons, each with its own plan.
- Drop A: otherwise. The candidate ends; no further small repairs on this line.
  The mixed record of the fixed-data comparison stays as it is.

**Paper writing starts now** on task, related work, the unified environment and
existing results; the method claim waits for step 3.

## Conditional and later (not current tasks)

- **After the A decision:** mechanism ablations and comparisons (below), each with its own plan.
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
