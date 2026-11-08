# Current plan

The single current task list. The research thesis, connected algorithm and paper
experiments are in [PAPER](PAPER.md). Measurements live in [RESULTS](RESULTS.md),
task and metric definitions in [BENCHMARK](BENCHMARK.md), implementation and
sources in [METHOD](METHOD.md), literature evidence in [NOVELTY](NOVELTY.md).

## Destination and current stage

Develop a world-model-based method for force-limited compliant cutting whose
policy improvements carry over to physical execution. The paper follows one
chain: **robot/material interaction data → motion/force model and task events →
BPTT policy optimization → executed cutting performance**.

Current stage: test whether the implemented terminal-supervision intervention A
improves actual control, on the existing single-cut task. Position prediction
already improved in the fixed-data comparison. This control comparison is the
next missing link. Multi-cut tasks, loss reweighting B, feature rescaling C and
new contact-model architectures are outside this stage.

## Decisions in force

- Benchmark: free-bottom `public-prism-ur5e-v1`. Fixed-bottom results remain in
  RESULTS; they do not change the main task.
- Current working baseline: `real` critic + h8 + BC 0.25, continued from
  policy62/WM562 with a fresh critic and 100 pre-fit rounds. Candidate: the same
  configuration with `--wm-terminal-gradient head` (A).
- Source, working baseline, V5, Gazebo SAC and unified-environment SAC keep their
  separate BENCHMARK names. Task, reward, material, action bounds and force-limit
  definitions stay fixed.
- Physics advances the robot and material; actor gradients pass through the
  learned model. Successful goal-pose completion and material separation retain
  their separate BENCHMARK measurements.
- Method development uses one training seed. Multi-seed confirmation follows a
  fixed method and comparison protocol.

## Completed evidence — WM diagnosis closed 2026-11-08

- True-state task predicates reproduce every checked recorded label.
- Isolating terminal BCE from dynamics fitting changes held-out H8 tip error
  from 7.88 to 0.28 mm after 100 rounds. The fixed training split has no
  force-limit and only 3 workspace endings; those predictions remain poor.
- On five checked successful policy trajectories, the WM captures the higher
  force of faster actions. The reward accepts this trade-off.
- Missed endings put critic evaluations beyond the physical episode. Their
  causal effect on actor learning remains to be tested by the control pair.

These results motivate the method experiment in PAPER. They are not a queue of
local metrics that must all improve before control can be evaluated.

## Next experiment — baseline vs baseline + A (proposed; awaiting approval)

This draft replaces the next-stage draft at `5678c97`. No new run is authorized
by this document update.

**Question.** Does isolating terminal supervision during WM fitting improve the
cutting policy learned through the WM, with the task and actor objective fixed?

**Hypothesis.** Better physical prediction near task endings improves executed
control. A changes the fitting gradient route; it is not a new force objective.

**Pair.** Both arms resume policy62/WM562, reset the critic and pre-fit it for
100 rounds; use `real` + h8 + BC 0.25, 1024 worlds, the same seed and update
budget. A changes only `--wm-terminal-gradient head`. Both retain the normal
online collection and WM update loop. The extra offline 100-round refit from
previous PLAN is removed: its outputs were not inputs to this joint comparison,
and fixed-data evidence already exists. The final supplement set remains sealed.

**Budget.** Retain the previous proposed budget: A6000 GPU 0 and 1, one arm per
GPU, 12 h wall each; `--max-iterations 860` (end iteration 2421). Existing
checkpoint/progress/timeout handling applies. No automatic extension.

**Measurements.** (Amendment proposed 2026-11-08 by Claude, for Codex/user
review: noise reference and outcomes with statistical power.)

*Why an amendment.* From the fixed start the frozen task is near its ceiling:
policy62 already succeeds 8/8 in 20 steps and the working baseline 8/8 in
17–19; frozen peak force varies between 26 and 46 N across checkpoints of one
run. Eight deterministic frozen episodes per checkpoint cannot separate the arms,
and with one seed an arm difference may be training noise.

1. **Perturbed frozen evaluation (primary).** For each arm and checkpoint
   (updates 150/200/250/300/last), the frozen policy runs 256 episodes from the
   benchmark start with action perturbations from fixed random streams, identical
   for every arm and checkpoint (same scheme as the reliability supplement:
   sigma 0.3/0.6/1.0, first perturbed step 0/5/10, i.i.d. or one persistent
   direction per episode; stream seeds disjoint from the supplement's).
   Reported per checkpoint: success, force-limit and workspace rates with 95%
   intervals (episode-paired bootstrap between arms, since streams are shared),
   steps to success, material peak/mean force and sensor safety margin of
   successful episodes. This measures control robustness and force-limit risk,
   the paper's outcome; it is not a WM diagnostic. Needs a small evaluation
   entry reusing the collector's perturbation code (one policy, no dev/final
   split, fixed episode count).
2. **Training episodes (primary, on-policy).** Terminal-reason rates per 20
   vector steps and over the last 100 vector steps (1024 worlds, thousands of
   episodes); not held-out generalization rates.
3. **Unperturbed frozen 8 episodes (reported).** Success, steps, return,
   material peak/mean force, cut fraction — the fixed-start task result.
4. **Noise reference.** The completed run `critic_abl_a250_20261106` has the
   baseline arm's configuration (same start, flags and iteration budget; older
   code commit with identical default behavior). Measurements 1–3 are also taken
   on its checkpoints. The baseline-vs-replicate difference estimates run-to-run
   variation. If the new baseline arm reproduces that run bitwise, training is
   deterministic and this reference is zero; the report says so and relies on the
   intervals of measurement 1 alone.
5. WM prediction errors on the arms' own data explain the result; they do not
   decide it.

**Budget addition.** Perturbed evaluation: 15 checkpoints (A, baseline,
replicate) × 256 episodes, about 2 h on one GPU after training (400 supplement
episodes took 651 s with 32 worlds).

**Decision.** Primary comparison at the final checkpoint, with 250/300 as
support; all checkpoints reported.
- **Benefit:** A − baseline improves the perturbed success rate or lowers the
  force-limit rate with an interval excluding zero, by more than the
  baseline − replicate difference, with no other failure rate worse beyond its
  interval, and no collapse (unperturbed 0/8) at any checkpoint. Training-episode
  rates point the same way. Then A advances to the main comparisons.
- **No benefit:** differences within intervals or within the replicate
  difference, with better prediction → A is reported as a model-fitting
  improvement only, not a control method.
- **Worse:** A − baseline worse beyond both → A is rejected for this setup.
- **Mixed:** reported as mixed; this candidate round closes.
No claim of simultaneous faster and lighter cutting is assumed; force is
compared at comparable success.

**Expected limitation, stated before the run.** Every WM missed the contact-onset
force spikes of the force-limit episodes (A included). A changes the fitting
route, not the force representation, so a reduction of force-limit failures is
not the expected main effect; an effect is more plausible near task endings
(success timing, workspace). A null result on force-limit risk is therefore an
anticipated outcome, not a reason for a new repair stage.

**If A is not a control method.** The method claim falls back to the working
baseline (`real` critic + h8 + BC on the WM/BPTT framework) with the mechanism
evidence as analysis, and the paper's comparisons (SAC, same-WM MPC, fixed
compliance) proceed with it. A further model change is proposed only against the
measured failure that limits control (currently the onset force spike), with
its own plan and literature basis (NOVELTY section 7).

An implementation error that invalidates the pair is repaired and its affected
arm rerun within a separately recorded budget. Another imperfect local metric
alone does not create a new blocking stage.

## Paper evidence after this decision

PAPER specifies the purpose of these comparisons; each needs a concrete run plan.

1. Select the method supported by control evidence and finish its mechanism
   ablations, reusing completed source/critic/BC-only results where applicable.
2. Compare it with SAC, same-WM MPC and fixed compliance control in the unified
   task. Match information and action access; account for warm-start data and
   model fitting when comparing cost. Prescribed trajectories remain separately
   labeled because they bypass the policy action mapper.
3. Test material variation and then held-out shapes/materials. Multi-object
   per-world parameters passed P0. Free prism, jello cylinder and apple sphere
   placement checks are complete. Reference control completes prism/jello;
   apple ends at the force limit. Choose material ranges and object access in
   that stage's plan, keeping the main task definitions.
4. Once method and protocol are fixed, run independent seeds and the real-robot
   study in [REAL2SIM](REAL2SIM.md). The robot plan still needs user review.

Paper drafting can already cover task, platform, source method, connected
algorithm and completed mechanism evidence. Main comparative claims follow the
control experiments. DiSECt trajectory optimization stays parked; it is not a
prerequisite for the same-WM MPC comparison.

## Last recorded machine snapshot (2026-11-08)

- A6000 GPU 0/1 and GPU7: recorded idle. Refresh before scheduling approved work.
- Run 1 (SliceIt SAC, Gazebo, tamsGPU9): unreachable ("no route to host") in the
  last check. Last recorded snapshot: 46044 environment steps, 43747 updates,
  2297 episodes; the 72 h deadline from 2026-11-03 has passed.
- Routes: `runs/operations/MACHINES.md` on the Mac (outside Git).
