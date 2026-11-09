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

Current stage (2026-11-09): the A decision is closed (RESULTS, "A control
pair"): no clear benefit on the pre-stated main quantity, so A is not advanced
as a control method; its prediction improvement and the observed training
stability difference go to the analysis. No extra seed pair (user decision).
Next: the main comparisons with the working baseline (proposal below).
Multi-cut tasks, loss reweighting B, feature rescaling C and new contact-model
architectures remain outside this stage.

## Decisions in force

- Benchmark: free-bottom `public-prism-ur5e-v1`. Fixed-bottom results remain in
  RESULTS; they do not change the main task.
- Method for the main comparisons: the working baseline, `real` critic + h8 +
  BC 0.25, continued from policy62/WM562 with a fresh critic and 100 pre-fit
  rounds. A (`--wm-terminal-gradient head`) is reported as analysis, not as the
  method (decision 2026-11-09).
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

## Completed — baseline vs baseline + A (2026-11-08 to 2026-11-09)

Outcome in RESULTS (2026-11-09): final-checkpoint perturbed success
A − baseline −0.016 [−0.059, 0.031], no clear benefit; baseline arm
collapsed into step-1 force-limit collisions at updates 250/300, A did not.
The protocol below is kept as the record of what was pre-stated.

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

**Measurements.** (Revised 2026-11-08 after Codex review of `f316f74`.)

From the fixed start the frozen task is near its ceiling (policy62 8/8 in 20
steps; working baseline 8/8 in 17–19), so the main outcome is robustness of the
frozen policies under predefined action perturbations. Two layers of uncertainty
are kept apart: case-paired intervals describe the difference between these two
trained policies on this perturbation distribution; they do not include the
variation from another training seed. Method development stays at one seed; more
training episodes do not replace seeds.

1. **Perturbed frozen evaluation (main outcome).** `scripts/evaluate_perturbed.py`:
   256 cases fixed before any policy runs (index, sigma 0.3/0.6/1.0, first
   perturbed step 0/5/10, i.i.d. or persistent direction, own noise seeds; seed
   base 3000000, disjoint from the reliability supplement). Every arm and
   checkpoint executes the same cases, paired by index; each case runs to its own
   end and all four outcomes are kept (success, force limit, workspace,
   time limit); no stopping by outcome counts. Reported: outcome rates; material
   peak/mean force and safety margin over all cases; steps and force also on the
   cases both arms complete successfully. Results are called robustness under
   action perturbations.
2. **Training episodes (auxiliary).** Terminal-reason rates per 20 vector steps;
   on-policy exposure, not held-out rates.
3. **Unperturbed frozen 8 episodes (reported).** The fixed-start task result.
4. **Reference run (shown, not a threshold).** `critic_abl_a250_20261106` has the
   baseline configuration on an older commit; measurements 1–3 are shown for it.
   One baseline − replicate difference is one repeat, not a bound on training
   noise. A bitwise match would only show that this resume setup is reproducible.
5. WM prediction errors explain results; they do not decide them.

**Budget addition.** Perturbed evaluation of 15 checkpoints (A, baseline,
reference) × 256 cases, about 2 h on one GPU after training.

**Decision.** Main quantity: the case-paired difference A − baseline of the
perturbed success rate at the final checkpoint; force-limit, workspace and
time-limit rates reported separately; other checkpoints show the trend.
- Interval supports an improvement → development evidence; A advances to the
  main comparisons.
- Interval includes zero → "no clear benefit detected" (not "no benefit"); A is
  not advanced as a control method.
- Clearly worse → the candidate is rejected.
- Mixed (e.g. success up, force-limit rate up) → this round closes with a mixed
  conclusion.
"Not significantly worse" is not evidence that the arms are equally safe.
No claim of simultaneous faster and lighter cutting is assumed.

**Expectation, stated before the run.** The most direct effect of A is on
prediction and control near task endings. All WMs missed the contact-onset force
spikes of force-limit episodes; whether A changes force-limit risk through
changed predictions and policy behaviour is an open experimental question.

**If A is not a control method.** The comparisons (SAC, same-WM MPC, fixed
compliance) proceed with the working baseline. `real` critic + h8 + BC is an
existing combination, not a contribution by itself; the paper's contribution
must then come from mechanism evidence and comparison results. A further model
change is proposed only against a measured failure that limits control, with its
own plan and literature basis.

An implementation error that invalidates the pair is repaired and its affected
arm rerun within a separately recorded budget. Another imperfect local metric
alone does not create a new blocking stage.

## Paper evidence after this decision

### Next stage — main comparisons (proposal, awaiting approval)

Question: with the same task, observations/actions and evaluation, how does
WM + BPTT policy optimization compare with planning on the same WM, model-free
SAC and fixed control, in completion, force-limit risk, time, peak force and
impulse, and online computation?

- **WM + BPTT (method).** Working-baseline checkpoint: `critic_abl_a250_20261106`
  final (named as the working baseline in README before the A pair). The A
  pair's baseline arm (same configuration, collapsed at updates 250/300) is
  reported as a repeat with that variation, not replaced by the better run.
- **Same-WM MPPI.** Paired with that checkpoint (`--paired-checkpoint`), planner
  H16 / 512 samples / 4 refinements / temperature 0.1 / noise 0.5 / gamma 0.99.
  Terminal value: the paired `real`-target critic, fixed once for this comparison
  because the actor's objective uses it. Planning time reported per step.
- **SAC.** Unified environment, parallel worlds (stage P1), learning rate 3e-4,
  environment-transition budget matched to the method's total physical
  transitions including the run that produced policy62/WM562; needs its own run
  plan with the compute budget.
- **Fixed control.** Constant-zero action (fixed compliance gains) and the
  prescribed reference trajectory, labeled separately (it bypasses the action
  mapper).
- **Evaluation.** Unperturbed frozen task and the 256-case perturbation
  robustness of BENCHMARK for every controller, plus lateral goal error.
- **Compute.** A6000 is occupied by another user's jobs (2026-11-09); GPU7 is
  shared with Codex. Evaluation-only items (MPPI, fixed control) fit on GPU7;
  SAC training needs A6000 time.


### Evaluation plan for the main comparisons (draft for Codex/user review, 2026-11-09)

Evaluation only; no training. Purpose: development evidence for the main
comparisons on the fixed prism task. The paper's robustness claims are planned
with material/geometry variation later; action perturbations are not a main-text
claim.

**Controllers keep their own interfaces.** Only the task, physics, action mapper
and evaluation conditions are shared.

| Controller | Checkpoint / source | Input | Output | Notes |
|---|---|---|---|---|
| WM + BPTT policy (method) | working baseline `critic_abl_a250_20261106` final (`runs/sm_ckpts/abl_a250_last.pt`); policy62 shown as the common start | ten 24-component observation frames + goal | 6-D raw linear output; 40-50% of components exceed [-1, 1] and are clipped by the environment | 1 ms/step |
| SAC | unified-environment SAC, step 85000 (`runs/sac_frozen_ckpt85000_20261107`); SliceIt tf2rl Wave actor through the Python 3.8 process bridge, deterministic | current 24-component observation | tanh, inside [-1, 1] | one environment, lr 0.01; collapsed at 90000 (RESULTS 2026-11-07); stand-in until the parallel SAC retrain |
| Same-WM MPPI | the working baseline's own WM (`--paired-checkpoint`), H16 / 512 / 4 / temperature 0.1 | WM state (joint, force and control histories) | sampled in [-1, 1] | about 0.5 s/step |
| Fixed control | constant-zero action (fixed compliance gains); prescribed reference trajectory | none | constant / knife reference | the reference bypasses the action mapper (reference branch) |

The environment clips every executed action to the task bounds [-1, 1]
(`GPUEnvironment.step`); this is part of the task, not of the evaluation.

**Condition 1 — unperturbed frozen task.** Policy: 1 deterministic episode
(earlier 8-episode runs were identical); SAC 85k: existing 8 episodes (8/8,
18 steps, 49.6 N); MPPI: planner seeds 0/1/2 with the paired critic as terminal
value (fixed before the run) and without (reference), done 2026-11-09; fixed
control: 1 episode each.

**Condition 2 — 256 predefined perturbation cases** (BENCHMARK, robustness under
action perturbations). Where the perturbation is added — decision for review:
- *Rule used now (`580597b`): executed action.* Clip the controller's action,
  add the case perturbation, clip again. The disturbance acts on what the robot
  executes, independent of each controller's internal output scale.
- *Alternative: each controller's own output.* Add the perturbation to the raw
  output, then the environment clips (the A-pair runs used this). For SAC and
  MPPI both rules give the same executed action (their outputs are already in
  bounds). They differ only for the WM policy: its saturated outputs (e.g. 1.6)
  absorb perturbations toward the interior, so this rule favours it.
- Runs under the two rules are never compared with each other (`compare` checks
  the recorded rule).

Applied to: policy62, working baseline (parallel worlds, about 10 min each) and
SAC 85k (one world, about 1-2 h). Not applied to: MPPI (0.5 s/step x 256 cases
x 30-110 steps is many hours, and 4 of 6 unperturbed seeds already end at the
force limit), constant-zero (at least 346 steps per episode, about 10 h in one
world), reference trajectory (no action interface).

**Metrics.** Outcome rates (success, force limit, workspace, time limit); steps;
material peak and mean force (mean over actual simulated time); impulse; sensor
safety margin; for unperturbed runs also lateral and height goal error; planning
or inference time. Comparisons: case-paired bootstrap of controller minus
working baseline over the 256 cases; intervals describe these frozen controllers
on this perturbation distribution, not training-seed variation.

**Queue on GPU7** (`/dataHDD/dong/tmp/eval_fixed_p62_20261109.sh`, then
`eval_perturbed_main_20261109.sh`): reference and constant-zero (unperturbed),
policy62, working baseline, SAC smoke (2 cases), SAC 256 cases. About 2 h.

**Questions for review.** (a) Executed-action rule vs each controller's own
output. (b) Is SAC 85k (one environment, lr 0.01) acceptable as the model-free
stand-in until the parallel retrain? (c) Is a perturbed MPPI evaluation needed,
given its cost and the unperturbed onset failures?

### WM–MPPI configuration selection completed (2026-11-08)

Recorded development planner: frozen WM563, H16, 512 samples, four refinements, temperature
0.1, noise std 0.5, discount 0.99, no terminal critic. The physics, task reward,
action bounds and force limits retain the benchmark settings.

GPU7 planner seeds 0/1/2 completed the original goal-pose task in 42/46/36 steps;
material peaks 13.42/15.75/16.26 N, with no force-limit or workspace ending.
The previous H8 configuration exhausted its 600 s budget. Full measurements,
selection evidence and the rejected frozen-critic candidate are in RESULTS.
The executable command is in BENCHMARK.

The planner is ready for the common algorithm comparison. That stage fixes
information access, development cases and compute/data budgets for SAC,
WM–MPPI, BPTT policy and fixed control; it records planning time separately from
physical stepping. The A control pair retains its own decision rule above.

### Four-week delivery target (2026-11-08 to 2026-12-08)

1. Week 1: close the A control decision; establish MPPI, SAC and fixed-controller
   comparisons on the same task. Freeze the method claim after control evidence.
2. Week 2: selected single-cut material variations and held-out conditions;
   matched mechanism ablations. Finish the method and experimental-protocol draft.
3. Week 3: independent seeds after method/protocol freeze; conduct the scoped
   robot study if hardware, sensing and safety procedures are ready.
4. Week 4: finish plots, manuscript, reproducibility commands and demonstration;
   independent review of claims against recorded results.

Target: a complete manuscript and reproducible evidence. Robot availability and
an end-to-end advantage are the schedule's dependencies. The end of week 1 is a
decision point: if the candidate has no control benefit, choose a supported
empirical claim or revise the method and delivery target. Repeated local fixes
do not extend the same candidate indefinitely. Multi-cut tasks, new simulators
and visual-policy migration remain outside this four-week scope.

### Comparative evidence

PAPER specifies the purpose of these comparisons; each needs a concrete run plan.

1. Select the method supported by control evidence and finish its mechanism
   ablations, reusing completed source/critic/BC-only results where applicable.
2. Same-WM MPC requirements (review 2026-11-09): pair each planner run with the
   WM and critic of the policy it is compared with (WM562 pairs with policy62;
   the working baseline and A use their own final WM/critic); fix the terminal
   value option once for that comparison with the paired critic. The rejected
   MPPI critic was real-target with 100 pre-fit rounds, evaluated on imagined
   candidates; it was not the original imagined-target critic. State the
   information difference (the WM uses joint/force/control histories, the policy
   ten frames of 24-component observations plus its goal); report planning time per control step
   (paired WM562 run 0.49 s; WM563 development runs 0.55-0.64 s; control period 50 ms) and the lateral goal error for every
   method. Compare it with SAC, same-WM MPC and fixed compliance control in the unified
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

## Last recorded machine snapshot (2026-11-09)

- A6000 GPU 0: a_pair_base_20261108; GPU 1: a_pair_head_20261108.
  Both run frozen code 7c1ab92 in Cutting_DEV_wmterm and retain all active paths.
- GPU7: canonical Cutting_DEV main; inactive experiment checkouts and data
  consolidated. No new planner/collection/training job is started during cleanup.
- GPU9 is retired from the working plan.
- Machine routes, storage and protected dependencies:
  server runs/operations/MACHINES.md. Development now uses Remote SSH.

### WM–MPPI episode video completed (2026-11-08)

The seed-0 frozen WM563 H16 episode reached the task goal in 43 steps / 2.15 s,
with material peak 13.25 N. The recorder shares the benchmark control loop;
video uses original UR5e visual meshes, the DiSECt surface including virtual cut
vertices, and measured force traces. Start/contact/end frames and raw-data
alignment checked. Output: `runs/videos_20261108/wm_mppi/`; commands in RUNNING,
numbers in RESULTS. Physical task and controller settings unchanged.


## Completed — remote repository consolidation (2026-11-09)

The user requested cleaning the repositories and GPU storage first, moving
source and experiment files off the Mac, and using Remote SSH afterward.
GPU7 main is the development entry. Archive inactive experiment checkouts and
preserve their uncommitted changes. Keep the A6000 A pair and every path it uses
in place while running. GPU9 is retired from the working plan.

After consolidation, the next MPC question is task feasibility across different
objects/materials or cutting conditions, with additional interaction data where
the WM lacks coverage. The recorded WM563/MPPI result establishes completion of
the existing prism goal-pose task. It does not supply apple or multi-material
training coverage. Choose the next task and dataset scope before launching new
collection or fitting. A broad SAC/policy/reference comparison is not the next
GPU7 job. No new MPC, collection or training batch is started during cleanup.

### Completed — checkpoint-pair repair (2026-11-09)

Code `031ffaf`: manifests read actual WM/policy counters and critic training
identity; `--paired-checkpoint` rejects unequal WM/policy weights and unequal
critics when used. The historical planner is correctly named WM563, with its
real-target pre-fit critic. Original WM562 is stored separately.

The original WM562 paired run finishes in 109 steps, peak 24.22 N; policy62
finishes in 20 steps, peak 30.66 N. Both reach the original goal predicate
without force-limit/workspace termination. Root 225 passed / 5 skipped,
backend 28 passed. Endpoint prediction error explains an observed discrepancy
in goal-entry counting; full numbers and commands are in RESULTS (2026-11-09).
The next MPC task remains coverage across objects/materials, as specified above.
