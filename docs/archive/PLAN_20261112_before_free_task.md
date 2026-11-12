> Archived 2026-11-12: PLAN as it stood before the free-object task decision (mixed older decisions such as the bottom hold).

# Plan

Updated 2026-11-12 after task/platform review. The only current task list.
Research scope: [PAPER](PAPER.md); measurements: [RESULTS](RESULTS.md);
implementation: [METHOD](METHOD.md); assets: [DATA](DATA.md).
Earlier plans: [2026-11-12 before the review](archive/PLAN_20261112_pre_review.md)
(defect table, ordered steps, topic options), [2026-11-11](archive/PLAN_20261111.md),
[2026-11-10](archive/PLAN_20261110.md).

## Current state

- First S3 multi-object runs are stopped. Their task-level comparison is
  withdrawn; retained evidence and deletion records are in RESULTS and DATA.
- The force-spike problem statement is retired. The research question is open.
- WM+BPTT, risk penalties, official SuperMimic adaptation, SAC and MPPI are
  implemented code, not selected contributions or proof of task validity.
- The initial cleanup changed only documentation/labels. Subsequent authorized
  repairs changed net-wrench handling and material contact coordinates (below);
  they did not change rewards or success thresholds and did not restart learning.
- Machine availability is a timestamped observation; check before launching work.

## Decisions by the user

Standing decisions (unchanged since before the review):
- Method paper for ICRA/IROS/RA-L. Platform, objects and baselines are experiments.
- One shared WM/policy/critic over scanned objects, randomized materials, knife start
  height, target cutting speed as a task input.
- One force limit for every object; no object identity, material label or per-object
  reward/success setting reaches the learner; it adapts from force feedback. Historical
  S0 used [50, 50, 50 N, 5, 5, 5 N·m]: a simulation threshold, not a Mini45 rating or
  real-robot safety limit.
- The paper's core must follow from the task, not from a failure seen in our own runs.
- One change at a time; a written plan is approved before a paid run. New experiments
  need a question, setting, decision rule and bounded budget.
- No action-perturbation suite, additional seeds or new long training is scheduled.

Decisions and assessments of 2026-11-12:
- Assessment: the candidate topic's motivation is too simple and has too few modalities
  for ICRA/IROS as it stands; "fixing the WM's force spike" was a pseudo-problem.
- Goal: one policy that cuts through several objects. Sawing is required (a pure press
  is not enough to show a method). Hardness is not classified or given; the policy has
  to learn it while cutting.
- Simulation first, sim-to-real intended. The real robot is available from the following
  week and is not the current focus; the user calibrates materials on it later.
- Objects stay held by their bottom for now.
- The simulation is to be changed so that sawing is represented (approved); other
  simulators were surveyed (RESULTS). Geometry input is favoured; tactile sensing was
  considered and not adopted (no credible sim-to-real path).
- Hand-tuned speed-governor baselines are not a current step and not a gate.

Decision after the fixture/label discussion (2026-11-12, supersedes "objects stay held by
their bottom" above for the new task):
- Task: single cut of an object placed freely on the board. Slight passive sliding is allowed
  and is not a failure; active rolling and repositioning are out of scope. New fixture schemes
  are cancelled.
- Success: the slice is fully separated from the body (no remaining material connection)
  without a force-limit event. A remaining connection is recorded as incomplete; 98 % spring
  release is a diagnostic, not completion.
- Order: one representative scene confirms object-board contact, knife contact and the
  severed-state check (done, RESULTS "Free-object single-cut task"); then the reward for this
  task; then training with a stated budget. The fixed reference need not succeed on every object.

Cautions kept from the review: a method must address a practical limitation supported by
literature and task evidence; more modalities or objects do not by themselves establish
novelty; a shared multi-object policy is the chosen setting, not by itself a contribution.

## What has to be resolved

1. **Real task and completion.** Specify the intended cut/product, knife,
   realizable fixture, allowable movement, desired speed and unacceptable
   outcomes. Reaching a pose does not establish separation. Proposed and not yet
   approved by the user: released cut-spring fraction >= 0.98 without a force-limit
   event (real robot: knife at board height); its relation to physical separation
   has to be checked first.
2. **Platform and force meaning.** The inherited material bridge sums contact
   magnitudes, places the scalar on one axis and contributes zero material torque;
   native MuJoCo wrench is also present. Optional net feedback is now implemented
   and checked for contact location, sign, frame, moment and sampling consistency.
   Select the task's feedback convention and calibrate it; implementation checks
   do not validate friction or real cutting. Model a real fixture rather than
   tuning holds per object to obtain desired force curves.
3. **A bounded reference demonstration.** Once 1–2 are specified, use a few
   representative cuts to distinguish separation, sliding, blade-face contact
   and board contact. Reference travel must reflect the actual start. Residual
   force is not automatically wrong while the blade remains in contact. Check
   reward preferences against intended behavior before another long run.
4. **Research question and comparison.** Read the closest task papers for real
   limitations, assumptions, measurements and deployment requirements. A simple
   force-feedback controller is a useful baseline after task correction. Its
   success does not invalidate the field; its failure does not prove a WM is
   necessary. Select the method and comparisons from the remaining practical gap.
5. **Training after that decision.** Version the corrected task and write the
   bounded training/comparison plan. Old S0 files remain reproduction inputs,
   not launch-ready replacements for a corrected task.

These are decisions for one task closure, not an open-ended diagnostic queue.
This cleanup introduces no numerical acceptance thresholds or material ranges.
The old proposal (98% release, <3 mm shift, zero residual force) is unapproved,
not a task contract.

## Conditional options

Geometry input is useful only if the task needs information unavailable from
proprioception/force. The early scalar/high-transition-velocity measurements showed
negligible sawing benefit. Later net-feedback measurements with transition velocity
0.01 m/s showed a lower vertical force and a smaller reduction in net force on the
prism (RESULTS). These are configuration-specific reference cuts, not a learned
sawing policy or real-material validation. Heterogeneous materials
need their own practical motivation and validated representation. More modalities
or objects do not by themselves establish novelty.

## Storage and execution

Use Remote SSH on GPU7, /dataHDD/dong/Projects/Cutting_DEV, main; the A6000 holds a
second checkout. Code, data, environments and outputs stay on servers, not the Mac.
Check `nvidia-smi` before using a GPU. Not to be launched without a new approved plan:
`train_gpu.py --conditions configs/multi_condition/s0_*.json` (with or without
`--risk-constraint`), `train_official_supermimic.py`, `train_sac_gpu.py` on S0. Preserve historical
contracts and evidence. Do not restart S3 from a reproduction command.


Net-wrench interface repairs and the bounded press/saw rerun are recorded in
[RESULTS](RESULTS.md#2026-11-12--codex-net-wrench-repair-and-physical-rerun).
The contact-coordinate correction is implemented as `edge_coordinate_consistent_v2`.
Kernel confirmation and the bounded press/saw comparison are recorded in
[RESULTS](RESULTS.md#2026-11-12--contact-coordinate-correction-and-kernel-confirmation).
It resolves the source inconsistency identified in that interface review; it does not
replace material calibration or approve restarting training.

Review handoff for Claude: [CONTACT_FIX_REVIEW](CONTACT_FIX_REVIEW.md).
