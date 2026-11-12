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
- This cleanup changes documentation, entry descriptions and diagnostic labels,
  not physics, rewards, success rules or training behavior.
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
   native MuJoCo wrench is also present. Decide whether the study can use that
   proxy or needs validated net contact wrench/other physics. Replacing the sum
   alone does not validate friction or cutting. Model a real fixture rather than
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
proprioception/force. Sawing first needs evidence that the platform represents
its physical effect: the measured configurations showed negligible benefit,
not a universal result about DiSECt or real sawing. Heterogeneous materials
need their own practical motivation and validated representation. More modalities
or objects do not by themselves establish novelty.

## Storage and execution

Use Remote SSH on GPU7, /dataHDD/dong/Projects/Cutting_DEV, main; the A6000 holds a
second checkout. Code, data, environments and outputs stay on servers, not the Mac.
Check `nvidia-smi` before using a GPU. Not to be launched without a new approved plan:
`train_gpu.py --conditions configs/multi_condition/s0_*.json` (with or without
`--risk-constraint`), `train_official_supermimic.py`, `train_sac_gpu.py` on S0. Preserve historical
contracts and evidence. Do not restart S3 from a reproduction command.
