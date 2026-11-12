# Plan

Updated 2026-11-12 after task/platform review. The only current task list.
Research scope: [PAPER](PAPER.md); measurements: [RESULTS](RESULTS.md);
implementation: [METHOD](METHOD.md); assets: [DATA](DATA.md).
Earlier plans: [2026-11-11](archive/PLAN_20261111.md), [2026-11-10](archive/PLAN_20261110.md).

## Current state

- First S3 multi-object runs are stopped. Their task-level comparison is
  withdrawn; retained evidence and deletion records are in RESULTS and DATA.
- The force-spike problem statement is retired. The research question is open.
- WM+BPTT, risk penalties, official SuperMimic adaptation, SAC and MPPI are
  implemented code, not selected contributions or proof of task validity.
- This cleanup changes documentation, entry descriptions and diagnostic labels,
  not physics, rewards, success rules or training behavior.
- Machine availability is a timestamped observation; check before launching work.

## Decisions retained

- Cutting is the application; a method paper is the aim. A method must address
  a practical limitation supported by literature and task evidence.
- Shared policies across objects remain a candidate setting, not a reason to
  preserve a defective task or a settled scientific contribution.
- Historical S0 used [50, 50, 50 N, 5, 5, 5 N·m] for all objects: a simulation
  threshold, not a Mini45 rating or real-robot safety limit.
- No action-perturbation suite, additional seeds or new long training is scheduled.
  New experiments need a question, setting, decision rule and bounded budget.

## What has to be resolved

1. **Real task and completion.** Specify the intended cut/product, knife,
   realizable fixture, allowable movement, desired speed and unacceptable
   outcomes. Reaching a pose does not establish separation; neither 98% spring
   release nor board height is an accepted substitute. Spring release can be a
   simulation diagnostic after checking its relation to the intended separation.
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

Use Remote SSH on GPU7, /dataHDD/dong/Projects/Cutting_DEV, main. Code, data,
environments and outputs stay on servers, not the Mac. Preserve historical
contracts and evidence. Do not restart S3 from a reproduction command.
