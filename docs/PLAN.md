# Plan

Updated 2026-11-11. The only active plan. Method and claims: [PAPER](PAPER.md);
measurements: [RESULTS](RESULTS.md); code: [METHOD](METHOD.md); data: [DATA](DATA.md).
The previous long plan (route B screening rules, S0–S4 rationale) is archived as
[PLAN_20261110](archive/PLAN_20261110.md); its B1 rule has been applied.

## Fixed decisions

- Method paper. Problem: a learned WM under-predicts the force spike at contact
  onset and policy optimization exploits it. Method: the working base
  (real-data critic + 8-step BPTT + BC 0.25) plus a force-risk constraint.
- Experimental setting: one shared WM/policy/critic over several objects with real
  shapes, randomized materials, knife start height and target cutting speed.
  The platform, objects and baselines are the experiment, not a contribution.
- **One force limit for every object.** The learner gets no object identity,
  material label or per-object limit; it must infer hardness from force feedback.
- Task goal = target pose + target cutting speed (tracked by a reward term).
- No action-perturbation evaluations; no extra seeds or local diagnostics as
  preconditions. Engineering problems are fixed inside the step they block.

## Status (2026-11-11)

| Item | State |
|---|---|
| Base method | Implemented; fixed-prism development evidence (PAPER §5) |
| B1 risk screen | Passed at the minimum: risk head 13/13 vs ensemble 9/13 held-out force-limit events, equal false alarms (RESULTS 2026-11-10) |
| Force-risk penalty in training | Implemented (`--risk-constraint`): online 0.95-pinball risk head, penalty on predicted excess over 0.9 in BPTT, weight adapted from the physical force-limit episode rate (adaptive penalty, not an exact Lagrangian) |
| Multi-object environment | Implemented (`--conditions`): grouped scenes; knife start height drawn once per parallel world at construction (by IK); material scales and target speed drawn every episode; one force limit |
| Objects | Scanned OmniObject3D apple, potato, two cucumbers (fTetWild meshes, DiSECt official materials) plus the benchmark prism |
| End-to-end check | 46-iteration multi-object training with and without the constraint completes (WM, risk head, joint updates, checkpoint) |
| Missing before S3 | Parallel group stepping; SAC on the multi-object task; ensemble-risk comparator; multi-object evaluation script |
| Main experiment (S3) | Not started |

## Next steps, in order

1. **Object set and splits (S0).** Keep the task force limit (50 N / 10 N·m)
   unless it is changed deliberately as a new task setting, not tuned to produce a
   desired pass/fail pattern. Use the objects the reference or a sawing reference
   can cut under it (`scripts/measure_cut_force.py`); the apple, which reached
   ~200 N without completing at 2 and 5 cm/s, is deferred from the first S3 run.
   Report goal completion and physical separation separately (released-spring
   fraction is only a proxy). Held-out: unseen material scales and start heights
   on training shapes, plus one scan not used in training (an unseen instance /
   geometry, not an unseen food class).
2. **Code before S3.** (a) One persistent process per object group with the learner
   updating centrally; keep the sequential path as the reference and check equality
   on a short run (the GPU is shared, so four groups will not cost one group's time).
   (b) SAC on the multi-object task with the same goal input. (c) Ensemble-risk
   comparator: same penalty and weight adaptation, risk upper bound from an ensemble
   (mean + 2 std). (d) Multi-object frozen evaluation.
3. **S3 training (A6000).** Fresh weights, same distribution, same physical
   interaction budget: base, base + risk penalty, base + ensemble risk, SAC.
4. **S4 evaluation and writing.** Frozen policies on the held-out conditions:
   completion, force-limit and workspace endings, duration, peak force and
   impulse, speed-tracking error. Results go to RESULTS and PAPER.

Real-robot work follows [REAL2SIM](REAL2SIM.md): the simulated force sensor must
then be placed at the Mini45 mounting and the limit expressed in its frame.

## Storage and execution

Code and data live on GPU7 (`/dataHDD/dong/Projects/Cutting_DEV`, branch `main`)
and the A6000; the Mac keeps nothing. Check `nvidia-smi` before using a GPU.
