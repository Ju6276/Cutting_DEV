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
| Force-risk constraint in training | Implemented (`--risk-constraint`): online quantile risk head, risk penalty in BPTT, dual update of its weight |
| Multi-object environment | Implemented (`--conditions`): grouped scenes, start height by IK, per-episode material and target speed, one force limit |
| Objects | Scanned OmniObject3D apple, potato, two cucumbers (fTetWild meshes, DiSECt official materials) plus the benchmark prism |
| End-to-end check | 46-iteration multi-object training with and without the constraint completes (WM, risk head, joint updates, checkpoint) |
| Main experiment (S3) | Not started |

## Next steps, in order

1. **Force limit and object set (S0).** Measure the force needed to cut each object
   at slow and fast knife speeds (`scripts/measure_cut_force.py`, running). Choose
   one limit such that a careful (slow) cut stays below it and a forceful cut
   exceeds it; drop objects that cannot be cut below it at any speed. Fix the
   training ranges (material scales, start clearance, target speeds) and the
   held-out conditions (material scales and start heights outside training,
   one held-out object if available), then write them into one config.
2. **Parallel object groups.** Groups are currently stepped one after another
   (~10 s per step for four groups). Step them concurrently so a multi-object
   step costs close to a single-group step; verify identical results to the
   sequential path on a short run.
3. **S3 training (A6000).** Fresh weights, same distribution, same physical
   interaction budget: (a) base, (b) base + force-risk constraint, (c) SAC.
   A precedent comparator (ensemble-pessimistic force cost) is added if time
   allows.
4. **S4 evaluation and writing.** Frozen policies on the held-out conditions:
   completion, force-limit and workspace endings, duration, peak force and
   impulse, speed-tracking error. Results go to RESULTS and PAPER.

Real-robot work follows [REAL2SIM](REAL2SIM.md): the simulated force sensor must
then be placed at the Mini45 mounting and the limit expressed in its frame.

## Storage and execution

Code and data live on GPU7 (`/dataHDD/dong/Projects/Cutting_DEV`, branch `main`)
and the A6000; the Mac keeps nothing. Check `nvidia-smi` before using a GPU.
