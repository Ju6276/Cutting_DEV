# Plan

Updated 2026-11-11 (evening). The only active plan. Method and claims: [PAPER](PAPER.md);
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
- **Force sensing and limit.** The real robot uses an external ATI Mini45 (Net F/T),
  not the UR5e built-in sensor, so the inherited SliceIt 50 N / 10 N·m limit is not
  the task limit. The task limit is a required six-component config (`force_limits`),
  still to decide using the actual Mini45 calibration, mounting/tool and experimental
  requirements. Sensor measurement range is not itself the task stop threshold;
  the simulated sensor is moved to the Mini45 mounting for sim2real.
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
| Parallel object groups | Done: one process per group, 2.5× faster than sequential, consistent within the simulator's run-to-run variation |
| Ensemble-risk comparator | Done (`--risk-estimator ensemble`) |
| Multi-object evaluation | `scripts/evaluate_multi.py`: loads and runs (functional run 2026-11-11); per-episode rows not yet exercised |
| Missing before S3 | SAC on the multi-object task (training is still single-environment; only SAC evaluation is batched); task force limit value |
| Main experiment (S3) | Not started |

## Next steps, in order

1. **First-run task specification (S0) — proposal, to decide.**
   - Force limit: one shared six-component task setting, not yet chosen (see above).
     Do not raise it just to admit an object. Apple remains deferred: its old,
     unheld straight-press measurements do not establish its required cutting
     force; held/sawing measurements would be separate future work.
   - Training objects: benchmark prism, potato_003, cucumber_003 (all cut through when
     held, RESULTS 2026-11-11). Held-out object: cucumber_002 (unseen instance/geometry,
     not an unseen food class). Apple: deferred unless the limit and a sawing cut admit it.
   - Per episode: Young's modulus and cut-spring stiffness ×0.5–2, density ×0.8–1.25
     (log-uniform); target speed 1–5 cm/s (the measured range). Per world: knife start
     0.5–3 cm above the object.
   - Held-out conditions: material scales ×0.35 and ×3, start 3–4 cm, cucumber_002.
   - Scale: 256 worlds per object (A6000); the same physical-interaction budget for
     every method (~2–2.5 M transitions, as in the from-scratch source chain).
   - Report goal completion and physical separation separately (released-spring
     fraction is only a proxy).
2. **Code repair for review (user scope, 2026-11-11).** Implemented: ensemble WM
   learning-rate scheduling, explicit valid multi-object force limits, and worker
   lifecycle handling. Validation and patch details are in RESULTS. Claude reviews
   this patch before the next experimental step.
   No new policy training, evaluation experiment or SAC parallel implementation
   in this step.
   **Deferred:** multi-object SAC training and expanded frozen evaluation. Before
   formal comparison, evaluation must check the checkpoint/task interface and use
   case-indexed condition sampling for paired cases (the current shared reset RNG
   is not a paired-case guarantee). Complete-episode evaluation remains untested.
   SAC collection/update scheduling will be specified when that work resumes.
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
