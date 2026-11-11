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
| Missing before S3 | Task force limit (Mini45 details pending); SAC on the multi-object task (deferred, not started) |
| Main experiment (S3) | Running since 2026-11-11: base and base + quantile risk penalty (RESULTS run record); SAC and ensemble comparator not started |

## Next steps, in order

1. **First-run task specification (S0) — decided except the force limit (2026-11-11).**
   Configs (drafts until the limit is filled in; the loader refuses a missing limit):
   `configs/multi_condition/s0_train.json` and held-out `s0_heldout_material_low.json`,
   `s0_heldout_material_high.json`, `s0_heldout_start.json`, `s0_heldout_geometry.json`.
   - Training objects: benchmark prism, potato_003, cucumber_003; apple deferred.
   - Per episode: Young's modulus and cut-spring stiffness ×0.5–2, density ×0.8–1.25
     (log-uniform; a simulation design range, not a measured calibration); target speed
     1–5 cm/s. Per world: knife start 0.5–3 cm above the object. 256 worlds per object.
   - Held-out, one factor at a time: material ×0.35 and ×3; start 3–4 cm; cucumber_002
     (an unseen geometric instance, not an unseen food class).
   - Force limit: not set. Needed first: the Mini45 model and calibration (or file), its
     mounting position and axis directions, and the sensor-to-blade distance; the stop
     threshold then follows from these and the experiment requirements, not from the
     measured cutting forces plus a margin. The 50 N / 10 N·m values remain only for
     historical interface checks (`smoke.json`).
   - Report goal completion and physical separation separately.
2. **Later, not started automatically:** SAC on the multi-object task (keep SliceIt's tf2rl
   SAC, parallel collection, one update per collected transition) and the S3 launch,
   after the force limit is set. Done: parallel object groups, ensemble-risk comparator,
   multi-object evaluation script.
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
