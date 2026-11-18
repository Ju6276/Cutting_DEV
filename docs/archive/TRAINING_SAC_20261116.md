> Historical — superseded by [CURRENT_BASELINES](../methods/CURRENT_BASELINES.md) and [PLAN](../PLAN.md), kept for traceability.

# GPU9 current-task parallel SAC — 2026-11-16

## Question and scope
Can model-free SAC learn the current shared three-object pose task under the same
physical-interaction cap as the WM+BPTT runs? (Same cap, not the same actual
interaction: SAC reached 1,281,024 transitions; the WM+BPTT runs are projected,
as of 2026-11-17 ~16:00 CET, to stop at their 24-hour bound with fewer. Use
recorded final counts and compare at equal transition counts.) This is a development training run,
not a claim of algorithm superiority or full material separation.

The user authorized GPU9 multi-environment SAC. GPU9 was idle at launch preparation.
Existing GPU7/A6000 training checkouts remain untouched.

## Fixed settings
- Implementation: Stable-Baselines3 2.7.0 PyTorch SAC, current vector bridge;
  not an identical reproduction of the original TensorFlow SliceIt network.
- Source before this documentation change:30a5eeb.
- Current DiSECt material backend; cucumber003, half potato, prism.
-3072 environments:1024 per object. Same task/distributions as GPU7 main;
  input file runs/baseline_validation_20261116/conditions.json has server-local assets.
- Observable10x34 history plus8-dimensional goal; six bounded actions, pitch off.
- pose_depth_v2; final-position-only;180 steps; gamma1; force limits[50,50,50,5,5,5].
- Actor/critics MLP256x256; lr3e-4; batch256; tau.005; automatic entropy.
-30720 warmup transitions, one SAC gradient update per subsequent physical
  transition; replay capacity1,000,000. This does not equate SAC update counts
  with world-model learner rounds.
- Seed0; start fresh. No test-case tuning or automatic budget extensions.

## Budget and recording
- Maximum1,280,000 new transitions or24 wall hours from construction.
  Full vector steps round the transition cap upward by at most3071 transitions.
  Wall stop is at a complete rollout/update boundary, so one cycle can overrun.
- Periodic rolling model/replay/metadata every100,000 transitions, plus final save;
  metadata hashes detect mixed checkpoint generations.
- Online W&B mandatory; verify remote history after startup. Local server logs
  and checkpoint/replay are recovery artifacts, not substitutes for W&B.
- Record transitions, updates and observed episode returns. Final frozen evaluation
  will report per-object pose outcomes, force, speed and material-separation diagnostics.
- A bounded3072-world smoke test first collects3072 transitions and completes
  SAC updates at the actual requested count; small-world reset/resume was already tested.
- No performance threshold changes the task or extends this run. Report the
  final frozen performance and any numerical/runtime failure as measured.

## Launch evidence
The3072-world preflight completed:3072 transitions,3072 SAC gradient updates,
3072 replay transitions, zero residual update credit, finite parameters and saved
checkpoint. Total475.37s, of which439.94s was construction/initialization.
This was one vector-step capacity/update check, not a long-run learning result;
masked terminal reset and resume were previously checked with6 worlds.

Formal run launched from frozen checkout5be7d90:
- GPU9 /dataHDD/dong/Projects/Cutting_DEV_sac_long_20261116
- supervisor167613, trainer167614
- output runs/sac_long_20261116
- log runs/sac_long_20261116.log
- supervisor status runs/sac_long_exit.json (written only after trainer exit)
Online W&B authenticated and remotely read back: run i00yr0fg was running,
with the3072-transition history row visible through the API. Local progress
had reached6144 transitions; updates0 during the prescribed30720-transition
warmup. This is startup evidence, not learned performance.

[Online run](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/sliceit-supermimic/runs/i00yr0fg)
