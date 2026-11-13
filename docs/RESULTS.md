# Results

Current task: [free_cut_v3](TASK_FREE_CUT.md). This file contains the current
November 13 training/diagnostic record, with corrections beside the affected
claims. [PLAN](PLAN.md) owns the authorized continuation; dates here are evidence
snapshots, not a live process monitor.

Earlier measurements, negative results and retractions are preserved intact in
[November 11–12](archive/RESULTS_20261111_20261112.md) and
[November 2–10](archive/RESULTS_20261102_20261110.md). These use different task and
physical revisions and must not be pooled with free_cut_v3.
[Archive and Git recovery index](archive/README.md).

## 2026-11-13 — free_cut_v3 short training-path validation (free cucumber, single object)

Engineering check of the production training path; not a learning result. Config
`configs/free_cut/short_train_cucumber.json` (256 worlds, S0 material randomization, original
simulation limits, 180-step horizon, v3 reward unchanged). `train_gpu.py` with compile, critic `real`,
actor horizon 8, BC 0.25, 500 WM-only updates (unchanged). Run 1: 3.6 h cap
(`runs/free_cut_short_20261112/training`), then resume from its `last.pt` with replay for 2.3 h
(`.../training_resume`). GPU7 RTX 4090. During resume, a verification job shared the GPU for about
17 minutes (00:55–01:12 Berlin, 2026-11-13); resumed timings are not exclusive-GPU measurements.

**Chain.** Collection, per-material re-settling resets, terminal settlement, WM pretraining, transition
to joint training, 43 actor/critic updates, periodic checkpoints (250 ... 1250, `wm_only_complete.pt`,
`first_joint_update.pt`) and resume with replay all ran. 543 update records, no non-finite value, no
traceback. 2425 episodes: success 1026, timeout 1399, force limit 0, workspace 0; every success has
complete separation and no separated episode is recorded otherwise. Peak limit utilisation 0.74;
the original limits were never reached (data coverage, not adjusted).

**Time and memory.**

| Stage | s per vector step (256 worlds) | Share in run 1 |
|---|---:|---|
| WM-only (iterations 40-1330) | 14.5 | physics 21 %, reset 16 %, WM updates 63 % |
| Joint (from about iteration 1330) | about 19.4 | adds actor/critic updates |

WM update about 24 s compiled (104 s eager in the earlier profile). GPU memory 2.8 GB in the WM-only
stage, about 7.8 GB once actor/critic were compiled. Estimate for one object, 5000 iterations: about
5.2 h to the first joint update plus about 20 h of joint training (about 1500 policy updates), i.e.
about 25 h. Three objects (three worker processes on one GPU, potato at 8e-6 s) are not measured and
must not be extrapolated linearly.

**WM progress and completion.** Fraction of sampled WM transitions that are separations: about 0.2-0.3 %.
Missed completions among them (head probability < 0.5): 100 % at WM update 61, about 80 % at the
end of WM-only training, 73 % at update 543. False completion among incomplete transitions: 0.07-0.09 %
at the end. In imagination after 43 policy updates the mean per-step completion probability was 0.11
with predicted progress 0.71 and 8-step survival 0.55. These are different
states/actions/statistics from the replay separation fraction; comparing 11% with
0.3% is **withdrawn as evidence of model exploitation**. In v3 completion probability
is predicted release fraction times classification-head probability. A low total can
come from either factor; same-trajectory measurements below distinguish them.

**Issues in this historical run (runtime fixes below).** A resume restarts the 40-iteration
update warmup (counted from the resume iteration). A WM metric ratio is logged as 1.0000026 (float
accumulation in weighted minibatch means). The W&B offline exporter logged transient "Bad Gateway"
retries. Timeouts end at a median progress of 0.85 under the initial/early policy.

#### 2026-11-13 — Deterministic evaluation of the two short-run checkpoints; corrections

Evaluation (`scripts/evaluate_multi.py`, `configs/free_cut/eval_cucumber.json`: training distribution,
seed 100, 64 worlds x 2 episodes; reports `runs/free_cut_short_20261112/eval_{first_joint,last}`):

| | `first_joint_update.pt` (1 policy update) | `last.pt` (43 policy updates) |
|---|---|---|
| Separated (task success) | 24/128 (19 %) | 88/128 (69 %) |
| Timeout | 104 (81 %) | 0 |
| Workspace | 0 | 40 (31 %), every one at step 52 |
| Force limit | 0 | 0 |
| Median episode length | 180 steps | 7 steps |
| Median speed error | 20 mm/s | 131 mm/s |
| Peak limit utilisation, median / max | 0.32 / 0.52 | 0.44 / 0.77 |
| Mean undiscounted return | 9.8 | 36.6 |

For reference, the fixed press at the commanded speed returned about 110 in the physical reward check.
After 43 updates the policy separates the object in about 6 steps (far above the commanded 1-5 cm/s);
the workspace endings had released about 95 % of the springs. Inference, not verified: the sawing
output keeps one sign, and the mapper integrates it into the target offset (about 2 mm per step), so
the target leaves the 10 cm half-width of the workspace after about 50 steps. 43 updates do not
judge the method; the cause of the behaviour change is not established (WM completion/progress
prediction, critic and the integrated lateral offset are all candidates). The same-trajectory check below subsequently recorded the lateral accumulation;
its causal contribution to learning failure is not isolated by this observation.

**Comparison scope.** The 128 episodes include second episodes whose randomized
conditions are not matched across checkpoints; do not pair all 128 by episode number.
The fixed-press return is a reference reward check, not a matched 128-case baseline.
The first-joint file already contains one policy update.

## 2026-11-13 — Same-trajectory prediction check of the two short-run checkpoints

`scripts/diagnose_free_cut_predictions.py`: deterministic policy, one episode per world on
`eval_cucumber.json` (the same 64 conditions for both checkpoints), every step recorded; then the
checkpoint's own WM, critic and policy are evaluated on those recorded states and executed actions.
Reports: `runs/free_cut_short_20261112/same_trajectory_{first_joint_update,last}/report.json`
(+ steps.npz), recorded at `11f60a8`. These are historical diagnostic numbers:
`81688ee` and `12b951f` subsequently corrected hard-failure absorption,
same-step failure priority and raw next-state recording. In particular, the
8-step cumulative completion row must not be treated as a result of the corrected
diagnostic. The current continuation uses the corrected script and separate outputs.

| | first_joint_update (1 update) | last (43 policy updates) |
|---|---|---|
| Episodes: success / workspace / timeout | 13 / 0 / 51 | 43 / 21 / 0 |
| 1-step WM error: progress / force component / EE position | 0.006 / 0.15 N / 0.12 mm | 0.09 / 1.15 N / 2.5 mm |
| Completion probability on separating steps: total (release fraction x head) | 0.04 (0.26 x 0.14) | 0.59 (0.61 x 0.97) |
| Completion probability on non-separating steps; share >= 0.5 | 0.0002; 0 % | 0.71; 74 % |
| 8-step cumulative completion when not separated within 8 steps | 0.001 | 0.999 |
| WM predicts the workspace exit in the last 8 steps before a real exit | (no exits) | 8.9 % (9.4 % when there is no exit) |
| Critic minus realised return-to-go, workspace episodes | - | +294 |
| Imagined actor objective vs realised return-to-go, workspace episodes | - | 97.7 vs -102.9 |
| Same, successful episodes | 4.3 vs 95.9 | 98.6 vs 100.6 |

- All 21 workspace endings: x axis, step 52, the sawing output had one sign at every step (mean
  |a4| before clipping 1.42) and the integrated target offset reached -102 mm, i.e. the lateral
  workspace half-width. The cut had stalled near 95 % release while the knife slid along the blade.
- On these trajectories the WM gives a high completion probability where the cut has not separated
  (mostly the stalled workspace episodes), does not predict the lateral exit, and the critic values
  them far above any achievable return (+294 above the realised return, above the 110 maximum).
  Inside the model these episodes therefore score like successful cuts (97.7) while they end at -103.
- Each row uses its own checkpoint's WM, policy and critic. The early policy has
  one update; it is not an untrained policy tested with an identical WM. Fourteen
  of 23 WM tensors differ between checkpoints. Small early trajectory state errors
  also do not establish accurate completion detection: its separating-step
  probability is only 0.04.
- Overall physical success and mean return improved across these two checkpoints.
  The workspace-failure subset nevertheless has optimistic predictions and values.
  The 97.7/-102.9 pair is a mean over recorded states' return-to-go, not two initial
  episode returns. These data alone do not prove causal model exploitation or that
  further data cannot correct the error.

## 2026-11-13 — Runtime repairs and bounded joint continuation

Runtime repair `b1bcb92` fixes replay-resume warmup and update-schedule persistence,
evaluation cleanup on exceptions/signals/timeouts, and integer-count completion
metrics. It changes neither reward, network, action mapping nor optimization
objective. Its recorded verification: root 344 passed / 5 skipped; backend 38
passed. Claude's base environment reproduced 342 / 7 (two optional MPPI checks
skipped). Artifacts: `runs/infra_fix_20261113/{root_tests.log,backend_tests.log,baseline_equivalence.json}`.
[Exact repair handoff](https://github.com/Ju6276/Cutting_DEV/blob/b1bcb92/docs/FREE_CUT_INFRA_REVIEW.md).

The authorized continuation runs from the separate `Cutting_DEV_infra` checkout
at `b1bcb92`. It resumed iteration 1441 with policy updates 43 and 256000 replay
transitions; later confirmation recorded iteration 1449 and policy updates 45,
without another 40-step warmup. The old checkpoint lacks the new schedule
origin/phase/config fields, so `legacy_defaults_origin0_phase0` is explicitly
recorded. This restores networks/optimizers/replay, not unknowable historical
phase or exact physical trajectories. New checkpoints persist the schedule state.

Budget and checkpoint rules are in [PLAN](PLAN.md). Training:
`runs/free_cut_joint_20261113/training`. Corrected diagnostics:
`runs/free_cut_joint_20261113/eval`, script revision `12b951f`. They use the same
64 first-episode development conditions at every node, with no new acceptance
thresholds. Critic training-target versus realised-return comparison remains to
be added in the diagnostic summary; historical optimization targets must not be
confused with targets recomputed using a later checkpoint.

No continuation trend or final result is asserted here before reports are read.
