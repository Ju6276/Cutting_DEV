# Historical fixed-task evidence archive

Updated 2026-11-10. This document connects saved measurements to claims. It is
not a new experiment list or manuscript skeleton. [PAPER](PAPER.md) describes
the algorithm; [PLAN](PLAN.md) is the only current task list;
[BENCHMARK](BENCHMARK.md#task-definitions) defines outcomes and forces;
[RESULTS](RESULTS.md) retains all historical measurements.

## Historical scope and present use

**Decision, 2026-11-10:** retire action-perturbation evaluation and exclude its
results/figures from the paper. They remain below solely as an auditable record,
not as a proposed paper argument or a queue of experiments to repeat. Figures
1–2 and the 256-case tables are excluded. The saved folder name
`paper_evidence_20261110` is historical and does not make every artifact current
manuscript evidence.

Nominal execution, provenance and the BC-only comparison remain development
evidence. The new shared multi-condition training direction is in PAPER/PLAN;
the old fixed-prism checkpoints do not answer that question. The unexecuted
ten-episode frozen-material pilot remains withdrawn.

The method learns robot motion and sensor-force response from physical
interaction, learns values from physical-trajectory lambda returns, and updates
the policy by short BPTT rollouts with recorded-action behavior regularization.
The purpose is to improve an executed compliant-cutting controller using a
learned model. Deployment uses the policy and FDCC, without an online planner.

This is the algorithm being investigated, not a claim that its individual
components are new. SuperMimic-style BPTT, lambda returns, short rollouts and BC
have precedents. The cutting-specific state/action interface and connected
learning procedure must be justified by control evidence, not by counting
components or bugs fixed. A remains a completed mechanism experiment.

| Historical observation | Saved evidence | Scope of the statement |
|---|---|---|
| The working continuation improves its starting controller under the specified disturbances. | 213/256 vs 191/256 successes; paired difference +8.6 percentage points [4.7, 12.5]. | These checkpoints and cases. Continuation changes data, critic, horizon and regularization; it does not isolate model gradients. |
| The working policy has lower material peak force than the existing SAC85k controller. | All-case mean peaks 29.7 vs 45.4 N; paired difference −15.7 N [−17.3, −14.0]. | SAC succeeds more often: 228/256 vs 213/256. Do not turn this into a safety or overall-superiority claim. |
| The learned policy executes this task with much less online decision computation than the tested same-WM MPPI. | About 1.02 ms per policy call vs 491–532 ms per MPPI call; policy completes, each MPPI configuration succeeds for 1/3 planning seeds. | Recorded implementations and settings. Physics is separate; timings are not a controlled throughput benchmark. |
| Better prediction need not translate into a detectable final control gain. | A improved held-out H8 tip prediction, but its pre-stated final perturbed-success interval includes zero. | The completed A experiment, its legacy raw-output disturbance rule and single training seed per arm. |
| The method transfers across material conditions. | Not established by these results. | Proposed shared multi-condition training is in PLAN; no such result exists yet. |

The present package supports a functioning learned control method and a
checkpoint-level empirical study. It does not yet establish a new general
algorithm, training-seed robustness, sample efficiency, real-robot safety,
multi-material learning, or arbitrary cut/topology generalization.

## Archived perturbed physical execution — excluded from the paper

Each controller ran the same 256 predefined executed-action disturbance cases
to its own termination. No cases were discarded or replaced. All task-timeout
counts are zero.

| Frozen controller | Goal / force limit / workspace | Mean material peak (N) | Mean material impulse (N·s) | Mean sensor limit utilization | Mean steps, successful cases only |
|---|---:|---:|---:|---:|---:|
| policy62 | 191 / 22 / 43 | 32.91 | 7.05 | 0.634 | 23.06 |
| Working WM + BPTT | 213 / 20 / 23 | 29.67 | 7.07 | 0.573 | 20.86 |
| Unified SAC85k | 228 / 16 / 12 | 45.36 | 6.89 | 0.894 | 19.48 |

Successful-case means use different subsets; paired comparisons below use the
intersection. Success here is the goal-pose criterion, not verified material
separation. Sensor utilization and material force are different signals.

| Working minus comparator | policy62, paired 95% interval | SAC85k, paired 95% interval |
|---|---:|---:|
| Success (percentage points) | +8.59 [4.69, 12.50] | −5.86 [−10.55, −1.17] |
| Force-limit endings (percentage points) | −0.78 [−2.73, 1.17] | +1.56 [−1.95, 5.47] |
| Material peak, all cases (N) | −3.24 [−3.61, −2.86] | −15.69 [−17.29, −14.01] |
| Material impulse, all cases (N·s) | +0.02 [−0.16, 0.23] | +0.18 [−0.15, 0.54] |
| Steps, both succeed | −1.73 [−2.57, −1.15], n=188 | +1.42 [0.79, 2.53], n=202 |
| Material peak, both succeed (N) | −3.97 [−4.27, −3.63], n=188 | −19.72 [−20.88, −18.46], n=202 |

Intervals are the already saved case-paired bootstrap intervals. They quantify
case variation for these frozen controllers, not variation across training
seeds. Intervals containing zero do not establish equivalence. Lower peak force
also does not mean lower mean force or total impulse: compared with policy62,
the working policy's all-case mean force is slightly higher.

## Unperturbed execution and computation

The package retains all six working-WM MPPI runs: three planning seeds for each
of two terminal-value settings. Each setting has one success and two force-limit
endings. These are planner random seeds, not independently trained models.
Short failed episodes are not fast successful cuts.

The reference succeeds in 40 steps, working policy in 19, policy62 in 20 and
SAC85k in 18. The reference uses a separate trajectory-control branch, so this is
a system reference, not an action-matched algorithm ablation. Zero action reached
323/500 steps before its computation budget ended; its 16.15 s simulated duration
and 35.92 N·s impulse describe only an unfinished prefix.

The saved SAC report contains eight deterministic repeats, with overlapping
points in the figure. It does not supply eight training seeds. Policy and MPPI
reports in this package contain one episode per listed controller/seed.

Decision calls and physics stepping are reported separately:
working policy 1.02 ms/call, MPPI 491–532 ms/call, task period 50 ms.
The coupled physics costs roughly 2–4 s per executed action in these reports.
SAC's recorded 3.61 ms/call includes a Python-process bridge. Different report
dates, execution paths and loads preclude interpreting these numbers as a
controlled hardware speedup experiment or claiming real-time simulation.

## Controller provenance and training exposure

| Controller | Input / training identity | What is and is not known |
|---|---|---|
| policy62 | Ten 24-D frames + goal; source iteration 1561, actor updates 62, WM updates 562; 1024 training worlds | The complete physical-transition count across resumes has not been audited. |
| Working policy | Same policy interface; iteration 2421, actor 390, WM890; real critic, H8, BC 0.25; fresh critic + 100 pre-fit rounds | Includes the source warm start. The later same-configuration collapse is retained in RESULTS. |
| SAC85k | Current 24-D observation, already including source wrench-history fields; one environment, lr 0.01, batch 1024, Wave/tanh | 85,000 environment steps at the selected checkpoint. Selected after observing later collapse; run stopped at 95,037, before the source 100,100 endpoint. |
| WM + MPPI | Working checkpoint's WM/control history; H16, 512 candidates, four refinements, temperature 0.1, noise 0.5, gamma 0.99 | No new optimizer updates. Critic/no-critic are separate configurations. Development settings were chosen on WM563 then transferred unchanged. |
| Reference / zero action | Prescribed trajectory / constant six-action command | No learning. Distinct action interfaces remain labeled. |

SAC85k retains source SAC settings in our unified public-prism environment; it is
not the independent Gazebo run or a full reproduction of SliceIt's experiments.
The known counts are recorded without inventing a matched-data curve.
See the generated `controllers.json` for report and checkpoint identities,
and `manifest.json` for input-file hashes.

## Figures, captions and reproduction

All artifacts remain on GPU7:
`/dataHDD/dong/Experiments/Cutting_DEV/runs/paper_evidence_20261110/`
(the canonical repository's `runs/paper_evidence_20261110` points there).
Open its `index.md` in Remote SSH Markdown preview. PNG and vector PDF versions
are available. No weights, data, figures or source copy is stored on the Mac.

1. **01_perturbed_outcomes_and_force — archive only, excluded from the paper.** Task endings and all-case ECDFs for
   material peak, material impulse and sensor limit utilization. The line at one
   belongs only to the sensor-utilization panel; no material-force threshold is
   implied. Fixed checkpoints, 256 shared cases each.
2. **02_paired_checkpoint_effects — archive only, excluded from the paper.** Working-minus-comparator differences and
   saved 95% paired intervals. All-case n=256; both-success n=188 or 202. The
   intervals exclude training-seed uncertainty.
3. **03_unperturbed_execution.** Each recorded episode's elapsed simulation
   time, material peak and impulse, labeled by ending. Time is logarithmic to
   retain both early failures and the long zero-action prefix. MPPI seeds and
   critic settings are separate; the eight SAC repeats overlap.
4. **04_recorded_online_cost.** Recorded controller-call cost and physics cost
   separately. SAC includes its process bridge; report dates differ. The task
   period is a reference for decision latency, not evidence of real-time physics.

From the canonical repository, using the existing CPU plotting environment:

```bash
/dataHDD/dong/Projects/SliceIt/.venv/bin/python scripts/build_evidence.py \
  --output runs/paper_evidence_20261110
```

The output directory must not exist; use a new directory for a repeated build.
Python >=3.8, numpy and matplotlib are sufficient; no simulation or learning
module is imported. The JSON recipe is
[configs/analysis/evidence_20261110.json](../configs/analysis/evidence_20261110.json).
The package covers 768 recorded perturbed episodes and 18 unperturbed records.
It includes raw-case CSV, paired-effects CSV, controller metadata, captions and
SHA256 provenance. It checks saved means and paired subsets against raw records;
it does not rerun the bootstrap, simulation or training.
