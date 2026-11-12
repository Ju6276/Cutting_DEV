# Free-cut review repair, 2026-11-12

**Latest: the second-review v3 repair below supersedes v2 as the current candidate.**
The v2 reward table is historical; its tiny high-speed margin was not training-ready.


## Latest handoff: free_cut_v3

Code: 0ef9718 (reset cache/timing), c720cd9 (v3 reward and completion), abe84b2
(stable completion gradients and actual emitted-probability metrics). Review branch remains
codex/free-cut-v1; not merged into main, and no long training was started.

- Reward: physical normal / 3x / 20x / completed-slow returns are
  109.82 / 105.21 / 94.30 / 99.62. All nine reference cases finished; no weak .08 speed margin.
- Completion: conditional release-fraction hazard bound, shared reward/survival, and direct
  sigmoid fractions on the BPTT path to avoid float32 subtraction instability near rho=1.
- Reset: bounded scratch cache with immediate object release; .75 s material-specific settling
  and inactive-world preservation unchanged. Reset remains a measured cost, not free.
- Actual execution: 32 staggered iterations x 256 worlds, WM/actor/real-critic updates, full-size
  eager update profile, all cache sizes, and an isolated real sensor-limit abort.
- Final regression: root 310 passed / 7 skipped; backend 38 passed; MPPI environment 12 passed.
  Root's two missing-MPPI skips are exercised in that dedicated environment; the remaining
  five are optional historical SAC/Gym/TensorFlow checks, not claimed as passing.

Observed whole-device memory in the full-size eager profile was 2892 MiB. Physical/reset
means were 3.025 / 5.473 seconds per vector step in the staggered stress; the source-sized
WM update took 103.687 seconds, actor/real-critic update 26.134 seconds. These are a
one-object eager workload, not a compiled three-object training budget.

Specific remaining boundaries: completion calibration and useful policy learning require
training data/validation; nominal trajectories did not naturally hit the original force
limits; the lower-limit probe checks wiring only. No physical calibration or algorithmic
advantage is claimed. Formulas: TASK_FREE_CUT section 9. Full measurements below; raw
reports are on GPU7 under runs/free_cut_second_review_20261112/.


Authorized by the user after Claude reviewed 48fcbef. No long training.

One connected repair: reduce material-specific reset cost without changing the distribution
or settling duration; conservative progress initialization; completion/progress diagnostics;
finite-horizon gamma=1 candidate and observable time; explicit workspace; correct logs.

Bounded validation:
- Reset benchmark: compare eager and optimized settling at 1 and 256 worlds, including
  sparse resets and independent materials. At most 20 GPU minutes including compilation.
  Preserve inactive worlds and compare material states/contact checks, report timing.
- Physical reference trajectories: same free cucumber/material/friction/limits as the prior
  check, gamma=1 candidate. At most 45 GPU minutes; record all outcomes, including incomplete
  trajectories. No tuning physics or thresholds to force reward ordering.
- Actual small WM/actor/critic update on recorded transitions, at most 10 GPU minutes.
  Demonstrates executable gradients only, not model accuracy or control success.
- Required root/backend regression suites. No new experimental claims.

Task changes are versioned in metadata; historical free_cut_v1 and source defaults remain
replayable. Reward remains provisional until the recorded trajectory preference checks
support its intended behavior. Failure to pass is reported, not hidden by test counts.


## Changes ready for review

- New opt-in free_cut_v2; v1/source still supported. Progress starts at .001 of the remaining
  fraction, with a finite nonzero gradient. Physical residual outputs still initialize to zero.
- gamma=1 for the finite-horizon candidate; actor/critic share it, MPPI rejects a mismatch.
  The policy now observes remaining time (25 observations, still an 8-value goal).
- Explicit robot-base EE workspace instead of old target-relative pose bounds. Shared
  kinematic observation derivation avoids the duplicated state-update implementation.
- Imagined event metrics no longer use pose success. Completion Brier and confusion
  fractions in fitting use recorded physical separation. A separate multi-step report exposes
  high completion probability at low predicted progress; it does not hide it with a rho gate.
- Cached scratch batches and stationary CUDA graph settling. Material/duration/limits are
  unchanged; only resetting rows are integrated and copied back. The old eager advance
  remains available for reference comparisons.

## Actual reset measurements

GPU7 RTX 4090, .75 s per-material settling; no concurrent GPU job for these timings.
E varies from 1 to 2 in the 256-world batch. Times exclude importing/compiling dependencies.

| Worlds / resetting | Old eager | Warm optimized | Ratio |
|---|---:|---:|---:|
| 1 / 1 | 21.06 s | 4.43 s | 4.75x |
| 256 / 1 | 25.66 s | 4.43 s | 5.79x |
| 256 / 256 | 25.47 s | 16.03 s | 1.59x |

First graph creation is included in separately recorded first-use times (4.51, 4.49, 16.19 s).
Inactive live rows are bitwise unchanged. This is a reset timing result, **not** an end-to-end
training speedup. Random resets still cost real simulation time; do not budget them as free.

Numerics: the initial benchmark's maximum position differences were 8.5 um (one world)
and 180 um (256 worlds), with identical spring stiffness. The new 4 ms per-particle velocity
test initially failed at 0.00104 m/s. A repeated eager control also differed by 0.00106-0.00128
m/s, whereas graph/eager differed by 0.00088-0.00100 m/s in that control; position differences
were 45-52 nm and stiffness was unchanged. Consequently the new execution-equivalence test
compares one **complete 32-substep graph replay**, not the eager remainder path. Existing
backend tests were not weakened.

An independent .75 s / 256-world repeat measured eager/eager position difference 0.131 mm;
graph/eager differences were 0.255 and 0.137 mm. Velocity differences were about .0041 m/s
for both comparisons, spring stiffness identical. These establish non-bitwise repeatability;
they do **not** prove distributional equivalence of long cuts or tight force-peak reproducibility.
No damage rule was changed to reduce the differences. Raw results are in reset_repeat_short.json
and reset_repeat_long.json under the run directory.

## Real update execution check

Collected 32 transitions through the actual six-action controller in two worlds, with a
16-step test deadline. These were **not** reference trajectories relabelled as zero actions.
One WM minibatch and one BPTT actor/real-critic update executed with finite gradients.

- WM loss .7366; gradient norm 3.4828.
- Actor gradient norm .8472; real-critic loss .00405.
- Explicit verification sizes: WM window 8, minibatch 4, one minibatch; actor horizon 8,
  minibatch 4, one critic epoch/minibatch; full production-width networks, compile disabled.
- This intentionally tests one update, not the default 500-update pretraining protocol.
  It produces no reusable trained policy/checkpoint.

The new diagnostic correctly flags the untrained completion head: physical sampled transitions
were incomplete but its probability was about .513; Brier .265 and probability-minus-progress
about .512. One update is not used to judge learnability or redesign the method. Completion
calibration remains something to check on physical data after WM pretraining, before trusting
imagined policy returns. There is still **no hard architectural consistency guarantee**.

## Regression

- Root suite: 303 passed, 5 skipped; skipped optional legacy integrations remain explicit.
- Subsequent affected v2/benchmark/MPPI tests: 29 passed, including the extra v2 gradient test.
- Backend suite after the execution-test correction: 37 passed.
- No long training. Work stays on the review branch until reviewed.

## Evidence locations

On GPU7: /dataHDD/dong/Projects/Cutting_DEV/runs/free_cut_repair_20261112/.
Reset report: reset/report.json. Update report: update/report.json and diagnostic_batch.pt.
All outputs stay on the server. See RESULTS for the completed reference-trajectory results.


## Reward settlement bounds for this exact candidate

For gamma=1, H=100 and dt=.05, the three bounded running penalties sum to at most
100*.05*(1+1+.1)=10.5. Progress telescopes and stays within [-10,10] for any bounded
initial/final rho (and is nonnegative for the monotone damage used here).

Consequently: successful return is at least 89.5; any unfinished, non-failing return is
at most 10; a force/workspace failure has return at most -90. Even allowing arbitrary
bounded progress differences, a non-failing timeout has return at least -20.5. Thus,
with the implemented single terminal settlement and failure priority, deliberately failing
cannot improve return over an unfinished non-failing trajectory for **this configuration**.
This is an algebraic property of the exact finite reward, not a claim that the model
predicts those events correctly. It also does not prove commanded-speed preference among
different successful cuts; those returns are checked on physical reference trajectories.


## Physical reward preference results

Same free cucumber, nominal material, .75 s settling, net wrench, friction transition .01,
simulation limits [50,50,50,5,5,5], gamma=1, unchanged reward weights. All values are actual
physics executions. Fixed reference motion bypasses the six-action mapping, so these
trajectories are reward probes, not learned policies or training data labelled with zero actions.

| Run / trajectory | Ending | Steps | Return | Peak sensor-limit utilization |
|---|---|---:|---:|---:|
| references_v2 / press | success | 43 | 109.8194 | 0.0630 |
| references_v2 / saw | success | 43 | 109.8055 | 0.0610 |
| references_v2 / fast | success | 17 | 109.2845 | 0.1343 |
| references_v2 / stall | timeout | 100 | -3.8080 | 0.0000 |
| references_v2 / retract | timeout | 100 | -1.7781 | 0.0290 |
| references_v2 / slow | timeout | 100 | 0.4490 | 0.0758 |
| references_v2 / force_attempt | success | 5 | 109.7352 | 0.1432 |
| profiles / press | success | 43 | 109.8114 | 0.0903 |
| profiles / slow_press | success | 94 | 107.5613 | 0.0612 |
| profiles / hold_in_material | timeout | 100 | 0.0294 | 0.0614 |

Within each run the cases share the same saved settled reset. The second run includes its own
press reference. The candidate meets the commanded-speed ranking on the **observed cases**;
it does not prove global reward optimality or generalize the force numbers to other conditions.

The force_attempt probe used a 20x commanded reference speed and continued beyond the nominal
endpoint, but separation occurred in five steps without a force event. It is recorded as a
fast successful cut, not an intentional force-failure test. Its return is lower than normal press,
but the margin is small (about .08). No physical force-failure case was obtained in this check.
slow_press also stayed far from the force limit; it is a slow-press test, not a validated
near-limit slow-press example. No force limit/material was adjusted to manufacture these cases.
The force/workspace penalty precedence and single-settlement paths have regression tests.

Harness correction is explicit: the first references run stopped after one case because a
leftover cleanup deleted the reused environment. In references_v2, seven valid cases were
retained; its last two planned profiles would switch only at the normal endpoint. They were
stopped and replaced by profiles, where slowdown/holding starts halfway through the original
travel. The corrected script records the prescribed velocity per transition. Initial attempt
logs and the script snapshot are preserved; no invalid profile is counted as passing evidence.

The check outputs took 1108 s for the seven-case batch and 594 s for the corrected profile batch.
No long training was started. This review does not provide a final multi-object training budget:
the 100-step deadline and workspace box are **review inputs**. A future training configuration
must cover its stated travel distances and commanded speeds; do not silently reuse this small
test horizon for every object.

## Handoff

Code commits: 1b287d5 (reset execution), 5362975 (free-cut v2 and diagnostic interfaces).
Review branch: codex/free-cut-v1; not merged into main.

For Claude: inspect the execution-stream/cache changes, explicit v2 interface/discount,
the real-label diagnostics (which are not a hard consistency constraint), and the recorded
reward ordering. Next is a bounded WM-pretraining / engineering-training proposal after review,
not another series of local numerical diagnostics or a revived S3 experiment.


## Second review repair (bounded validation, user authorized)

Claude correctly identified that v2 only ranked the sampled speeds, with an insufficient
0.08 return gap for the 20x-speed trajectory. It also had diagnostics but no completion
consistency constraint. This section supersedes any implication that v2 was training-ready.

Connected scope: explicit v3 reward and completion parameterization; bounded reset cache;
measure actual short-loop throughput and memory. Do not change materials, friction, force
limits, settling duration, or launch long training.

Bounded checks on GPU7 (sequential to avoid timing interference):
- Recompute v3 rewards on the existing stored physical traces, then repeat reference cases
  using the unchanged nominal physical setup, <= 35 GPU minutes.
- 32 real-control iterations with 256 worlds and full-width networks, real WM/actor/critic
  updates with explicit small review batches, material-randomized resets; <= 20 GPU minutes.
  A 16-step review deadline stresses resets; this is not the production training horizon,
  update budget or evidence of learning. Record per-stage time and allocated/reserved/device
  memory; visit all power-of-two reset sizes and exercise a full-duration sparse reset.
- Isolated physical force-abort wiring check with explicitly LOWER TEST-ONLY limits, <= 5
  GPU minutes. This is not evidence that the nominal 50 N / 5 Nm task is challenging, and
  its results must never be pooled with the unchanged-task references.
- Required root/backend suites including GPU cases. No statistical performance claims.


### Second-review implementation

- `0ef9718`: reset scratch capacity is bounded by one rounded-up parent material batch.
  Eviction breaks the cached Python step's self-reference; a weak-reference regression
  proves evicted scratch objects release immediately. Training logs separate reset time.
- `c720cd9` and `abe84b2`: opt-in free_cut_v3; old v1/v2 stay reproducible.
  Speed cost is paid per released fraction plus a bounded time cost. Exact formula and
  provisional weights are in TASK_FREE_CUT section 9. Physical limits/materials are unchanged.
- The finite cost bound is checked for every accepted v3 configuration. With H=100,
  dt=.05 and the review weights, successful return >=54.5, unfinished return in [-45.5,10],
  and force/workspace failure return <=-90. This excludes intentional failure as a way
  to avoid a larger accumulated penalty under the stated task signals.
- Completion hazard is bounded by this step's predicted fraction of remaining material
  released, not just total progress. Consequently constant partial progress cannot
  accumulate an arbitrary completion probability. Expected success reward and survival
  use the same probability; head-only fitting detaches both progress inputs and the
  fraction, while policy BPTT retains them. No new spring-release threshold.
- A float32 diagnostic demonstrated why the original network fraction must be used:
  reconstructing f from two rounded progress states, with true f=.9 and old rho=.99999994,
  produced recovered f=1 and a spurious d(f)/d(rho) of 1,677,722. At rho=1 it reached
  -7.66e37. The differentiable path now uses the network sigmoid directly; state-only
  reconstruction refuses state gradients. Tests cover the endpoints and near-complete
  values. Imagination logs retain the actual emitted probability rather than reconstructing it.
- The third-party current-stream/cache changes were from 1b287d5, not newly added here.
  METHOD explicitly lists their files and forward/backward/global scope. No new physical
  kernel, fracture, contact or friction modification was made in this review.

The completion constraint is an internal probability/expected-progress consistency bound,
not proof that the WM predicts real events correctly. Short update checks cannot establish
calibration, convergence or a useful policy. Eager one-object timing cannot promise memory
or wall time for compiled three-object training on another GPU.

### Harness provenance

The first timing attempt (`throughput`) failed before its first update because eight stored
rows did not yet provide a sampleable eight-transition window; it also overallocated replay
by treating capacity as total rather than per world. Both were in the new measurement script,
not train_gpu.py. The corrected harness uses the source 1000 rows per world and starts updates
at iteration 12. Logs are retained. Intermediate profiles used the earlier total-progress
completion bound; final implementation numbers must come from `throughput_final`.
All short profiles use an explicit 16-step deadline for reset stress and small early update
batches. The final full-size update is separately timed; do not multiply the small-batch loop
average by production iterations to estimate a long run.


### Second-review physical reward results

| Trajectory | Ending | Steps | v3 return | 2 ms net-force peak |
|---|---|---:|---:|---:|
| press | success | 43 | 109.821 | 4.146 N |
| saw | success | 46 | 109.720 | 1.880 N |
| fast | success | 17 | 105.210 | 6.172 N |
| stall | timeout | 100 | -19.999 | 0.000 N |
| retract | timeout | 100 | -14.555 | 1.122 N |
| slow | timeout | 100 | -6.594 | 3.769 N |
| extreme_fast | success | 5 | 94.300 | 9.677 N |
| slow_press | success | 94 | 99.625 | 3.807 N |
| hold_in_material | timeout | 100 | -12.566 | 2.576 N |

All nine cases completed their task ending (success or timeout) within the measurement
budget. Normal press versus 20x press now differs by 15.52, rather than .08 in v2.
No original-limit violation was triggered in this nominal condition; the lower-limit
physical abort probe is isolated and is not task-performance evidence. The earlier
rescoring of stored v2 traces agrees in ordering but is not substituted for these reruns.


### Isolated physical abort probe

To test the actual 500 Hz sensor-event path, a separate TEST-ONLY config used limits
[.1,.1,.1,.01,.01,.01]; the task/reference config remains [50,50,50,5,5,5]. Normal pressing
terminated at step 18 with collision=True, success=False, outside=False, timeout=False,
rho=0. The first sensor torque component reached .0102196 Nm against .01 Nm (utilisation
1.02196); net material force peak was .116932 N. Episode return -100.26715, last-step reward
-100.05671. The one failure settlement was charged and the episode ended. This is a physical
wiring/termination test, not evidence of difficulty or risk under the nominal task limits.
Raw report: runs/free_cut_second_review_20261112/force_abort/report.json.


### Final-code throughput and memory

GPU7 RTX 4090, one free-cucumber group, 256 worlds, full-width networks, source replay
capacity 1000 rows/world. The reset stress deliberately starts worlds with staggered
remaining deadlines in a 16-step test horizon; it is not a task-performance experiment.
32 real-control iterations = 8192 physical transitions, 16 timeouts/reset rows each step,
512 ended episodes in total. Materials are redrawn before each .75 s settle. Actual WM,
actor and real-critic updates ran, with finite reported losses/gradients. All episode endings
were deadline endings; no learned cutting-success claim or completion calibration is made.

| Timing | Measured |
|---|---:|
| Physical vector step, mean | 3.025 s |
| Reset of 16 selected rows, mean | 5.473 s |
| Small-review-update loop, mean total | 8.724 s |
| Full-duration one-world reset after cache stress | 4.700 s |
| One source-sized WM update: 10 x 2048, window 8 | 103.687 s |
| One actor/real-critic update: actor 4096, h8; critic 5 x 4 | 26.134 s |

The first loop uses batch 32, one WM minibatch, one critic epoch/minibatch, updating every
four physical steps after its data warmup. The final source-sized update uses microbatch
64 and eager execution. They are intentionally reported separately. Neither the 8–9 second
small-update loop nor these eager timings are a validated compiled multi-object long-run
budget. At source cadence 0.4 updates/vector step, the full update costs must be amortized
in addition to physical/reset time; do not omit them.

At the full-update measurement: PyTorch peak allocated 1961.7 MiB,
reserved 2284.0 MiB; observed whole-device used 2891.8 MiB.
The cache stress visited all capacities 1,2,4,...,256 and never retained more than
256 scratch-world capacity. Inactive material rows stayed bitwise unchanged.
This measures one-object eager allocation. Three object workers, different meshes and
compiled graph pools can cost more and are not certified to fit by this result.

Evidence: runs/free_cut_second_review_20261112/throughput_final/report.json, per-iteration
metrics and source diff. Whole final profile took 438.2 s including initialization,
cache pressure and the full-size updates. It produced no trained-policy checkpoint.
