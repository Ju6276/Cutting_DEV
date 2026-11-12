# Free-cut review repair, 2026-11-12

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
