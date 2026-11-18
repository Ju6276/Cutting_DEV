# Current-task baselines — 2026-11-16

## SAC provenance and search

The user recalled a PyTorch parallel SAC and requested reuse if available, with
Stable-Baselines3 as the fallback. No cutting-connected PyTorch multi-environment
SAC trainer was located in the inspected branches, server working directories,
archived cutting checkouts or run records on GPU7, GPU9 and A6000.

Concrete findings:
- `62feee8` changes only `evaluate_perturbed.py`: parallel deterministic actor
  evaluation, not multi-environment learning.
- `21ffc54` and GPU9's uncommitted `Cutting_DEV_sac` changes add restoration
  to the original single-world TensorFlow/tf2rl trainer. Preserve those changes.
- `a130a48` explicitly lists parallel SAC as future work.
- Research-source archives contain generic PyTorch SAC implementations; no
  evidence connected them to this cutting environment or a cutting training run.
This is a search result, not proof that no other copy ever existed.

Per the user's explicit fallback, the current parallel baseline uses
[Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
2.7.0, a PyTorch implementation. The original SliceIt/tf2rl runner is retained for
historical reproduction; these are separately identified implementations.

## Shared task and inputs

All new entry points consume the current shared `pose_depth_v3` config (the
pose_depth_v2 task rules plus the software board-stop consistency fix):
same robot, finite mounted blade, FDCC action mapping, force limits, rewards,
board boundary, randomized episode conditions and final-position rule.
Success is pose-task completion, not proof of material separation.
No task thresholds are changed to make a baseline succeed.

SAC receives the same ten observable 34-dimensional frames and eight-dimensional
commanded goal as the main policy (348 values). It outputs six bounded actions.
It receives no particle state or material labels. Its actor/critic are SB3
two-layer 256-unit MLPs, not the SliceIt TensorFlow Wave/TCN architecture.
Learning rate defaults to 3e-4, automatic entropy coefficient, gamma from the
task. These are explicit baseline settings, not a claim of published SliceIt
hyperparameter reproduction.

The vector bridge saves terminal observations before independently resetting
ended worlds. Timeouts are genuine task terminals, with no timeout bootstrap.
Optimizer cadence is expressed as physical transitions per gradient (default 1);
environment count does not silently multiply the data/update ratio.
The interval is configurable and must be fixed before a formal comparison.
Model, optimizers, replay and update credits resume; physical episodes restart.
This is not exact restoration of the simulator or RNG state.

## Frozen WM + MPPI

`evaluate_pose_policy.py --method wm_mppi` uses the pinned upstream
[UM-ARM-Lab/pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi)
optimizer and a compatible frozen checkpoint. Optional terminal critic comes
from that same checkpoint. Task, observation, geometry and model-layout metadata
are checked; learned parameters must remain unchanged.

Each world has its own receding-horizon plan. Plans are evaluated sequentially
with shared model weights. Absorption, terminal reward, survival and discount use
the retained model-rollout implementation. Geometric depth is derived from
kinematics; no cut-spring progress is supplied. The planner receives joint state
and controller/history state as well as force history, so its information access
is not identical to the deployed policy's observation.

This is an adaptation of an open-source optimizer to our learned model, not a
reproduction of a particular published cutting-MPC method. A compatible
checkpoint is required; an early pretraining checkpoint only verifies interfaces.

## Nonlearned MPC

`--method analytic_mpc` uses the same MPPI optimizer without loading any network.
Its approximate dynamics use robot kinematics, the existing FDCC and action
mapping, coarse ideal joint-command tracking, and constant wrench over the
planning horizon. The wrench is the last sample in the supplied observation
history; the inherited history extraction is not the newest full sensor-queue
sample. The controller's internal state is reconstructed from measured q/qd at
each model step. Replanning uses new physical measurements every control step.

It does not model changing food contact, fracture or unknown material properties.
It cannot forecast contact-force spikes or claim constraint satisfaction.
Measured physical forces still trigger exactly the common task's force limits.
This baseline answers what this simple nonlearned receding-horizon approximation
can do; it is not a validated cutting-force model or a published algorithm replica.
Analytic substeps, horizon, candidates, refinements and wall cost are recorded.

## Commands

Use an isolated environment with `requirements-baselines.txt`, current prepared
assets, `PYTHONPATH=.:backends/mujoco` and the backend runtime setup.
Keep active training environments/checkouts unchanged.

```bash
# Small integration test; not a formal trained SAC comparator.
python scripts/train_sac_multi.py --conditions CONFIG.json \
  --worlds-per-object 2 --learning-starts 24 --batch-size 32 \
  --buffer-size 2048 --max-transitions 240 --max-hours 1 --smoke --output OUT

python scripts/evaluate_pose_policy.py --method sac \
  --checkpoint OUT/model.zip --conditions CONFIG.json --output SAC_EVAL
python scripts/evaluate_pose_policy.py --method wm_mppi \
  --checkpoint CURRENT.pt --conditions CONFIG.json --output WM_MPPI_EVAL
python scripts/evaluate_pose_policy.py --method analytic_mpc \
  --conditions CONFIG.json --output ANALYTIC_EVAL
```

All support the backend-selecting `--blademaster-config` alternative. A code
interface is not evidence of a completed physical test on both backends; see
the measured coverage below. Non-smoke SAC launches require online W&B, an
approved budget and comparison settings. No formal SAC long run is launched by
these commands; the separate bounded GPU9 run is recorded in
[TRAINING_SAC](TRAINING_SAC_20261116.md).

## Validation record

Workspace: GPU9 `/dataHDD/dong/Projects/Cutting_DEV_baselines_20261116`.
Outputs: `runs/baseline_validation_20261116/`. Source and data stay on servers.
The bounded validation is separate from GPU7/A6000 active long runs.

GPU9 physical validation completed on the three matched first-episode cases
(seed20261116, one world per object), with the unchanged180-step task:

| Controller | Pose successes / cases | Timeouts | Mean return | Physically separated |
|---|---:|---:|---:|---:|
| SB3 SAC smoke checkpoint,264 transitions |1/3|2|25.62|0/3|
| Frozen WM+MPPI |3/3|0|98.10|0/3|
| Nonlearned FDCC+MPPI |2/3|1|60.06|0/3|

These are integration outcomes, not algorithm performance comparisons.
All nine episodes finished with complete records; there were no force-limit or
workspace failures. The `cases.json` files are identical across controllers.
SAC trained on6worlds (2/object),240transitions/216updates, then restored and
added24transitions/24updates, yielding264transitions/240updates.
The SAC terminal observation, independent reset, timeout and replay semantics
also passed targeted tests.

Both planners used horizon4,32candidates,one refinement,no terminal critic.
The WM checkpoint filename `checkpoint_00000032.pt` encodes iteration32:
actual WM updates105 (1050optimizer steps),policy0,critic0. This is an early
pretraining checkpoint, not a mature policy model. Its weights remained frozen.
WM-MPPI finished potato/prism/cucumber at100/100/124steps, with324plans.
Nonlearned MPC used2analytic substeps, completed potato/prism at151/137steps
and timed out on cucumber at180steps, with468plans. Ended worlds stopped planning.
Mean planning times were0.1525s and0.1335s per world respectively, exceeding
the50ms control interval. Do not present these as real-time controllers.
Some regression tests overlapped the evaluation period, so wall-clock episode
times are not isolated throughput benchmarks.

The optional learned-critic MPPI path exposed a compatibility bug: pose error
received the eight-dimensional pose-plus-speed goal. It now receives only the
seven pose coordinates, matching the actor-training code. Targeted tests cover
current8D goals, critic evaluation and frozen critic weights. No separate
physical episode with the terminal critic was run.

Independent reviews of both planner paths found no additional blocking issue.
This validation itself ran all three methods on the current DiSECt-material task
only. Later one-world native-BladeMaster executions of WM+MPPI and nonlearned
FDCC+MPPI are recorded in the DP3 section below; they are single illustrations,
not a bounded BladeMaster evaluation.



SAC wall-budget probe: the first probe exposed that changing SB3's schedule
horizon does not stop its local learning-loop budget. The runner now raises a
dedicated stop at the next complete rollout/update boundary and saves normally.
A regression verifies that all counted transitions are inserted and their
scheduled updates finish. A physical expired-budget restore (`sac_wall_stop_v2`)
kept240transitions/216updates/replay240 and stopped with `wall_budget`, adding
no data. Initialization and an indivisible physics/update round are not forcibly
interrupted by this deadline. These are engineering budget semantics, not a
hard real-time process watchdog.

Final GPU9 regression: root577passed/3skipped; bundled backend41passed.
The three skips are optional original TensorFlow checkpoint/bridge/learner
checks, not the new SB3 or MPPI paths. First attempts in the isolated worktree
lacked historical test fixtures; those assets were copied before the final
full run. Active GPU7/A6000 training code and Python environments were unchanged.


Interrupted-save guard: model/replay payloads are first fully written to a
temporary directory, then individually replaced, with the checksum metadata
replaced last. Resume checks both payload hashes; evaluation checks the model.
A mixed-generation snapshot is rejected instead of silently resumed. This is
detection, not automatic rollback to the previous generation. Legacy smoke
snapshots without hashes emit an explicit warning. A regression corrupts replay
after a valid save and verifies rejection.

## DP3 pre-cut visual handoff — 2026-11-17

WM+MPPI and nonlearned FDCC+MPPI now have actual one-world native-BladeMaster
visual execution through evaluate_visual_cut.py. DP3 selects the midpoint from
visible depth; the planner is initialized after visual placement and updated
from robot/wrench feedback each step. No continual visual tracking.
SAC visual execution remains unimplemented. See VISION_ROUTE1 and RESULTS.
The WM trial explicitly transfers the exact policy66-paired WM566 from DiSECt;
nonlearned planning uses no learned weights. This demonstration is not a
matched-backend algorithm benchmark or calibrated material-aware MPC.
