# Collection-scaled multi-object restart — 2026-11-14

## Authorization and scope

The user requested at least 256 environments for both A6000 DiSECt and GPU7
BladeMaster, capacity/bug checks, corrections, then a fresh long-run restart.
The previous engineering launches were stopped with SIGTERM and final checkpoint
plus replay retained. They are not continued as an identical training setting.

- DiSECt old run: 192 worlds, iteration 802, 153984 physical transitions,
  304 completed WM rounds, no policy updates.
- BladeMaster old run: 3 worlds, iteration 552, 1656 physical transitions,
  204 completed WM rounds, no policy updates.
- SIGTERM arrived inside a WM round. Optimizer-step counters include completed
  minibatches of the interrupted round; fresh restarts do not reuse these weights.

## Why the settings change

192=64/object was a deployment probe size, not a capacity optimum. Three BM
worlds=one/object was an integration probe; its independent sequential material
solvers were not suitable as a many-world training implementation.
Both retained a vector-step update cadence despite their different data volume.
This was a configuration/design mismatch, not a failure of replay indexing.

The new collection schedule is opt-in (--schedule transitions); source defaults
and its tests remain unchanged. Before any fitting, collect 10240 physical
transitions. Thereafter earn one WM round per 640 NEW physical transitions.
One round still has ten optimizer minibatches, each of 2048 sampled windows of
up to eight transitions. Sampling is with replacement; sampled transitions are
not counted as new experience. These initial values preserve the approximate
data/update ratio of the earlier 256-world source setting, NOT an established
optimal ratio. The 3-world stale-data regime is retired.

Warmup must additionally cover at least window+1 vector steps, validated before
training. A fractional vector boundary contributes only its post-warmup
transitions. Integer credit is checkpointed. Resume from a complete vector-step boundary with replay restores cadence; deliberately empty replay refills first. Changed world count or schedule
requires a fresh run, not a silent continuation.
LR uses collected-transition count divided by 256 as its clock, with the
unchanged source cosine schedule length. Changing world count does not change LR
at a given transition threshold. Checkpoint-every remains measured in vector
steps. --max-transitions is an additional transition budget rounded up at most
one vector step; wall-clock cap remains authoritative.

The initial 500 completed WM rounds remain explicit, not a model-quality gate.
With the above cadence, their data threshold is about 330240 transitions, plus
vector rounding; joint learning starts only after those rounds. Source actor
objectives, effective batches, network widths and reward remain unchanged.
Full-six-axis policy wrench observation is a separate explicitly versioned
interface; the old selected-axis observation remains reproducible.

## Preflight and restart contract

Capacity testing must include real stepping, selected reset with unchanged other
worlds, current six-axis observation, replay allocation, WM+actor+critic gradient
updates, finite checks, checkpoint and continuation. Compare wall time and
whole-process GPU memory, not only Torch's allocator. BladeMaster requires real
batched material solving and independent world contact history, not a serial
loop relabelled as parallel.
Do not terminate or change other users' jobs. A6000 GPU1 is currently shared by
an unrelated job occupying about 44 GiB; capacity conclusions apply only to
available memory and measured conditions.

Each new long run is capped at 24 hours or 1280000 new physical transitions,
whichever first. Start from fresh networks, seed 0, critic=real, actor horizon=8,
BC alpha=.25, no added risk penalty. Record measured world count and exact
source/config/asset identity before launch. The two backends still differ in
material models and randomization; this is not an algorithm comparison.

## Results

Capacity and launch measurements will be appended after execution.


### A6000 material capacity (real physics, before full learning allocation)

Both probes used the original three-axis observation (physics is unchanged by
the later six-axis interface). Each included six physical steps, one selected
reset per object and another step; nonselected worlds were bitwise unchanged.
- 384 (128/object): initialize111.40s; step8.865–8.937s; selected reset26.77s.
- 768 (256/object): initialize191.46s; step13.199–13.317s; selected reset26.89s.
GPU1 was free of the unrelated norm-stat process by the end of the384 probe.
These probes did not allocate learner/replay and do not establish total training
VRAM or a maximum hardware capacity. The subsequent A6000 full-learning attempt was cancelled for resource contention; GPU7 checks below supersede the next-step plan.
Measurements are in the A6000 frozen checkout's runs/scale_preflight_20261114.


### Resource change and user decision

A6000's 768-world full-learning probe in learning768 was deliberately interrupted
during initialization after another user's openpi training occupied about44GiB.
progress.json records KeyboardInterrupt, NOT a diagnosed algorithm or physics bug.
No policy/WM update ran in that attempt. Its workers shut down and GPU allocations
were released. Do not reuse it as a successful learning-capacity measurement.
The user explicitly chose: validate both backends on GPU7; then launch DiSECt
main training there at a measured safe scale. BladeMaster long training is
deferred until a later user instruction. No A6000 resource queue is launched.

Optional --snapshot-at records actual running group states at a vector boundary.
It reads material/robot states without stepping/resetting the environment.
The renderer tiles isolated worlds for viewing, shows the URDF visual robot and
one mounted knife, adds no crack overlay, and records global world IDs/source
state hashes. Preflight captures are labelled PREFLIGHT, not long-training output.

### Full-learning probe correction and BladeMaster result

The initial BM script collected 24 vector steps despite a 32-step real-critic
window. This was a preflight configuration error, not evidence of a complete
learning pass. It was replaced with 40 real steps; the critic window was not
shortened. The DiSECt probe now derives its collection length from WM/critic/
policy history settings, asserts the intended actor=8 and critic=32, explicitly
samples real critic windows, and verifies finite updates/parameters before and
after replay/checkpoint restore.

BM fresh 256-world probe: 40 steps, 10240 new physical transitions; full
WM(10 optimizer steps), policy(1) and critic(20) updates passed, followed by
checkpoint/replay restore and another round. Effective critic samples were
4096 x 32 = 131072. Driver peak recorded 22624 MiB, including restoration.
This is engineering validation, not a learned cutting result. Root420/skip5,
backend39; external forward batch regressions4 passed. External full2164 suite
was not completed (CPU attempt hits a hardcoded CUDA test); no claim of full
external differentiable-extension regression. See BLADEMASTER_BATCHING_REVIEW.

DiSECt GPU7 probe begins with 3072 worlds (1024/object), explicitly a capacity
candidate. It must retain 3 GiB free after complete learning and restore to be
accepted; otherwise use a smaller candidate. New run is not launched on an
allocation-only result. Task, material discretization and learning batches are
unchanged.

### Resume scope and learning configuration audit

Both backends retain critic window32 and actor imagination8. Policy history10,
34 observations/frame, goal8, action6; WM batch2048 x10, policy batch4096, real
critic4096 x32 effective state targets; storage microbatches512/8192. Finite pose
task gamma1 overrides the source0.99. Optimizer base rates remain2e-4; real critic
uses lambda-return targets (not pure Monte Carlo). No risk penalty. Warmup for
a large count is max(10240,10*worlds); it does not certify model accuracy.

A normal vector-boundary checkpoint restores schedule credit and replay. A hard
wall-clock/interrupt inside a gradient round can save partially executed
optimizer steps; not-yet-executed updates in that vector's queue are not retained.
That interrupted artifact is a resumable training state, NOT exact equivalence
to an uninterrupted run. Prefer periodic completed-vector checkpoints for a
controlled continuation. Physical episodes in progress are closed on replay
restore rather than spliced into the newly reset physical worlds.

Named checkpoints now retain matching <checkpoint-stem>.replay.pt instead of
overwriting the sole replay.pt. last.pt keeps replay.pt for compatibility.
Restore prefers the named pair and accepts a matching legacy replay only when
the named pair is absent; checkpoint identity/iteration checks remain mandatory.
Regression verifies a later last.pt replay does not prevent restoring an older
named checkpoint. Two new pair-retention/legacy tests and the focused
resume/schedule/snapshot suite passed27 tests.

### DiSECt GPU7 full capacity result: 3072 worlds

Passed on the RTX4090: 1024 cucumber +1024 flat-bottom half potato +1024 prism.
Task/groups in the launch JSON are identical to the tested configuration;
only description/status labels differ. No reduced material discretization.

- Initialize and per-material settle:380.23s.
- Actual40 vector steps:122880 new physical transitions, mean23.924s
  (23.876–23.984s), about128.4 transitions/s. No episodes ended in this short
  untrained-policy collection; terminal/reward coverage is from the separate
  physical-reference and regression evidence, not invented from this probe.
- Full WM round:121.87s; actor+real critic:58.13s, including cold compilation.
  These are first-update timings, not steady long-training throughput.
- Selected reset:18.89s; nonselected worlds bitwise unchanged.
- Actor horizon8; critic window32 and131072 effective real targets per round.
- Checkpoint+replay restored and another WM/actor/critic update passed.
  Final optimizer steps:WM20,actor2,critic40; finite metrics and parameters.
- Free GPU memory after first full learning:3.449GiB; after restore/update:
  4.701GiB. Initial full-size settling cache was resident during the first
  learning check. This exceeds the preset3GiB reserve.
- Actual geometry capture succeeded without changing environment state.
- Total probe1558.62s. Full report:
  runs/disect_capacity_20261114/worlds3072/report.json.

3072 is the largest completely tested safe configuration here, not a claim of
an exact hardware maximum. No attempt is made to spend the remaining reset/
training reserve simply to increase the advertised environment count.

### Main-run launch recipe (after combined root/backend regression)

Fresh networks, seed0, no risk penalty. Question: can the full-wrench shared
WM+BPTT policy train on the three-object distribution with a consistent physical
data/update cadence? This is an engineering learning run, not a claimed method
comparison. Report per-object pose completion, failures, speed error, force/
torque utilization, returns and model/critic losses; material separation remains
an independent metric. End at the fixed budget; do not extend or retune silently.

Command uses pose_three_object_wrench6_3072.json, real critic, actor8, BC.25,
500 WM-only rounds; warmup30720 transitions (10 vector steps), one WM round per
640 post-warmup transitions; storage microbatches512/8192; effective batches
unchanged. Budget:1280000 new transitions or24hours, whichever first.
Checkpoints every32 vector steps (~98304 transitions), paired replay retained.
Local JSON and offline W&B logs avoid making training depend on login/network.
Capture actual geometry at step1 and120; validation screenshots are separate.

Combined committed-candidate regression on GPU7: root425 passed/5 skipped
(239.39s), backend39 passed(5.49s). Logs in
runs/disect_capacity_20261114/{root_tests,backend_tests}.log.
The paused 192/3 runs remain archived; no weights or replay are reused.
