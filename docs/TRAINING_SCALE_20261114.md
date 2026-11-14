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
transitions. Integer credit is checkpointed. Resume with replay restores exact
cadence; deliberately empty replay refills first. Changed world count or schedule
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
an unrelated job occupying about 36 GiB; capacity conclusions apply only to
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
VRAM or a maximum hardware capacity. Next:768 full-wrench real-learning probe.
Measurements are in the A6000 frozen checkout's runs/scale_preflight_20261114.
