> Historical record. Use [PLAN](../PLAN.md) and [SHARED_SCENE](../SHARED_SCENE.md) for current configuration; this file does not authorize a launch.

# Framework provenance and training-readiness audit — 2026-11-16

Scope: current shared-scene contract, backend adapters, renderer export, learning
preflight and launch documentation. This is not a claim that every historical
file or numerical physics regime is bug-free. Frozen runs remain unchanged.
Starting main revision: `a327260`; native BladeMaster: `c51ef733364aa9cbbbceb6ba5b422cabf67ddefc`.

## Unjustified rendering addition: removed

Codex added a cylindrical shank and collar on November 15 (`1f0aa56`), later
reused for the shared blade (`ba347e9`). These were invented display geometry,
not measured CAD, URDF parts, material contact bodies, or a user-selected design.
The intent was to bridge the displayed handle and physical blade, but this
misrepresented the installation. Both meshes are now removed without replacing
them or moving the recorded blade. The original handle remains. There is no
measured adapter mesh for the new physical blade; a visible gap is not repaired
by inventing hardware. Historical videos retain their old appearance and must
not be described as re-rendered.

Re-export check uses all 59 recorded frames from
`runs/tool_position_demo_20261116/recording_final`: physical blade vertices,
triangles, material geometry and time are identical; exactly two display meshes
are removed. Rigid-mount residual is 5.674e-9 m. This validates coordinate
consistency, not the real mechanical design.

## What is upstream and what we implemented

| Component | Provenance / local implementation | Limit |
|---|---|---|
| Learning foundation | SuperMimic reproduction; real critic, actor horizon 8 and BC .25 are the discussed working variant | Not an untouched upstream algorithm or an established novel contribution |
| Robot/control | SliceIt-derived UR5e assets, FDCC and action mapping; local MuJoCo Warp integration | Six policy outputs, five active channels with pitch disabled |
| Observation | User-requested six-component wrench, 34 values/frame, ten-frame policy history plus goal | Not the original SliceIt observation unchanged |
| Task | User-selected pose completion + board constraint; local depth/speed costs, timeout and final-position implementation | Pose success does not prove material separation |
| Task numbers | Horizon 180, gamma 1, reward weights 10/20/1/.1 and terminal +/-100; wrench limits 50 N / 5 N m | Engineering settings, not measured Mini45 limits |
| Final position | User chose final-only by default; local implementation extends the former single-axis 40 mm tolerance to EE XY radius | Fixed distance, not adaptive to object size |
| Optional path bounds | User requested an optional process constraint; 25/60 mm and weight20 were implementation choices | Disabled in the default contract; no claimed learned benefit |
| Start/settle | Shared per-object settled-top placement, 5–30 mm gap, 1 s material settling | Proxy for pre-cut localization; no camera/error/calibration implementation |
| Board/workspace | Local finite-blade software interlock and explicit EE workspace | Not a calibrated compliant board; rendered board edges are not physical edges |
| Knife/mount | User requested one knife; local rigid transform and BladeMaster two-slope contact profile implement the DiSECt blade | Mount is derived from canonical robot geometry, not measured real adapter CAD |
| DiSECt physics | FEM/prepared cut springs plus documented contact-coordinate/wrench/stream corrections | Local corrections; not unmodified upstream physics |
| BladeMaster physics | Native MPM/damage reproduction plus robot port, batched worlds, per-world material buffers and shared knife profile | Not an official drop-in robot package; its law/integration differs from DiSECt |
| Material definitions | Shared E, Poisson ratio, density and scale distributions; BladeMaster starts from `paper-banana` for all three geometries | Its nominal yield stress 30000 Pa and cutting energy 700 J/m² remain banana-preset values, not calibrated potato/cucumber/prism fracture properties |
| Friction | Shared board/knife coefficient .2; BladeMaster static/dynamic coefficients both set to .2 | Chosen engineering value; matching coefficients does not match contact laws |
| Render surfaces | Local studio lighting/wood board, FEM scan-appearance mapping, particle isosurface and optional approximate color transfer for BladeMaster | Display choices are not physical geometry measurements or original UV recovery |
| Infrastructure | Local replay/resume, per-world reset, sampling, W&B, geometry recording and studio rendering | Integration work, not claims of upstream scientific method |

User authorization of a feature direction is not evidence that each numeric
choice was personally selected or physically calibrated. The inappropriate
visual connector is specifically a unilateral addition; the requested backend,
observation and task integrations should not be confused with that addition.

Retained alternatives (the official SuperMimic adapter, SAC, WM + MPPI and
risk variants) are separate paths. The MPPI integration combines the external
optimizer with local WM state/cost/termination adapters. These two learner
preflights do not certify those alternatives for the newest shared task.

Exact active values and commands: [SHARED_SCENE](../SHARED_SCENE.md),
[scene_v1.json](../../configs/shared/scene_v1.json). Source revisions:
[SOURCES](../reference/SOURCES.md). Task equations: [TASK_FREE_CUT](../TASK_FREE_CUT.md).

## Corrections in this audit

- Removed the invented renderer connector, retaining recorded geometry and the
  rigid-transform check. No reward, material or physical mounting change.
- SHARED_KNIFE now labels the early maximum-height/5-mm images as historical;
  current reset behavior is per-object settled top + paired sampled clearance.
- Replaced broken commands pointing at moved check configurations with the
  current shared preparation/replay entry point.
- The BladeMaster patch README now labels the original two patches as historical
  and identifies the current native revision; those two patches alone cannot
  reproduce the current backend.

## Training-readiness boundary

The current shared input has three worlds per object (nine total) for functional
checks. It is not the formal long-run resource setting. The historical 3072-world
DiSECt launch uses an older task and physics identity and must not be reused as
though it were this configuration. Old policies are not updated by these fixes.

Current actual-data preflight collects 40 steps (360 transitions) and checks WM,
real-critic 32-step windows, actor horizon8, BC.25, selected resets and checkpoint/
replay/sampler restore followed by another update. Evidence is under
`runs/framework_audit_20261116/{disect_learning,blademaster_learning}`. This tests
execution, not learning quality or long-run capacity. Results are appended below
when complete; no long run is launched by this audit.

Before a new long launch: prepare its explicit current-contract configuration,
measure capacity/throughput with that configuration, select its interaction and
wall-clock budgets, and verify online W&B. GPU9's historical native SIGILL has
not been resolved by GPU7 tests. No requirement to resolve every material-model
research limitation is imposed before an engineering training run; those limits
must be disclosed rather than silently changed to make results look better.

## Completed actual-data learning checks

Both backends passed on GPU7, each with three objects x three worlds, 40 steps
and 360 physical transitions. Each exercised 20 WM optimizer steps, two actor
steps and 40 critic steps across the initial and restored updates. The critic
window remained 32, actor horizon 8, BC .25 and policy observation width 34. Parameters
and metrics were finite; selected resets preserved all other worlds exactly;
episode sampler and replay restoration succeeded.

- DiSECt: 624.57 s including first-update compilation and settling.
- BladeMaster: 323.51 s including compilation and settling.

Neither random-policy capture completed an episode within its 40 steps. These
are execution/restore checks, not success-rate measurements or evidence of
learning quality. They do not certify the old 3072/256-world resource settings.
There was no concurrent GPU job during these two probes. The BladeMaster report
records a dirty tree consisting of this audit's documentation and renderer-only
edits; learner, config and physics code were unchanged throughout both probes.

Regression after the renderer/documentation correction: root **547 passed,
5 skipped** (242.25 s); bundled robot/DiSECt backend **41 passed** (5.66 s).
Root skips belong to the legacy SAC Gym/TensorFlow/explicit bridge paths, not
the two current learner probes. Native BladeMaster source was unchanged;
its prior full regression is recorded at the same revision in SOURCES.
Logs: `runs/framework_audit_20261116/{root_tests,backend_tests}.log`.
Edited Markdown links resolve and `git diff --check` passes.


## Follow-up: overview and controller portability

The follow-up recheck found that the all-world overview still used the old URDF
blade. It is now corrected with per-world physical blade capture and shared
studio handle logic; DiSECt base/world transforms are explicitly recorded.
Both shared-tool renderers use the existing main-run 24x16 cm board exterior;
this is display geometry only. The white Gaussian point-cloud overview path was
replaced with recorded-centre/radius sphere instances using explicit display
colors. No invented adapter or altered physical blade is involved.
Fresh 18-world coordinate checks, eight actual renders, the 59-frame video export
comparison and 549/5-skip + 41 regression results are recorded in RESULTS under
"Controller/scene/recording interface recheck". UR10e portability requirements
and official FDCC/UR-driver references are in RUNNING. The current scene is
verified at this functional scale; no real mounting CAD, material calibration or
new formal-scale long training is claimed.
