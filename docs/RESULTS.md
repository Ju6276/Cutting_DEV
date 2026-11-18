# Results

Dated measurements from 2026-11-16 on (shared scene, pose_depth_v2 runs and pose_depth_v3 work).
Earlier entries: [2026-11-11 … 2026-11-15](archive/RESULTS_20261111_20261115.md) (multi-object
task, free_cut_v1–v4, pose-v1 runs, including the 2026-11-14 evidence index) and
[2026-11-02 … 2026-11-10](archive/RESULTS_20261102_20261110.md) (fixed prism development).
[PLAN](PLAN.md) says what happens next; old next-step prose in entries is not a plan.

## 2026-11-16 — Shared geometric depth reward (pose_depth_v2)

User request: remove dependence on DiSECt spring release in progress reward and
use BladeMaster-style downward depth. Implemented as a new opt-in task for both
backends. Pose +100, failure precedence, the arrival counter, board interlock,
force limits and action interface are unchanged. The speed penalty's progress
factor was changed as well; setting only the positive progress weight would
have left spring-dependent reward in place.

Per-world progress uses maximum depth from the actual reset blade-proxy gap to
the positive board stopping clearance. The new span is stored in replay/state
history, reset per world, and preserved on exact physical-frame restoration.
The shared WM computes this progress from predicted joint state analytically.
It has no learned release-progress output and no spring-progress dynamics input.
Old WM/replay/critic shapes and task metadata are incompatible; no old checkpoint
is silently resumed or its published rewards reinterpreted.

Verification on GPU7: three objects, one world each per backend, 16 reference
steps (48 transitions each), with a masked reset of world 0 after step 8.
This is interface verification, not a complete-episode or trained-policy test.

| Backend | Maximum physical/WM equation reward difference | Wall time |
|---|---:|---:|
| DiSECt | 1.46e-08 | 157.5 s |
| BladeMaster | 1.43e-08 | 45.2 s |

Unselected worlds' frame tensors were exactly unchanged by reset for both.
Progress was monotone within each episode and zero after reset. All measured
wrenches remained finite. Physical and model termination flags agreed on these
short segments. Unit tests additionally cover pose settlement, force/workspace
precedence (retained tests), bounded/re-entry progress, independent separation
labels, equal equations across backend identities, finite BPTT gradients,
a finite WM fitting backward pass, and rejection of old model tensor shapes.

Raw reports: runs/depth_reward_20261116/disect_env.json and blademaster_env.json.
Commands: source runs/sac_env.sh, then use scripts/check_pose_depth.py with
configs/free_cut/depth_three_object_check.json (base .venv) or
configs/blademaster/depth_three_object_check.json (cutting-mppi-20261108 Python,
which includes SciPy). Use PYTHONPATH=.:backends/mujoco.
The first invocations omitted the DiSECt cache environment / used a BM Python
without SciPy; they failed before collecting transitions and remain recorded
separately. Successful checks used the documented environments.

Physics interpretation: eval_cut_springs lowers stiffness only when the spring's
knife contact is nonzero and its extension is positive. Its update is
max(0, ke - incontact * max(length-rest,0) * softness).
There is no unconditional severing when a blade passes a geometric height.
Thus arrival with residual connections is allowed by the implemented mechanism;
this does not identify the cause of any particular residual spring or exclude
contact/geometry bugs. No damage rule or residual spring was changed here.

Limitations: depth reward is motion shaping, including free-space descent; it
does not establish material separation. The 10-reference full-horizon preference
audit was adapted to the new task but not rerun in this bounded check. No new
long training or performance claim. Physical knife geometry/mounting still
differs between existing backends and remains a separate unification task.

Final CPU regression on GPU7: root 426 passed / 22 skipped; backend 27 passed / 12 skipped. GPU execution was checked separately by the two three-object probes above; these counts do not mean every optional GPU test ran.


## 2026-11-16 — Shared physical knife and frozen-policy mechanics

Implementation parent: fe36fe8; new revision disect_knife_rigid_mount_v1,
paired with pose_depth_v2. External forward source parent e2e5221, worktree
BladeMaster-Warp-shared-knife-20261116.

The physical blade has the DiSECt ten-vertex geometry (150.000006 x 59.040001 mm,
3.500000 mm spine thickness, 0.080 mm shoulder). Both backends use a canonical
rigid EE mount. BladeMaster uses two-slope contact/constraint kernels; DiSECt
synchronizes the tool to rigid FK through substep interpolation. Material laws
and backend-specific integration remain separate.

Three-object reference probe, GPU7:
- DiSECt: cucumber/half potato/prism pose success at 41/33/40 steps; maximum
  physical-tool/FK vertex discrepancy 1.02e-8 m; minimum board clearance
  1.004e-6 m. All three still had material connections.
- BladeMaster: pose success at 50/50/50 steps; discrepancy 6.43e-9 m;
  minimum clearance 1.108e-5 m. Each native material graph had two components.
- Sixteen-step reward and partial-reset checks passed on both backends; maximum
  physical/WM reward discrepancy 1.47e-8 (BM), 1.46e-8 (DiSECt).
Reports: runs/tool_analysis_20261116/disect_full_tool_probe.json,
bm_full_physical_tool_probe.json, shared_disect_check_v2.json and
shared_bm_check_final.json (final configuration). The earlier BM FK-only probe is superseded by the
physical-solver-pose probe.

Existing-policy analysis uses the recorded original physics: main 1109 updates
(48 first episodes, all pose timeouts), BM 866 updates (48 first episodes,
all pose completions). Old tools, task rewards and conditions differ.
Checkpoint identities and reconstructed actions match (max 6.68e-6/1.67e-6).
Re-executing the analysis from raw traces produces identical output arrays.

- Sensor force/moment, policy outputs, mapped FDCC gains/increments, feed,
  target errors and board clearance are plotted against time.
- Both checkpoints disable pitch target increments with max_twist=[1,0].
- PCA includes reset, pre-contact-proxy, contact+3/+5 and goal-only controls.
  BM reset features overlap and later separate by object; other motion/history
  factors vary at the same time. A material-identification claim is unresolved.
- Replacing older force history by current force leaves the main mapped gains
  unchanged and changes its x-target increment by 6.45 micrometres/step on
  average; BM changes stiffness and x increments. Main action clipping is
  about 84%, masking some raw-network sensitivity.

Figures, arrays and reproduction check:
runs/tool_analysis_20261116/{main,blademaster}/ and analysis_reproduction.json.
Detailed interpretation: POLICY_MECHANICS_20261116.md. No new policy training.


Final root/backend GPU regression on GPU7: **446 passed, 5 skipped** at root,
**39 passed** in backends/mujoco. The GPU runs include the actual material/tape
and packed-world tests; DeprecationWarnings come from the upstream dflex AST
compiler. Shared-profile external tests use the external repository's pinned
requirements in an isolated dependency directory.


External BladeMaster full regression: **2163 passed, 8 skipped, 18 subtests
passed**, 2124.99 s, using the isolated pinned-requirement environment.
Recorded separation-event audit: the main-policy trajectories contain zero
complete-separation events; BM traces have final component counts but no
per-step separation timestamp. Figures preserve this distinction.


## 2026-11-16 — Shared-knife scene and reward audit

Captured one settled nominal world for cucumber/half potato/prism per backend;
no policy, control steps or optimizer updates. Shared pose_depth_v2 reward,
physical blade and rigid mount; actual start clearance remains unmatched:
DiSECt 5.52/4.47/4.71 mm, BM 17.45/29.45/5.59 mm above highest material sample.
BM currently starts all groups using the maximum object height. The scene
comparison preserves this difference.

Eight studio PNGs (six close views, two wide views) use the existing studio
renderer. Four corresponding camera transforms/scales compare exactly equal.
The robot-base transform is now recorded explicitly for BM: the old export
assumed the base link had visible geometry and raised ValueError otherwise.
The shared optional knife uses the same display connector for both backends.
Report: runs/shared_scene_views_20261116/{scene_geometry_audit,render_audit}.json.
The wood-board rectangle is a display footprint, not the support-plane extent
or robot EE workspace. TASK_FREE_CUT now states the exact speed-penalty formula,
six action meanings and distinction between EE limits and object sliding.

Validation after the capture/export repair: root GPU suite 446 passed / 5 skipped
(239.70 s); backend 39 passed (6.22 s). The six exported scenes and eight
rendered images completed; camera identity and snapshot provenance checks passed.
No physics, reward, action parameters or policy weights were changed in this audit.


## 2026-11-16 — Optional material-motion envelope (shared task)

User selected small free motion, an excess-motion soft cost and a larger failure
boundary. Added opt-in task.object_slip, revision planar_mass_rms_v1, in both
shared_motion_check configurations. Physics, friction, original configs and old
checkpoints are unchanged. No new long training.

Existing first-episode geometry traces (16 cases/object/backend) were summarized
using mass-weighted RMS horizontal particle displacement from episode start.
This measure includes deformation and fragment opening; it avoids cancellation
between oppositely moving halves. Different tools, materials, starts and episode
durations make these old traces descriptive only.

| Backend / object | Min / median / max peak RMS (mm) |
|---|---|
| DiSECt cucumber | 33.51 / 50.32 / 59.87 |
| DiSECt half potato | 29.62 / 44.28 / 76.50 |
| DiSECt prism | 2.21 / 36.02 / 62.94 |
| BladeMaster cucumber | 20.23 / 20.28 / 20.33 |
| BladeMaster half potato | 19.94 / 19.95 / 19.95 |
| BladeMaster prism | 18.00 / 18.09 / 18.21 |

Source: runs/object_slip_20261116/old_policy_motion.json, derived from
runs/frozen_pose_eval_20261115/{disect_video,blademaster_video_ready}.
Development values: 25 mm free, >60 mm failure; capped quadratic excess cost
20/horizon per step. They are not calibrated physical limits or benchmark gains.
This is not a retrospective change of the recorded successes/failures.

Checked actual ground-friction settings: DiSECt contact_mu=0.2 for all three
scenes; BladeMaster paper-banana-based object configs load static=dynamic=0.2,
board_half_extent=(0,0). Equal coefficients do not imply equal contact dynamics.
No friction values were changed.

Implementation checks include partial reference reset, opposing fragment motion,
free-zone and soft-cost gradients, analytical WM/physical task equivalence,
simultaneous failure precedence, checkpoint layout rejection and one finite WM
optimizer update. Policy input dimension is unchanged; WM/critic gain one
supervised displacement scalar and require new task/state metadata.

Three-object physical checks used 16 reference control steps per backend, with
one world's reset forced at step 8. Other worlds' frames remained bitwise equal
through that partial reset. In each transition, the *measured next physical state*
was passed into the analytic WM task calculation. This tests task equations, not
learned dynamics accuracy. Short traces did not reach the displacement failure
boundary; that boundary and its gradients/settlement are covered by constructed
state tests, not a natural large-slip episode in this probe.

- disect: 169.3 s, maximum absolute task-reward difference 1.46e-08; reward/termination and masked reset checks passed.

- blademaster: 44.5 s, maximum absolute task-reward difference 1.43e-08; reward/termination and masked reset checks passed.


Artifacts: runs/object_slip_20261116/{disect_physical,blademaster_physical}.json
and the corresponding logs. No claim that these changes already reduce sliding
in learned policies; no new trained policy was produced.

Regression after the changes: root **463 passed / 5 skipped**, with
CUTTING_BACKEND_ROOT set and the MPPI Python environment; backend **39 passed**.
Logs: runs/object_slip_20261116/{root_tests,backend_tests}.log.
The DiSECt scene-friction read succeeded in the base Python environment; the
first BladeMaster config read lacked SciPy there and was rerun successfully in
the documented MPPI environment. Both physical probes passed.

Reporting correction: describing the old runs as merely two physics backends
was incomplete. Their actual physical blades and mounting/coupling differed,
as did task shaping and condition distributions. Shared-tool revision
ffede36 and scene audit ba347e9 address the new configurations, not the provenance
of old policies. See SHARED_KNIFE for remaining reset and 2 ms coupling differences.

## 2026-11-16 — Shared per-object start cases and optional bounded pitch

User decision: prepare simulation starts like the intended real-robot sequence,
using each placed object's height and variable clearance. No separate fixed
5 mm debug task. Implemented source: configs/shared/scene_v1.json; full
settings, remaining mismatches and commands are in SHARED_SCENE.

Each object has 3 worlds in this resource-bounded probe. The 9 sampled gaps (mm)
are cucumber [24.448, 10.463, 13.140], half potato [19.864, 26.958, 9.601],
prism [17.545, 13.648, 24.904], identical between backends. Actual heights are
measured after settling; identical clearance does not imply identical absolute
knife height or material shape. The camera/local-height estimator is not implemented.

| Measurement | DiSECt | BladeMaster |
|---|---:|---:|
| Physical steps / worlds | 16 / 9 | 16 / 9 |
| Maximum actual gap error | 1.171e-07 m | 1.936e-07 m |
| Maximum physical/WM reward-equation difference | 1.485e-08 | 1.473e-08 |
| Partial reset: non-reset frame fields identical | yes | yes |
| Wall time (includes setup/settling) | 184.8 s | 55.7 s |

Reports: runs/shared_scene_20261116/disect.json and blademaster.json.
Both preserve requested per-world gaps on partial reset. The reward check supplies
actual next-state fields to the WM task equations; it is not a learned-model
prediction accuracy result. Short reference probes do not measure trained success,
active-pitch contact, randomized-material throughput or real-robot performance.

During implementation, the first new-scene export serialized CPU tensors, which
the packed CUDA path could not consume. Export now preserves CUDA tensors and a
loader guard rejects a mismatched device before kernel launch. A partial BM reset
also exposed an empty-group tool-array assignment shape error in the new alignment
helper; it now skips empty masks and indexes the singleton tool dimension explicitly.
The successful reports above were generated after both fixes.

Optional pitch defaults off. Enabled mode has angle/rate clamps and matching
differentiable physical-action/WM mappings; physical tilted-contact validation is
still pending. Shared reference controls now use the same target speed. Nominal
knife friction is explicitly .2 on both newly prepared backends, with old assets
unchanged. At that revision, per-episode BladeMaster material randomization was outstanding; the following entry records its implementation.

Regression on GPU7: root 487 passed / 5 skipped (241.52 s), backend 41 passed
(5.61 s). Root used the MPPI-capable environment and CUTTING_BACKEND_ROOT,
so available GPU tests were enabled. Changed-document relative links and git
diff whitespace checks passed. No long training started.


## 2026-11-16 — Paired per-episode materials in both backends

Both physical backends now share deterministic per-world episode streams:
log-uniform E .5–2, density .8–1.25, cutting-resistance .5–2; uniform target
speed 1–5 cm/s. Geometry groups remain fixed. Cutting resistance scales DiSECt
spring stiffness and BladeMaster cutting energy respectively. Selected worlds
restore raw state, apply their new coefficients and settle for 1 s. BladeMaster
uses persistent per-world GPU coefficient arrays and a timestep selected at
maximum E / minimum density. Saved streams resume from each world's next episode;
physical episodes restart on trainer resume.

GPU7 checks, three objects x three worlds:
- Both 16-step physical/reward/reset checks passed; inactive worlds unchanged.
- Corresponding initial/reset material, speed and gap cases agree to 5.83e-8.
- Physical versus WM task-equation reward differences: DiSECt 1.46e-8,
  BladeMaster 1.52e-8 (using actual next-state inputs).
- BladeMaster collected 40 steps / 360 transitions; real critic window 32,
  actor horizon 8, BC .25, observation 34. WM, policy and critic updates,
  masked reset, replay/checkpoint and sampler restore all passed; optimizer
  steps 20 / 2 / 40. Forty collection steps contained no naturally ended episode;
  explicit partial resets exercised episode resampling.
- Root regression: 496 passed / 5 skipped, 241.40 s. Robot/DiSECt backend:
  41 passed, 12.51 s. Native BladeMaster: 2165 passed / 8 skipped / 18 subtests passed, 2140.56 s; source c51ef73 (SOURCES).

Reports: runs/paired_material_20261116/{disect,blademaster,paired_conditions}.json
and bm_learning/report.json. Final prepared configs are in assets_final;
the obsolete unused fixed reference-speed field was removed after the probes,
with semantic equivalence checked against their inputs. Commands are in
SHARED_SCENE and RUNNING. The learning probe overlapped regression workers;
its wall time is not an exclusive-device capacity result. No long run started.

## 2026-11-16 — Optional final/path robot position constraints and interface cleanup

The user replaced particle-based sliding constraints with two selectable robot
position modes. Current shared default is final only; path plus final is selected
by common.task.tool_motion.enabled. Both require EE robot-base XY distance <40 mm
from the prescribed goal, extending the old 40 mm single-axis test, together with
the existing height/pitch tolerances and >10 qualifying arrival samples. Samples
remain cumulative. Final success is +100 only without a simultaneous failure.

The optional path mode leaves <=25 mm free, applies a quadratic cost up to
20/horizon per step, and treats >60 mm as failure (-100 once). Final-only mode
does not add this running cost or failure. Force, workspace and board rules
remain in both modes. Fragment separation/opening carries no penalty in the
canonical configuration. Distances are fixed engineering values, not calibrated
hardware limits or size-adaptive thresholds. No policy has been retrained.

Implementation: completion_xy_error is derived from joint FK and the known goal,
not independently predicted by WM. Physical reward/termination, imagined reward/
absorption, arrival counter, critic history and logging share the same rule.
Policy dimensions remain 34 per frame / six outputs (pitch disabled). Task and
state metadata change; historical checkpoint/replay is not silently resumed.

Other issues found and repaired:
- Evaluation resizing had left explicit clearance arrays at their original size;
  evaluation seed did not reseed the new episode-condition sampler. Both now
  regenerate from the shared case protocol.
- Evaluation validates physical knife, board geometry, progress semantics and
  contact discretization, and reports optional motion termination causes.
- Scene capture had still forced old nominal materials, fixed 5 mm starts and
  30 mm/s speed. It now requires the shared contract plus matching prepared
  manifest, preserving all sampled conditions.
- Six superseded depth/knife/motion check JSONs moved to
  configs/archive/scene_checks_20261116. Historical trained-task inputs remain.
- Real deployment docs now distinguish policy I/O from timestamping, wrench
  compensation, coordinate transforms, action mapping, FDCC and hardware stopping.
  This change does not implement or validate a real-robot execution runner.

Boundary bug found by the physical-transition audit: requesting exactly 60 mm
can yield 0.06000000000000005 m after robot-frame subtraction. A strict raw
comparison incorrectly failed 6/9 DiSECt and 2/9 BladeMaster cases in the initial
audit. Comparison now allows eight floating-point epsilons at metre scale.
The 60 mm path boundary remains included; 40 mm final boundary remains excluded.
Translated-coordinate tests cover float32 and float64. This numerical tolerance
is not a new task safety margin.

Evidence directory: runs/interface_motion_review_20261116/ (server only).
tool_assets/ contains current prepared inputs. capture/ contains completed
nine-world settled-state geometry from each backend with actual condition logs;
no policy was loaded. Both use the same paired material/speed/gap distributions.
Older disect.json / blademaster.json in this directory tested the now-retired
particle-RMS interface. tool_disect.json / tool_blademaster.json preceded the
roundoff fix; only the *_final.json reports should be used for current boundaries.

The tool audit holds measured physical kinematics fixed and changes the goal XY
offset to test both reward switches. It is not a naturally occurring 60 mm robot
motion, a trained-policy evaluation or a learned-model accuracy measurement.
Unit tests also cover final arrival, failure precedence, gradients, WM fitting/
BPTT and checkpoint restoration under the new state layout.

Validation completed on GPU7:
- Final physical checks: three objects x three worlds/backend, 16 reference
  steps, masked reset preserves all other worlds exactly. Physical-versus-WM
  reward-equation max error 1.369e-8 (DiSECt), 1.545e-8 (BladeMaster).
- For both backends, the counterfactual 60 mm endpoint fails 0/9 worlds and
  61 mm fails 9/9 only in path mode. Final-only adds no distance running cost.
  At 35/50/60 mm, path costs are .00907029/.05668934/.11111111 per step;
  at 61 mm the reward change is -100.11111111 including failure settlement.
- Root with explicit CUTTING_BACKEND_ROOT: 546 passed, 5 skipped in 241.12 s.
  Skips are legacy SAC gym/TF checkpoint/separate bridge prerequisites, not the
  new position-mode or physical adapter tests. The earlier no-backend-env pass
  was 531/20 and is not the GPU validation result.
- Bundled backend: 41 passed in 5.62 s. Native BladeMaster source is unchanged;
  its prior 2165 passed /8 skipped suite was not repeated.
- Both scene captures completed with nine worlds and the sampled shared
  contract. Relative Markdown links resolve and git diff --check is clean.
- Current artifacts: tool_disect_final.json, tool_blademaster_final.json,
  their raw .transition.pt fixtures, root_gpu_tests.log, backend_tests.log,
  capture/{disect,blademaster}/manifest.json in the evidence directory above.

These checks establish the optional rules and interface consistency. They do
not establish reduced drift after learning or successful real deployment.

## 2026-11-16 — Actual motion demonstration of the two position modes

Evidence: runs/tool_position_demo_20261116/recording_final/comparison.json.
One paired-contract cucumber case (seed 0, first world), with its sampled
material, clearance and target speed. A scripted reference controller requests
a 75 mm lateral excursion over 1.5 s followed by return over 1.5 s, alongside
the commanded downward reference. No policy weights, task thresholds, friction
or material properties were adjusted for this demonstration.

Only the final-only task executes physically. Both task rules are evaluated on
each identical measured transition; the path-rule episode absorbs at its first
termination and never accrues later rewards. The video holds its terminal frame.
It does not simulate hardware braking or compare two newly trained policies.

Results:
- Path mode ends at step 25 (1.25 s) for tool_motion_failure.
- Final-only mode returns to the goal region and succeeds at step 58 (2.90 s).
- Maximum actual planar deviation is about 73.65 mm; final deviation about
  11.7 mm. Maximum wrench-limit utilization is about 0.318, so the path-mode
  difference is not caused by force failure.
- Path running position cost before termination is about 0.625; the terminal
  failure charge is 100. Total returns additionally include ordinary progress/
  speed/load/settlement terms; use the JSON for their exact totals.
- The recorder verifies the replayed final-only reward and termination against
  the actual environment on every transition, including success settlement.

Interpretation of the configured weight: at 35/50/60 mm, each 50 ms control
endpoint costs .00907/.05669/.11111. Ten steps at 50 mm cost .5669; the maximum
position running cost over the full 180-step horizon is 20. Thus it is a mild
preference inside the allowed region, with a hard episode boundary outside
60 mm. It does not force all movement to stay inside 25 mm, and the demonstration
cannot establish trained-policy behavior. No reward weights changed.

A real recording bug was discovered by the exporter's rigid-mount guard.
GeometryRecorder read DiSECt body_X_sc after reset alignment; that cache is
refreshed on the next material integration, while alignment already updates the
authoritative material pose. The initial captured blade was displaced about
2.047 mm relative to the following frames. Recording now reads frame.knife_pose.
No material dynamics, control, task or reward code was changed. A regression
test uses an intentionally unavailable/stale link cache and checks the captured
transformed vertices. The final re-recording passes the exporter's all-frame
rigidity check (error below 1e-8 m), using the same physical knife and mount.

Rendering uses the shared studio exporter/renderer with verified cucumber
appearance, one metal physical blade and no blue proxy/crack highlighting.
The exporter now preserves explicit scripted-capture provenance instead of
mislabeling every episode as a frozen learned-policy rollout.

Regression after the recorder correction: root 547 passed /5 skipped (244.24 s),
bundled backend 41 passed (5.85 s). The focused recorder/position tests pass 41/41.
Video: tool_position_modes.mp4, 1800x1110, 10 fps, 6.9 s, 0.5x playback.
Reviewed initial, terminal-boundary and final frames; ffprobe confirms the stream.
Only the requested finished video/previews are copied to the Mac Downloads folder.

The 60 mm path threshold is a radius in robot-base XY around the prescribed
end-effector goal, not a square, object-centre radius or distance from an object's
surface. It constrains one robot reference point, not the whole blade volume.
The studio exporter contains a display-only cylindrical connector/collar between
the retained URDF grip and the physical blade; its shape is schematic, not a
measured installation part and not collision geometry. The comparison overlay
now states this explicitly. No material/tool geometry changed for this video.

Remaining limits: path checks use 50 ms control endpoints, not a continuous
physical stop; post-training effectiveness is untested. The previously recorded
GPU9 SIGILL inside libcuda remains unresolved (PLAN); this recording does not
establish long-runtime reliability on that host.


## 2026-11-16 — Framework provenance audit and latest-task learner preflight

See [FRAMEWORK_AUDIT_20261116](archive/FRAMEWORK_AUDIT_20261116.md) for upstream/local
choices and remaining training-readiness limits. Removed the invented renderer
shank and collar; corrected stale shared-knife/reset and integration-patch docs.
No learner, reward, material parameter or physical mount change in this audit.
Re-exporting 59 saved frames preserves physical blade/material geometry exactly
and removes exactly two display meshes; rigidity residual 5.674e-9 m. Historical
videos have not been overwritten or claimed to show this correction.

Both current shared-contract backends passed real-data learning/restore probes:
three objects x three worlds; 40 collection steps, 360 transitions; real critic 32,
actor 8, BC .25, 34-dimensional observations. Each ran 20 WM, 2 actor and 40 critic
optimizer steps, with finite metrics/parameters. Masked reset left other worlds
exactly unchanged; replay and episode-sampler restoration passed. Total times:
DiSECt 624.57 s, BladeMaster 323.51 s, including first-update compilation.
Neither 40-step random-policy capture completed an episode; these are not
performance/success measurements or full-scale long-run capacity tests.

Evidence: `runs/framework_audit_20261116/` on GPU7. Learner source: a327260;
native BladeMaster: c51ef733364aa9cbbbceb6ba5b422cabf67ddefc. Reports store config
and script hashes, phases, losses and source status. No new long training started.

Regression after the renderer/documentation correction: root **547 passed,
5 skipped** (242.25 s); bundled robot/DiSECt backend **41 passed** (5.66 s).
Root skips belong to the legacy SAC Gym/TensorFlow/explicit bridge paths, not
the two current learner probes. Native BladeMaster source was unchanged;
its prior full regression is recorded at the same revision in SOURCES.
Logs: `runs/framework_audit_20261116/{root_tests,backend_tests}.log`.
Edited Markdown links resolve and `git diff --check` passes.


## 2026-11-16 — Controller/scene/recording interface recheck

Controller source and UR10e-port limits are now explicit in RUNNING. Current
simulation uses the SliceIt/FZI ROS-free FDCC position-controller port (pinned
cambel source f4359b9), not a running unmodified ROS2 controller. UR10e remains
an unimplemented robot-asset/frames/control/data port; no hardware test occurred.

Found and corrected a remaining visualization mismatch: the all-world overview
still drew the obsolete full URDF knife even though studio movies displayed the
physical blade. Snapshots now record every world's physical blade vertices and
triangles. Overview and studio share robot/handle clipping, hide the cable-envelope
box and add no connector. Old snapshots without per-world blade data are rejected.
DiSECt episode files now explicitly save base/world transforms, as BladeMaster
already did. The existing main-run 24x16 cm display board is shared by both new
rendering paths; original studio recording corners are retained. This changes no
physical support boundary (still an infinite plane).

Actual image inspection also found white particle clouds in the Gaussian overview
path despite the requested palette. The overview now displays low-resolution
sphere instances at recorded material-particle centres/radii with explicit display
colors; studio surfaces/scan appearance are unchanged. This is a display correction,
not a material-law change or new learned result.

Fresh scenes: three objects x three worlds per backend. All 18 physical blades
match recorded EE kinematics: maximum set-distance 3.163e-9 m. Requested and actual
per-world starting clearances differ by less than 1.88e-7 m. Task, force limits,
force-observation revision and episode-condition configurations match between the
two prepared backend inputs. These numbers check software coordinates, not real
hardware accuracy or equality of the two material models.

Evidence under `runs/interface_recheck_20261116/`:
- `final_scenes/`: actual settled snapshots and condition manifests;
- `geometry_check.json`: per-object/world physical blade and start-clearance checks;
- `final_renders/`: six studio stills, two nine-world overviews, `visual_review.jpg`
  and `render_audit.json`; shared display dimensions and no fabricated connector;
- `video_export_check.json`: all 59 frames from the prior demo re-export correctly;
  original blade/material/robot arrays unchanged, board corners numerically equal.
- `root_tests.log`: 549 passed / 5 skipped (242.14 s); `backend_tests.log`: 41 passed
  (5.57 s). The skips remain legacy SAC environment checks. Two new tests cover
  distinct physical tool poses, base/world rotation, no state mutation and invalid
  inputs. Additional final renderer checks executed on actual saved geometry.

The whole-potato appearance map was correctly rejected for the half-potato mesh
(topology mismatch); its failed attempt is retained in `renders/`. Final half-potato
and particle-backend surfaces use disclosed approximate color transfer, not recovered
original UV. Overviews use illustrative colors/LOD. These are validation snapshots,
not evidence that a new long training run has started. Native material laws, reward,
policy/WM training code and formal long-training resource settings were unchanged.


## 2026-11-16 — Current shared-scene long-training readiness and live GPU status

Question: is the current unified task ready for a bounded engineering long run,
and are the historical GPU7/GPU9 runs still active? No long run was started.
No learning, reward, material or renderer code was changed by this audit.

Current-code capacity check on GPU7, code `7db74f2`:
- Three objects x 1024 worlds = 3072, serial object-group execution as generated
  by the shared preparation. Same `pose_depth_v2`, final-only position mode,
  disabled pitch, per-episode material/speed sampling and per-world 5–30 mm gap.
- 40 actual vector steps / 122880 physical transitions; mean step 27.472 s.
  Initialization including per-material settling took 434.925 s.
- Real critic window32, actor horizon8, BC .25, observation34. Effective batches
  WM2048 / actor4096, memory chunks512 / critic8192. No shortened critic window.
- WM20, actor2 and critic40 optimizer steps across initial/restored updates;
  metrics and parameters finite. Non-reset worlds unchanged exactly; episode
  sampler and checkpoint/replay restoration passed, followed by another update.
- Free GPU memory: 5.268 GiB after learning, 6.135 GiB after restored learning;
  peak Torch allocation 17420.37 MiB. Selected-reset test took 27.696 s.
- Final status `passed`, 1758.954 s total (29.32 minutes). The random-policy
  40-step capture ended no episodes; this is not a success-rate evaluation.

The first probe had an undersized 1500-second allowance. Once measured timing
showed initialization plus 40 collection steps alone needed about 1534 seconds,
we interrupted it explicitly after 993.654 seconds (KeyboardInterrupt), then
repeated the unchanged probe/config with a 2400-second ceiling. It was not an
OOM or numerical failure. The aborted run is separate evidence, not counted as
passing. No process timer or simulation settings were altered during either run.

Evidence: runs/long_readiness_20261116/{contract.json,prepared/,
disect3072/report.json,disect3072_complete/report.json,capacity_complete.log,
status.json,proposed_launch.json}. The launch draft is not executed: fresh
networks, 24h or 1280000 new transitions (first bound), 500 WM-only rounds,
30720-transition warmup, 640 transitions/update, online W&B and paired replay
checkpoints every32 vector steps. The current main configuration is eligible for
bounded engineering training; passing this check does not guarantee convergence.

Live check at 18:48:35 CET: GPU7 RTX4090 40 MiB, GPU9 RTX4090 39 MiB,
A6000 GPU0 43 MiB and GPU1 17 MiB; all at 0% utilization and no compute processes.
Availability is a snapshot, not a reservation. A6000's development checkout is
still 2b73009 and needs an isolated current-code/assets deployment before use.
GPU9 driver remains595.80; its historical SIGILL cause remains unresolved.

Historical job status reverified from local logs and cloud API:
- GPU7: budget_complete, 24h, 1062912 transitions, policy1109. Saved frozen
  evaluation remains 0/48 prescribed-pose successes, all timeouts, return−10.97.
- GPU9: native SIGILL(-4), last logged policy998 / 968960 transitions (~19.74h).
  Saved policy866 evaluation remains48/48 nominal-replica pose successes,
  return106.94. This is not an evaluation of the unsaved policy998.
- Both old W&B runs are remotely readable and show finished. GPU9's cloud badge
  does not establish normal exit or completed budget. These two old evaluations
  differ in material distribution, tool/task settings and are not a matched
  algorithm or backend comparison. Pose success is not a separation guarantee.

BladeMaster's current nine-world learner/reset/restore check remains valid, but
this audit did not certify its new randomized/shared-tool configuration at large
scale on A6000/GPU9. Prefer a current isolated A6000 deployment for that next
capacity check rather than treating the old GPU9 launch as already repaired.
Regression remains the unchanged-code result at7db74f2: root549 passed/5 skipped,
bundled backend41 passed. Only docs changed after this probe; no redundant suite
rerun. Updated PLAN and server-local MACHINES distinguish completed old jobs
from the current readiness check.


## 2026-11-16 — Long-run entry audit: pending updates and shutdown status

Follow-up to the current-scene capacity check, at parent a102afe. Scope: active
`train_gpu.py` launch/config path, physical-task/WM regression, checkpoint/replay
resume and shutdown status. No new long training or task/network/reward change.

Two confirmed control-flow defects were corrected:

1. A vector step can earn multiple learning rounds, but the checkpoint saved
   only collection credits, not the remaining unstarted rounds. For example,
   3072 transitions with a640-transition cadence earn four rounds plus residual
   credit; a milestone after the first round lost the other three on resume.
   `UpdateQueue` now checkpoints that tail and its LR clocks; the active GPU
   runner drains it before collecting further physical data. Completed-round
   milestone callbacks run after acknowledgement. Budget checks before beginning
   a round preserve all pending work if no round can start.
2. Environment shutdown previously happened in the outer guard after tracker
   finalization. A close error could therefore accompany a completed progress
   record/cloud finish. Shutdown is now attempted before tracker finalization;
   errors enter failed status, and the setup guard prevents a duplicate attempt.
   This addresses Python cleanup reporting, not GPU9's native SIGILL.

Limits are explicit: an interrupted round may already contain optimizer steps.
Those saved parameters/optimizer states are retained; its unfinished remainder
is recorded and the whole round is not silently replayed. Only unstarted rounds
are recovered. Old checkpoints without the queue cannot reconstruct omitted
work and warn; intentional empty-replay resume records discarded queued work
and follows the existing refill warmup. Physical episodes still restart at
resume. This is not a claim of exact arbitrary-mid-minibatch continuation.

Verification:
- Targeted tests:35 passed, including actual runner control flow with CPU doubles
  for physics/learners. A saved WM milestone with two pending rounds resumes
  both before any new physical step; its completed update counts match the
  uninterrupted runner. A forced close failure reaches progress and tracker.
- Full root regression, including GPU asset paths and optional MPPI environment:
  **562 passed,5 skipped**,242.10 s. The five skips are legacy SAC dependency /
  explicit bridge/checkpoint opt-ins, not current-task failures.
- Bundled robot/DiSECt backend: **41 passed**,5.57 s. Native BladeMaster source
  was unchanged; this is not a rerun of its standalone native suite.
- The first full-root invocation stopped during collection: the MPPI Python did
  not have the bundled backend on its import path. Root conftest now supplies
  that path independently of test collection order; RUNNING shows the explicit
  PYTHONPATH too. Isolated pitch-test collection with PYTHONPATH=. then collected
  all9 cases. The initial collection error is preserved, not counted as passing.

Evidence: runs/runner_audit_20261116/{root_tests.log,
root_tests_with_backend_path.log,backend_tests.log,isolated_collection.log}.
Tests added in test_update_queue.py and test_gpu_runner_resume.py cover queued
round recovery, partial-round disclosure, budget-before-start, empty replay,
legacy/invalid queue records and actual runner lifecycle. No physical simulation
or learner equation changed, so the preceding3072-world capacity result remains
separate evidence; it was not silently rerun or relabeled as this test.

No other new blocking issue was found on the reviewed main launch path. GPU9's
native fault and the pending large-scale current-config BladeMaster deployment
remain open; neither is declared fixed by these changes. Main fresh-run proposal
and online-W&B requirement remain unchanged.


## 2026-11-16 — Shared-task deployment and offline wrench panels

User authorization: inspect docs/visualization, synchronize code, then launch
GPU7 main and A6000 BladeMaster training after checks. The old no-launch plan was
archived; PLAN and TRAINING_SHARED_20261116 now separate current task/budgets
from historical GPU7/GPU9 runs. DATA's stale0.75s current-settling and old native
source entries were corrected. SAC remains the historical single-world bridge;
current shared multi-object SAC and MPPI physical evaluation are not completed.
MPPI package/import test success is not a new-task rollout result.

The two isolated training deployments are at5ec9f85, with native BladeMaster
c51ef73. Robot MJB, source task/reset, FDCC parameters and all three intact assets
match SHA256 on both hosts. Task/reward, six-axis observation, force limits and
episode distributions match; first170 worlds per object have identical sampled
clearances. Different vector counts/hardware and native fracture laws remain
explicit, so this is not a claim of identical execution or matched compute.
Evidence:runs/shared_long_20261116/deployment_audit.json on GPU7 development.

A stale video-caption sentence still described the removed display rod. It was
removed without changing geometry. The new compose_wrench_video helper appends
two unit-separated sensor panels below an existing shared-studio movie. It checks
case/episode identity, observation continuity, reset counters, timing and encoded
frame counts. Signals are signed filtered sensor-frame values reconstructed
from full-wrench normalized observations and recorded per-axis limits.
They are20Hz endpoints, not500Hz peaks or material resultants. Source video is
preserved in the top panel, with ordinary H264 reencoding differences.

Validation used historical GPU7/GPU9 cucumber movies only, not new training
results:100/27 encoded frames preserved; upper-panel first-frame pixel MAE
1.197/1.225 on a0–255 scale. Time/unit/invalid-trace tests pass. The initial
composition invocation used the lean training Python and failed because Pillow
was absent; rerunning in the existing optional MPPI/render Python passed for
both recordings. No packages were added to the live training environment.
Panel source/data/provenance:runs/wrench_panel_validation_20261116.
The historical full movies are retained as validation artifacts, not presented
as current shared-knife training videos.

Full root regression after the visualization changes:569 passed,5 skipped,
294.78s; bundled backend41 passed,6.23s. Tests ran on GPU7 while the already
validated frozen-code training collected/fitted data; those overlapping wall
times are not exclusive-device throughput measurements. Only offline rendering
code and docs changed after the frozen launch. Native solver/learner code did
not change. Current launch state/URLs and A6000 capacity result are maintained in
TRAINING_SHARED_20261116 and the server runtime records.


## 2026-11-16 — Shared long runs launched and cloud verified

A6000 GPU1's510-world full capacity probe passed in2044.1s. It collected
20400 physical transitions, used critic32/actor8 and effective batches2048/4096,
then exercised WM/policy/critic updates, masked reset, physical snapshot and
replay restoration.20 WM,2 actor and40 critic optimizer steps were executed
including restored updates. All results finite; untouched worlds exact;
snapshot state unchanged; sampler restored. Memory headroom7.912GiB after
learning and7.904GiB after restore. This40-step probe ended no episodes and
does not itself validate physical task success.

The guarded launcher then started the510-world BladeMaster long run on A6000
GPU1. GPU7's3072-world DiSECt-backed run is also active. Both use clean5ec9f85,
same algorithm/shared-task distribution, online W&B and independently bounded
24h/1280000-transition budgets. W&B API verification at20:12 CET returned
running state and nonempty physical-transition history for scr1v5np (GPU7)
and pvord4sx (A6000). Exact URLs/PIDs/paths are in TRAINING_SHARED_20261116.
A6000's first510 transitions were recorded after about563s; GPU7 was past80k
transitions in WM pretraining. Neither policy had updated at this launch check.
No training-performance conclusion is made.

GPU7's actual3072-world first-step snapshot was rendered using the common
physical geometry entry point; grid offsets are display only. Correct VTK and
ROS asset-root invocation is now documented. Final documentation audit corrected
historical/current status labels and ten relative links in the moved plan archive;
all scanned local Markdown targets now exist. No active training code was edited.

## 2026-11-16 — Current-task SAC and two MPC adapters

See [CURRENT_BASELINES](methods/CURRENT_BASELINES.md) for the SAC source search, explicit
SB3 fallback, commands and limitations. GPU9 completed identical three-object
first-episode cases: smoke-trained SAC1/3 pose success, frozen early WM+MPPI3/3,
nonlearned FDCC+MPPI2/3; all other outcomes were timeout, and all9 episodes
retained material connections. These are interface checks, not a ranking of
algorithms or proof of material separation. Root outputs remain on GPU9 at
`Cutting_DEV_baselines_20261116/runs/baseline_validation_20261116/`.

SAC resumed264total physical transitions/240updates including its restored
replay. Terminal observations precede masked reset, timeout does not bootstrap,
and update credits count transitions instead of vector steps. Periodic saves
were moved out of SB3's pre-insertion callback to an update boundary. The new
8D-goal MPPI test found and fixed a pose/goal slicing error in optional terminal
critic evaluation; old7D goals retain behavior. Running long jobs were untouched.

A wall-budget regression also fixed the new SAC runner's stop boundary: changing
SB3's schedule horizon alone did not end its local loop. The corrected check
runs before the next rollout; physical expired-budget restore retained the
exact240transitions/216updates/replay240 without a dropped transition.

Final verification: root577passed/3optional original-TF checks skipped;
bundled backend41passed on GPU9. New SAC and planner tests ran, not skipped.

Checkpoint payload hashes now reject mixed model/replay generations after an
interrupted save; temporary payload writes precede the final metadata update.
The mixed-generation rejection test is included. This does not promise rollback.


## 2026-11-16 — GPU9 current-task SAC capacity and launch
The3072-world SAC capacity/update preflight completed3072 physical transitions,
3072 optimizer updates and checkpoint/replay saving, with finite parameters and
normal cleanup. Total475.37s including439.94s initialization; this is not a
measurement of complete-episode or sustained training throughput.
The authorized fresh SAC run then started from5be7d90 on GPU9, same current
three-object distribution and1,280,000-transition/24h cap as documented in
[TRAINING_SAC_20261116](archive/TRAINING_SAC_20261116.md). Online W&B i00yr0fg authenticated
and its first3072-transition history row was independently read through the API.
Local6144 transitions and zero updates are expected during30720 warmup.
GPU7/A6000 running checkouts were not changed. No new SAC success-rate claim.

## 2026-11-17 — Prescribed-cut depth perception prototype

User requested route 1 first, retaining the current prescribed cut. Added a
calibrated depth-to-start/goal module and shared-scene camera capture. Camera:
front above board, offset [0, 0.32, 0.55] m in base coordinates, looking down
approximately 30 degrees from vertical; 640x480, focal length700 px.

Recorded settled cucumber / half-potato / prism surfaces gave visible-top errors
approximately0.34 /0.14 /0.03 mm. These are ideal-depth, single-frame measurements,
not real-camera accuracy. Known robot projections mask self-occlusion; no food
mesh or material labels enter the estimator. All three produced an 8D base-frame
goal compatible with frozen checkpoint128 and its actual mounted blade.

No policy inference or physical cutting episode was executed from these plans.
No RL retraining, changed reward, new cut plane or modified running job.
The end-to-end physical handoff remains unverified. See
[VISION_ROUTE1](methods/VISION_ROUTE1.md) for inputs, commands and limitations.
Evidence is on GPU7 under runs/vision_route1_20261117; no Mac project copy.
GPU9 regression: root589 passed/3 skipped; bundled backend41 passed.

## 2026-11-17 — A6000 throughput and WM-only schedule check

Read-only inspection of the unchanged shared-run checkout (5ec9f85), GPU1,
W&B pvord4sx. At iteration333:169,830 physical transitions,217 completed WM
rounds,0 policy updates. This agrees exactly with
floor((169830-30720)/640)=217. A WM round contains10 optimizer minibatches.
The launch requests500 WM-only rounds, not500 optimizer steps.

Over the most recent30 vector steps:50.44 s/step, comprising28.67 s physics,
10.93 s reset/settling and10.83 s WM fitting. Cumulative physics and reset
accounted for approximately84% of elapsed time. The512-sized network microbatch
is unrelated to the510 physical worlds. Material randomization resets selected
worlds, advances1 s of settling in500 exchanges, and reads each exchange back to
CPU for contact/finite checks; this is measurable overhead and a possible
optimization target, not evidence of skipped learning or deadlock.

At this rate joint learning starts around vector iteration688 (roughly5 hours
after the inspection), subject to subsequent throughput. The24h wall cap remains;
1.28M transitions are an alternative stop condition, not a promise within24h.
No schedule, physical timestep, reset conditions or running job was changed.


## 2026-11-17 — Depth-selected plane and frozen physical feedback trial

The user clarified that perception must choose the cut plane, rather than check
a supplied plane. Added visible-major-axis midpoint selection with no preset XY
input. Translated and rotated depth fixtures verify that selection moves with
the cloud; near-round axes reject. Policy goal quaternion sign is anchored to
the source policy representation, not merely checked as an equivalent rotation.

Seven independently scanned, textured OmniObject3D assets were rendered at native
metric scale: cucumber001/002/003, potato001/002/003 and apple002. Three cucumbers
and two elongated potato views passed; potato003 and apple002 rejected as
ambiguous. These are static ideal-depth geometry examples, not seven physical
cutting tests or calibrated material models. GPU7 evidence:
runs/vision_variants_20261117 (including per-object visual plans and gallery).

A single native BladeMaster cucumber trial used frozen main-run checkpoint128,
policy66. The A6000 native actor had zero updates, so this is explicitly a
cross-backend inference demonstration. The selected plane shifted the EE goal
by X -0.355 mm / Y +8.986 mm relative to the old preset, with yaw about0.46 deg.
Only calibrated depth selected its location. Pre-cut IK/reset changed the robot
pose while verifying unchanged material positions; no physical approach motion
was claimed. Thereafter actual force/motion feedback drove180 policy steps.

Final trial06: timeout, return -6.754, no force-limit or workspace failure;
peak material force2.720 N, peak limit utilization0.0850, depth progress0.95869.
Blade-board gap ended at2.476 mm, EE target height error2.491 mm, XY error14.103 mm.
Material component diagnostic remained1. The chain executes, but this policy did
not complete the selected task. This is neither native-policy performance nor
proof of arbitrary cut generalization. No counterfactual separates visual goal
change from cross-backend transfer; no policy/reward/material parameter was tuned
to make the trial succeed.

GPU9 evidence: runs/vision_closedloop_20261117/trial_06 contains all raw transitions,
per-step geometry, visual plan, conditions, source/checkpoint provenance and
handoff_audit.json. Shared studio rendering uses the actual recorded blade and
original handle; no adapter rod or blue blade is added. Camera display coloring
was corrected on the source polydata before its normal filter; depth and plan
were independently verified unchanged (max depth difference0).

Earlier attempts are retained rather than counted as episodes: trial01 startup
exit139, cause unconfirmed; trial02 overly strict raw quaternion IK comparison;
trial03/04 renderer dependency / snapshot identity failures. Trial05 ran but used
the opposite raw goal quaternion sign from the source policy, so its timeout is
an interface diagnostic. Trial06 corrected the input convention; it still timed
out. Long training checkouts and algorithms were not changed. All evaluation and
regression time shared GPU9 with SAC; no exclusive throughput claim.

Final regression on GPU9: root596 passed/3 skipped; backend41 passed.
19 visual geometry/interface tests are included.


## 2026-11-17 — Documentation consolidation and current cut-plane audit

This change touches documentation and Git organization only. No Python source,
task/config JSON, asset, model weight or running job is changed. README/PLAN/
PAPER now separate current implementation, method2 fixed-heading center-cut
pilot, and historical evidence. METHOD explicitly labels its old24D/7D source
mapping as historical. Three dated review/launch files move to archive with
links rebased. DATA records retired branches and preserved heads.

The GPU7 active prepared DiSECt assets reconstruct interfaces at material x=0.
Fractions along their initial x bounds: cucumber49.8683%, half potato49.7168%,
prism50.0000%. SHARED_SCENE records file hashes, frames and midpoint offsets.
These are approximately centered geometric cuts, not equal-volume claims.

Method2 is scoped, not trained. Visual physical execution currently supports
only the frozen WM+BPTT policy; SAC, WM+MPPI and analytic MPC visual adapters
remain to be implemented.

Validation on GPU7 RTX4090 with the prepared environment:
root588 passed/6 skipped (298.70 s), bundled backend41 passed (6.70 s);
relative Markdown links and git diff whitespace checks pass.
Root used the optional MPPI Python; backend used .venv. Existing deprecation
warnings remain. The first invocation omitted runs/sac_env.sh and failed on
TORCH_EXTENSIONS_DIR; RUNNING now includes that prerequisite.
These test counts describe this environment, not certification of untested
hardware deployment or all historical physics paths.


## 2026-11-17 — Fixed-heading visual learning and two physical handoffs

Authorized scope: train a midpoint predictor on three existing geometries with
fixed headings and small XY translations, then execute with the frozen policy.
This did not retrain the WM, actor or critic, or alter either active long run.
Source data: recorded settled DiSECt studio surfaces of cucumber003, half potato,
prism. Static rendering translations are not900 independently simulated settles.

900 cases (300 per shape), split600/150/150 by unique placement ID, +-15mm XY.
Ideal640x480 depth, fixed calibrated synthetic camera; no sensor noise or new
shapes. Visible cloud centroid normalization excludes absolute XY and full mesh
center from input. Offline labels are full surface-bounds midpoints along +base Y.
Checkpoint selection used validation only; test read once. CPU120 epochs took
42.58s; best epoch117, validation MAE0.2199mm. Online W&B abn7qktr is finished;
remote history verifies epoch120. Dataset SHA256:
8dac0c38e933dc29a4ef8e7575045c1ef63872d7e05d54fadc48c972bfb62da7.

| Held-out150 placement cases | Midpoint MAE (mm) | P95 (mm) |
|---|---:|---:|
| Learned local-cloud residual |0.2099|0.5653|
| Fixed-heading visible2-98% extent midpoint |5.4995|9.1961|
| Constant world coordinate fitted on training labels |7.6851|15.2314|
| Visible point-cloud centroid |15.1286|22.3001|

The network can exploit the known shapes/view and occlusion bias. This is not
new-object generalization, real camera accuracy or evidence of novel geometry
learning. The geometric fixed-heading variant is compared here; original route1
PCA-heading selection remains available separately.

Physical evaluation on GPU9 used the same explicitly cross-backend transferred
policy66 (source checkpoint128, SHA256
3bf6d7017bcfe31b5e38da68d61372156c2c85a77e8328b84989e126d138a8fd).
One native-BladeMaster nominal cucumber episode per selector,3cm/s,20mm
clearance,180 steps. Policy identity and recorded material/command conditions
match. Native numerical initialization is not claimed bit-identical.
One-shot depth generates the goal; motion and wrench provide execution feedback.
Start uses simulation IK/reset, not a validated robot approach trajectory.

| Quantity | Geometric fixed heading | Learned fixed heading |
|---|---:|---:|
| Absolute midpoint error against initial full rendered surface (mm) |8.8515|0.6880|
| Final target-height error (mm) |2.4964|2.8897|
| Final blade-board gap (mm) |2.4798|2.7957|
| Height-qualified samples (<2mm) |0|0|
| Peak material-force norm (N) |2.7469|8.6046|
| Maximum six-axis limit utilization |0.0854|0.2262|
| Return |−6.7568|−6.9893|
| Ending |timeout180|timeout180|
| Connected material components |1|1|

No force/workspace failure; neither pose completion nor complete material
separation occurred. Learned localization improved in this single physical view,
but control did not complete. The direct missing completion condition is target
height; this alone does not identify the learning/transfer failure's cause.
Do not turn these two cases into success-rate or force-superiority claims.

Evidence: GPU7 runs/visual_center_20261117/{dataset,learning};
GPU9 runs/visual_center_closedloop_20261117/{fixed-midpoint,learned-midpoint},
comparison.json and queue.json. Both complete transitions and geometry recorded.
Same shared studio renderer, single physical blade, original gripper, no invented
connector or blue overlay. Visual results are not physical-separation labels.

Validation on GPU7: targeted visual tests26 passed; root595 passed/6 skipped,
backend41 passed. Root skips cover unavailable SB3/gym and explicit historical
TensorFlow/physical bridge cases; no newly added visual test is skipped.
Data IDs, placement splits, relative/base label consistency and checksum verified.
New scripts compile; complete root run used the prepared CUDA/MPPI environment.

## 2026-11-17 — Official DP3 point encoder vs local midpoint network

User-authorized replacement trial; upstream MIT PointNetEncoderXYZ pinned at
47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97, class AST verified identical. Only its
colored diagnostic printing is suppressed. Official XYZ/64D/LayerNorm settings;
our bounded midpoint head, not the DP3 diffusion policy. All weights trained
from scratch. Original input normalization/target/goal conversion unchanged.

Paired600/150/150 data,120epochs,Adam.001,batch32. Independent shuffle generator
ensures the same data order across architectures for each seed. Local27073
parameters; DP363297. Same epochs do not imply equal capacity or compute.
Validation selects each checkpoint; test was already seen in the previous
pilot, so this is a reused development benchmark, not a new final evaluation.

| Seed | Local MAE mm | DP3 MAE mm |
|---|---:|---:|
| 20261117 | 0.2067 | 0.1519 |
| 20261118 | 0.1945 | 0.1361 |
| 20261119 | 0.2027 | 0.1230 |

Mean MAE0.2013 ->0.1370mm (about32% lower); pooled95th percentile0.5014 ->
0.3533mm; maximum0.9083 ->0.4869mm. Each paired seed favors DP3 on MAE.
Case bootstrap of the three-fit mean difference DP3-local: -0.0643mm,
95%[-0.0872,-0.0433]. This conditions on these fits, does not include a
population of training seeds, unseen objects or actual camera errors.
Use DP3 as the new visual encoder; retain local_v1 for historical checkpoints
and comparison. No test-driven tuning or threshold changes were performed.

Frozen physical handoff used preselected seed20261117 (not lowest test error),
same policy66 SHA3bf6d7017bcfe31b5e38da68d61372156c2c85a77e8328b84989e126d138a8fd,
nominal native-BladeMaster cucumber and explicit cross-backend inference.
DP3 plane error -0.8453mm (previous local -0.6880mm),180-step timeout,
return -7.0070, peak material force10.8438N, peak limit ratio0.2513, one material
component. Final EE height error2.9135mm exceeds2mm tolerance; zero
height-qualified samples. No force/workspace failure. One execution per version
cannot establish a force difference; smaller dataset error did not improve
task completion, and did not improve this particular physical localization.
This uses old policy66 deliberately; it is unrelated to the newer policy527
video below. No retraining or edits to either running controller.

Code validation: root598 passed/6 skipped, backend41 passed; center10 passed.
Optional SAC/Gym/historical TF skips remain. GPU7 training/evaluation/tests
overlapped: timing is not exclusive-machine throughput.

## 2026-11-17 — Requested GPU7 policy527 studio videos

Fixed checkpoint224 (policy527,WM1027), SHA
d542414c9f783ed9055b08bfd391f84ca1e1b9ada70af8ebf704fa861a7a4549.
One first episode per shape, seed20261117, generated from the actual training
conditions, unchanged task/reward, deterministic frozen policy, no updates.
All three reached pose-task success in16 steps, none hit force/workspace limits.
Material complete separation was false for all three.
Added 2026-11-17: the same summary records mean speed error 0.073 m/s (targets
0.031–0.046 m/s) and mean final XY error 22.7 mm; see the shared-run success
audit entry below. Pose success here is not evidence of the intended cut.

| Object | Return | Peak material force N | Peak limit ratio |
|---|---:|---:|---:|
| Cucumber |98.34|7.90|0.0958|
| Half potato |99.96|15.85|0.5389|
| Prism |97.63|17.99|0.3936|

These three selected-by-seed cases are illustrations, not the last100 training
episodes or a reliable success-rate estimate. Initial attempted evaluation
combined the newer evaluator with frozen source lacking shared_baselines and
failed before scoring; failed output retained. Successful rerun used the
current validated interface, without changing the active training checkout.

Shared studio renderer, verified cucumber scan map, potato appearance projection,
same existing cameras/palette, physical blade/original handle only, no blue proxy
or invented connector. Full clips0.5x,26frames10fps, plus synchronized signed
sensor force/torque panels (20Hz filtered endpoints, not500Hz limit peaks).
FFmpeg full decode and image inspection passed; downloaded MP4 SHA256 matches
server. Local destination: Downloads/cutting_policy527_20261117.

DP3 comparison W&B: all six remote runs verified finished at120epochs:
- [0cboc47l](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/Cutting_DEV/runs/0cboc47l)
- [usx01450](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/Cutting_DEV/runs/usx01450)
- [bn7bjlv8](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/Cutting_DEV/runs/bn7bjlv8)
- [57gbfk7p](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/Cutting_DEV/runs/57gbfk7p)
- [8duewbr9](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/Cutting_DEV/runs/8duewbr9)
- [cj4mrcsu](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/Cutting_DEV/runs/cj4mrcsu)

## 2026-11-17 — DP3 goals to RL, paired WM+MPPI and nonlearned MPC

User requested videos for all three. Kept native-BladeMaster nominal cucumber,
force limits/task/reward/horizon and DP3 seed20261117 weights unchanged.
The RL episode is the existing old policy66 execution (not policy527).
WM checkpoint128 contains exactly its actor tensors; WM566, source5ec9f85.
Both learned controllers are explicit DiSECt-to-BladeMaster inference transfers.
The nonlearned FDCC approximation holds measured wrench over its horizon;
it has no learned model, critic, material parameters or fracture predictor.
Both planners use pinned pytorch_mppi, horizon8,samples512,one refinement,
gamma1,no terminal critic; analytic5substeps. No retuning after results.

The three condition dictionaries and DP3 model hashes match. Each selector
observes its separately settled initial scene; the largest target spread is
0.02435mm in Y (X0.00115mm,Z0.000009mm). Therefore these are matched nominal
cases, not bit-identical physical resets or a common cached target.

| Controller | Outcome | Steps | Return | Peak material N | Max force-limit ratio | Material components |
|---|---|---:|---:|---:|---:|---:|
| DP3 + old RL66 | timeout |180|-7.007|10.844|0.2513|1|
| DP3 + paired WM566 + MPPI | timeout |180|-6.329|10.622|0.2543|1|
| DP3 + nonlearned FDCC MPC | pose success |105|101.452|10.808|0.2749|2|

No force/workspace failure. Final target-height errors2.913/1.771/1.527mm.
WM+MPPI first entered the2mm height band atstep176: only5qualifying samples
before timeout, insufficient for the task's >10 arrival counter. Analytic MPC
entered at95 and succeeded at105 (11samples). Do not infer that extending
the horizon necessarily fixes WM, or that analytic control is generally better.
Two material components can include fragments; no new intended-slice integrity
claim. One trial per method, reused RL run, no success-rate significance.

Planning time WM mean0.4691s,p950.6281s; analytic mean0.4597s,p950.5052s.
These exceed50ms control period and were measured alongside CPU rendering.
They are offline receding-horizon demonstrations, not real-time execution.

Execution found and fixed target_speed shape N vs N-by1 before any scored
planner step. First failed runs preserved as *.failed_speed_shape. A separate
render-watcher accidentally pre-created the analytic output directory; that
attempt did not start physics and is preserved separately. Corrected watcher
creates its staging directory only after evaluation. Restart changed no task
or planner settings. WM and analytic completed with frozen-weight checks.

Verification: root605passed/6skipped, backend41passed; GPU9 optional planner/SAC
plus transfer-contract tests14passed; visual/transfer targeted36passed.
Active long training source/checkpoints were unchanged.

Recording-label correction after execution: planner snapshot policy_updates66
was inherited from the reference checkpoint, not a trained MPC actor.
New captures put null there and record reference_policy_updates66 separately.
The scored trajectories remain unchanged; video titles identify actual methods.

Approved shared studio render uses recorded blade/original handle, scan-based
color transfer, no blue proxy/cut-face highlighting/fabricated connector.
Delivery movies start with the actual pre-cut camera image and DP3 selection,
then the full1x episode plus synchronized20Hz filtered force/torque endpoint
charts. Those charts are not the500Hz peak-limit measurement. Images were
inspected for all three controllers. Delivery filenames are dp3_rl.mp4,
dp3_wm_mppi.mp4 and dp3_analytic_mpc.mp4 under the DATA paths.

## 2026-11-17 — Current-entry cleanup and evaluation persistence repair

The repository audit found stale current-facing instructions, not a new
training-method change:
- PLAN still said A6000 had no trained actor and repeated completed visual
  budgets as future work. It now points to dated evidence and current actions.
- RUNNING still described the old blue contact-blade overlay, an obsolete
  development branch and old pose-v1 evaluation configs. The current path is
  frozen pose evaluation -> shared physical-scene export -> studio rendering
  -> optional synchronized sensor-wrench panels.
- The script index listed TensorFlow single-world SAC as the active unified
  learner and contradicted the now-implemented visual MPC dispatcher.
  It now identifies train_sac_multi.py and all three visual controller choices.
- Every configs/free_cut/pose_*.json still declares free_cut_pose_v1; their
  README incorrectly called them current. They are retained as historical
  fixtures. Current pose_depth_v2 inputs are generated from the shared contract.
- DP3's existing MIT license and pinned provenance are now indexed in NOTICE.

A real lifecycle defect was fixed in evaluate_visual_cut.py: planner/trajectory/
summary persistence errors could skip env.close(). Cleanup now always runs,
and failed persistence updates progress when writing remains possible.
The pose evaluator already closed unconditionally; it now also marks result
persistence failures instead of leaving completed status behind.
Fault-injection tests cover all three save stages, successful saving and an
unwritable error log. These do not establish new physical cutting performance.

No task/reward threshold, model architecture, physical scene, active training
checkout, replay or weight was changed. Historical code imported by shared
exporters and tests remains, explicitly labeled; it is not deleted merely
because it is old. Current-facing local links and first-party Python syntax
were checked. The previous long PLAN and script instructions remain at
Git revision e7b9e69 rather than another duplicated archive file.

Runtime snapshot during this audit (2026-11-17, around15:00 CET): GPU7
policy784,854016 transitions; A6000 policy113,423300 transitions; both joint.
GPU9 SAC progress records completed at1281024 transitions / transition_budget,
23956 seconds. These are status observations, not frozen evaluation scores.
Validation: initial GPU7 suite609 passed /6 skipped /1 failed; the failing
parallel-group initialization raised CUDA OOM while the long trainer occupied
about22GiB. An isolated retry reproduced the resource failure. Tests were not
relaxed and the trainer was not stopped. On idle GPU9 (RTX4090), the same
changed code passed the complete root suite619 passed /3 skipped in155.5s,
including the parallel reset test, and the backend suite41 passed in3.3s.
GPU9's optional baseline environment includes SB3/Gym coverage unavailable in
the GPU7 optional-MPPI environment, so collection/pass counts differ.
First-party syntax:157 files; current-facing local-link check:30 documents,
zero broken paths. Existing upstream AST deprecation warnings remain; no
physics, learning efficacy or real-robot calibration claim follows from tests.

## 2026-11-17 — Shared-run success audit (training logs, read-only)

Source: episodes.jsonl of the live shared runs (frozen source 5ec9f85), read at
GPU7 iteration 285 and A6000 iteration 849 (about 16:00 CET). Training-time
stochastic episodes, not frozen deterministic evaluation. "First" = episode 0
of every world. XY = median completion_xy_error_m over successes: robot-base
planar end-effector distance to the prescribed target (tolerance <40 mm), not
decomposed into along-blade and cut-normal components.

| Run / object | First success | Recent success | XY first→recent mm | Steps first→recent | Separated among recent successes |
|---|---:|---:|---:|---:|---:|
| GPU7 DiSECt cucumber | 558/1024 | 1117/1117 | 2.3→4.7 | 128→20 | 160 |
| GPU7 DiSECt half potato | 355/1024 | 1039/1039 | 3.8→19.2 | 139→20 | 0 |
| GPU7 DiSECt prism | 537/1024 | 520/844 | 19.7→22.9 | 152→22 | 0 |
| A6000 BladeMaster cucumber | 170/170 | 495/495 | 1.4→26.6 | 133→19 | not logged |
| A6000 BladeMaster half potato | 42/170 | 503/503 | 2.0→26.0 | 165→19 | not logged |
| A6000 BladeMaster prism | 170/170 | 502/502 | 1.3→26.9 | 128→19 | not logged |

GPU7 "recent" = last 3000 episodes; A6000 = last 1500.

First-episode meaning differs per run. GPU7's first policy update is at
iteration 115; only 35 of its 3072 first episodes ended before it (about 3030
after), so GPU7 "first" is NOT an untrained-policy rate. A6000's first policy
update is at iteration 689 and all 510 first episodes ended before it: 382/510
(75%) is the success rate of the not-yet-updated policy with training
exploration.

Measured: recent successes arrive in far fewer control steps than first
episodes and end with larger planar XY offsets (about 19–27 mm except GPU7
cucumber), inside the 40 mm tolerance. Among GPU7 recent successes, half potato
and prism never completely separate; BladeMaster separation is not logged.
The current +100 does not require separation or check the intended slice.

Not established: whether the offset means the blade bypasses or skims the
object (needs blade/cut-plane/object relative positions from trajectories);
whether low peak force means no penetration; the actual descent speed (episode
step counts alone do not give it); and BladeMaster separation. Exploitation of
the permissive final-pose region is a hypothesis to test, not a finding. These
pose-success rates must not be reported as cutting or separation success.

Separately confirmed by code reading (both affect the live runs' imagination,
size not measured): the physical board stop latches after contact while the WM
board projection still allows later lateral/upward motion; the learned
termination head reduces imagined survival without settling a terminal reward.

## 2026-11-17 — Shared runs end; cut-geometry check; BladeMaster separation

Run endings: see TRAINING_SHARED_20261116 "Final status" (A6000 budget_complete,
482,460 transitions; GPU7 failed at 22.4 h by CUDA allocation failure,
933,888 transitions).

Cut geometry (read-only, robot-base frame = world here). Blade edge along +x
(150 mm), cut-plane normal y; the policy actions are x, z, pitch, so y is not
controlled: measured blade y spread over all 3072+510 snapshot worlds and the
evaluations <= 0.2 mm, within 1 mm of the DiSECt cut-spring plane. The final
XY error is therefore along the blade edge, not a miss of the cut plane.

policy527 frozen evaluation (DiSECt checkpoint_00000224, 1 world/object):

| Object | Final x offset mm | y offset mm | Blade covers object width | Depth | Object drag x mm | Released springs | Separated |
|---|---:|---:|---:|---:|---:|---|---|
| Cucumber | +11.4 | -0.25 | 1.00 | 1.00 | 1.7 | 64/65 | no |
| Half potato | +28.3 | -0.20 | 1.00 | 0.98 | 22.9 | 94 | no |
| Prism | +28.3 | -0.19 | 0.96 | 1.00 | 20.2 | 275 | no |

Descent from ~55 mm to ~3 mm above the board in ~5 control steps (fastest step
~0.43 m/s; targets 0.03–0.046 m/s). After reaching the board, on potato and
prism the blade slides ~2 mm/step along +x to +28 mm and the object centroid
follows almost one-to-one (relative drift 5–8 mm). Early training (snapshots at
iteration 1 vs 120, worlds still in their first episode, both backends): median
x offset ~0 mm, object displacement < 1 mm. Measured: the drag-along-edge
behaviour appears with training; the pose success region admits it.
Not measured yet: late-training episodes (no per-step geometry was recorded; a
frozen evaluation with --record-geometry of the final checkpoints is running).

BladeMaster separation: material_components counts code-compatible grid
connectivity components (blademaster_backend.py, topology.components) at
termination; damage_saturated_fraction is a thin-band damage fraction. Neither
is a "prescribed slice detached" indicator (fragments, board-pinch labelling,
no plane check). BladeMaster separation is therefore reported as not
determinable (null), not as "not separated". DiSECt separation (all prescribed
cut springs released) and BladeMaster connectivity are different criteria and
must not be pooled.

## 2026-11-17 — BladeMaster v3 cut-topology diagnostics: first real run

Logging-only fields (TASK_FREE_CUT, v3 section). GPU7, shared scene_v1 (v3),
2 worlds per object, scripted reference descent (`reference=True`, sampled
speeds), 2 episodes per world, 151 vector steps, 461 s. All 12 episodes ended
in pose success. Driver and records:
runs/mem_headroom_20261117/{bm_separation_run.py,bm6_separation.json}.

| Object | Particles | Component sizes | Side-coded fraction | Side 1 / 2 | Band (±36 mm) particles | Band damage>=1 | Boundary offset (mm) | Code-0 bridges |
|---|---:|---|---:|---|---:|---:|---:|---:|
| cucumber_003 | 6258 | 3496, 2762 | .475 | 1466 / 1504 | 2970 | 0 | .8 | 0 |
| potato_half | 2777 | 1409, 1368 | .960–.961 | 1318 / 1347–1350 | 2668 | 0 | .8 | 0 |
| prism | 18513 | 9537, 8976 | .727 | 6732 / 6732 | 13464 | 0 | 1.0 | 0 |

Measured: in every episode the native topology had two components, split by
side codes, while no particle in the band (or overall) reached damage 1. This
agrees with the source reading: with the external tool, side codes follow edge
passage, not damage, so a two-component count here records where the scripted
edge went, not fracture. The band (6 x 6 mm grid spacing) covers most of the
small objects. Component sizes repeat exactly across episodes and worlds of an
object despite randomized material/speed. Not established: behaviour under learned
policies (partial or lateral cuts), and whether the zero damage is expected for
these presets. The candidate flag was true for all 12; it is not a separation
result.

## 2026-11-17 — GPU memory headroom: 3072-world DiSECt (GPU7)

Question: why the 2026-11-16 GPU7 run failed in a Warp allocation during reset at
iteration 304, and whether a cache/allocator default restores headroom. GPU7
(RTX 4090, 24564 MiB), only these processes on the card. Real v3 config
(prepared 3072 worlds, warmup 30720, microbatch 512, critic microbatch 8192,
real critic, actor horizon 8, BC 0.25, seed 0), but 3072 transitions per update
and 24 WM-only rounds, 44 iterations (10 warmup, 24 WM-only, 10 joint rounds).
No episode ended in any run, so no masked reset was measured. Init 463-465 s.
Records: runs/mem_headroom_20261117/{baseline_joint,fix_release,
fix_release_expandable}/steps.jsonl (gpu_memory_window per iteration).

Physics rows use iterations 13-35 (after WM-only rounds) and 36-44 (after joint
rounds 1-9); reserved is the physics-phase peak, free is device free memory at
physics start (after any release). MiB.

| Run | Torch reserved in physics, WM-only / joint | critic_data peak reserved / allocated | Free at physics start, WM-only / joint |
|---|---:|---:|---:|
| baseline (no release) | 16618 / 19034-19038 | 19034-19038 / 18401-18408 | 6776 / 4352-4356 |
| release before physics/reset | 16168-16238 / 16324-16532 | 19002-19122 / 18404-18409 | 7156-7246 / 6858-7066 |
| release + expandable segments | 16126-16188 / 16272-16398 | 18766-18930 / 18335-18339 | 7206-7288 / 6992-7118 |

Warp pool high-water was 841 MiB in all runs, reached during initialization
(full initial reset) and not exceeded afterwards. After joint rounds, release
returned 2.90-3.01 GiB (release only) and 2.73-2.89 GiB (with expandable
segments) of idle Torch cache before each physics step. Without release, the
first joint round (critic_data peak) raised Torch reserved by 2.4 GB and kept it;
it then stayed flat for 9 rounds. The long run's further growth (device 16.7 to
23.8 GB, W&B) was not reproduced, so its cause (possibly variable-size
allocations once episodes end) is not established. With release only, the
physics reserved peak still rose 16324 to 16532 MiB over joint rounds 1-9;
with expandable segments it stayed within 16272-16398 MiB.

Expandable segments with DiSECt Torch CUDA graphs: initialization (settle graph
capture) succeeded. First 10 warmup steps vs baseline at seed 0: reward mean
0.178512 vs 0.178511, max abs per-step reward difference 4.2e-3; two baseline
runs differ by 3.7e-3 (DiSECt is not bitwise deterministic). Default for
current runs: release always + expandable segments + headroom guard (1 GiB
margin over Warp high-water, armed after the first joint round; METHOD). In
the verification run the guard required 1865 MiB with 6992-7118 MiB free.
Not established: behaviour over hours, under resets and with episode-dependent
critic windows; 3072-world long runs remain unverified for memory.
