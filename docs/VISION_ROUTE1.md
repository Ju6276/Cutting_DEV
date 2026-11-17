# Vision route 1 — depth-selected cut and frozen policy (2026-11-17)

## Current implementation

The user requested a plane selected from vision, followed by a physical cutting
trial. The earlier prescribed-plane prototype is retained explicitly as a
height/interface diagnostic; it is not the current plane-selection method.

visual_cut.visual_midpoint_plan receives metric depth, camera/board calibration
and mounted-tool geometry/orientation. It never receives prescribed target XY.
Known robot pixels are masked, the largest foreground component is segmented,
and horizontal PCA identifies its visible major axis. The selected cut is the
midpoint of the 2–98% visible extent along that axis; its normal follows the axis.
A near-round visible shape (axis ratio below 2) rejects rather than silently
falling back to the old plane. This geometric rule is not an optimal cutting
location or a full-volume midpoint. The prototype restricts heading changes to
15 degrees; that restriction does not establish policy generalization.

Calibration converts blade-edge placement into a starting EE pose and the 8D
policy goal [x,y,z,qx,qy,qz,qw,target_speed] in robot-base coordinates. Raw goal
quaternions must use the frozen policy's sign convention: q and -q are physically
equal, but a network consuming the raw coordinates need not treat them equally.
The optional policy_quaternion_reference_xyzw field supplies that convention.
No object category, material label or food mesh enters the selector. It uses
visible depth only. Synthetic mesh truth is audited separately.

## Camera and commands

The camera is +base Y of the workpiece, offset [0,0.32,0.55] m from board centre,
looking back/down about 30 degrees from vertical. Image640x480, focal700 px.
These are synthetic values, not real camera calibration. Pre-cut clearance is
20 mm plus a 1 mm height margin (within the training5–30 mm range); commanded
speed is3 cm/s (training1–5 cm/s). Board normal is calibrated, only board height
is refined. Unknown/occluded tops or multiple objects can require another view.

~~~sh
PYTHONPATH=. .venv/bin/python scripts/plan_visual_cut.py \
  --selection visible-midpoint \
  --depth RUN/depth.npz \
  --camera configs/vision/front_camera_v1.json \
  --config configs/vision/visual_midpoint_v1.json \
  --output RUN/visual_plan.json
~~~

NPZ inputs: depth_m (optical-axis depth, not ray distance), optional robot_mask.
Optional --checkpoint validates the existing34D observation/8D goal contract,
base frame and blade/board calibration. It does not run the policy. Goal-relative
coordinates alone do not guarantee out-of-distribution robustness.

capture_visual_studio.py renders the actual settled physics snapshot into RGB
and metric depth, then selects the plane. RGB colors are the shared display
palette, not a material measurement. The video still uses the approved shared
studio renderer, single metallic blade and original robot/handle, with no blue
knife overlay or invented adapter rod. Yellow lines show the selected plane's
intersection with the board, not a material-separation prediction.

## Bounded physical trial

scripts/evaluate_visual_cut.py executes one native BladeMaster cucumber world.
DiSECt's baked cut topology is not silently relocated to support a new plane.
It uses policy66 from GPU7 checkpoint128 (original checkpoint SHA256
3bf6d7017bcfe31b5e38da68d61372156c2c85a77e8328b84989e126d138a8fd), since the new
A6000 run still has no policy updates. This is explicitly **cross-backend frozen
inference**, not evaluation of the eventual A6000-trained actor.

The handoff verifies controller/task/observation compatibility, then uses
simulation IK/reset placement before the episode, preserving material positions.
This is not a collision-checked real approach trajectory. The policy thereafter
receives measured simulated motion/force history and the visual goal at20 Hz.
This is one-shot pre-cut vision followed by motion/force feedback, not continuous
visual servoing. The trial is bounded to one180-step episode with a25-minute
external timeout; no weights, reward, material law or active training job change.

~~~sh
PYTHONPATH=.:backends/mujoco timeout --kill-after=20s 1500 \
  .venv/bin/python scripts/evaluate_visual_cut.py \
  --policy RUN/frozen_policy66.pt --backend-checkpoint RUN/bm_checkpoint192.pt \
  --allow-backend-transfer --wall-seconds 1450 \
  --render-python /dataHDD/dong/Envs/vision-render-20261117/bin/python \
  --output RUN/NEW_TRIAL
~~~

Outputs retain visual plan, provenance/source hashes, conditions, all physical
transitions, per-step geometry and final task/force/material diagnostics.
Do not interpret pose completion or reconstructed surface appearance as verified
material separation. The active native policy is not modified by this test.

## Evidence and provenance

GPU7 runs/vision_variants_20261117 contains seven independently scanned assets:
cucumber001/002/003, potato001/002/003, apple002. They come from
[OmniObject3D](https://omniobject3d.github.io/) through the AlbertS666 HF mirror;
source metric scale and textures are retained, with download/hash ledgers.
Five pass the visual-axis rule. Potato003 and apple002 reject as ambiguous.
These static ideal-depth examples have no robot occlusion and have not been
physically settled or cutting-validated. Whole potato scans differ from the
half-potato training asset. Scans supply geometry/colors, not calibrated physics.

Earlier prescribed-plane checks in runs/vision_route1_20261117 used three settled
recordings: cucumber/half-potato/prism visible-top errors0.34/0.14/0.03 mm. Those
are single-frame ideal-depth comparisons, not real sensor accuracy. Their yellow
lines were prescribed. Keep that older video distinct from visual-selection
results; its 589/3skip root and41 backend checks are historical verification.

Current physical trials are on GPU9 in runs/vision_closedloop_20261117. Early
attempts are retained: trial01 startup exit139 (cause unconfirmed); trial02 IK
comparison fixed for quaternion sign equivalence; trial03/04 renderer dependency
and missing snapshot identity errors. Trial05 completed but used the opposite
raw goal quaternion sign from the source policy; it is an interface diagnostic,
not the final transfer result. Trial06 uses the source goal convention. See
RESULTS for measurements and final regression counts.

The GPU9 rendering env is Python3.8 / VTK9.5.2 / pycollada0.8; pycollada0.9.3 uses
functools.cache unavailable on3.8. Prefer the standard Python3.12 render env for
new installations. Tests share GPU9 with SAC and are not exclusive throughput
measurements. No source/weights/data are copied to Mac; requested previews are.

Real deployment additionally needs measured camera/tool/board calibration,
noise/visibility checks, an actual approach controller and validated low-level
goal coverage. The simulation board stop is not a real contact tolerance.

Final verification: root596 passed/3 skipped, backend41 passed;19 visual tests.
The completed physical handoff timed out at180 steps (blade gap2.476 mm),
without force-limit failure. It demonstrates execution, not task success.

## Integration status and next pilot — 2026-11-17

| Controller | Current-task evaluator | Visual-goal physical execution |
|---|---|---|
| Frozen WM+BPTT policy | Implemented | Implemented for one BladeMaster world; trial06 timed out |
| Vector SAC | Implemented; bounded training launched separately | Not implemented |
| Frozen WM+MPPI | Implemented | Not implemented |
| Nonlearned approximate FDCC+MPPI | Implemented | Not implemented |

evaluate_visual_cut.py currently loads SliceItPolicy; it is not a general
controller dispatcher. evaluate_pose_policy.py supports the four controller
types but does not consume visual plans. Shared 8D goal semantics alone do
not establish tested visual integration.

Method 2 is the fixed-heading, centered-cut supervised pilot in [PLAN](PLAN.md).
The geometric midpoint rule supplies labels; there is no trained visual network
yet. Unlike route 1's estimated PCA heading, this pilot uses a known fixed
heading, so a round silhouette does not require inventing an orientation.
Keeping all poses fixed can make a constant predictor sufficient; this stage
tests the pipeline only, with generalization deferred by user choice.
