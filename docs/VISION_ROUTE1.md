# Visual goals — geometric and learned centers (2026-11-17)

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
A6000 run had no policy updates when this trial was run (it entered joint
training on 2026-11-17 09:40). This is explicitly **cross-backend frozen
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
| Frozen WM+BPTT policy | Implemented | Implemented for geometric and learned selectors in one BladeMaster world; both latest trials timed out |
| Vector SAC | Implemented; bounded GPU9 training finished (1,281,024 transitions) | Not implemented |
| Frozen WM+MPPI | Implemented | DP3 goal executed in native BladeMaster; one paired WM566 transfer trial timed out |
| Nonlearned approximate FDCC+MPPI | Implemented | DP3 goal executed; one nominal case reached pose goal in105steps |

evaluate_visual_cut.py dispatches --method wm_policy|wm_mppi|analytic_mpc.
Its current physical adapter is one native-BladeMaster world. SAC remains
available only in evaluate_pose_policy.py, without a visual-goal handoff.
There is no arbitrary DiSECt topology re-export or real-robot approach planner.

## Method 2: fixed-heading learned midpoint

Current default: official DP3 encoder; see the DP3 replacement result below.
The following records the initial local_v1 pilot for reproducibility.

Initial local implementation: visual_center_features.cloud_input uses512 visible points,
subtracts the **observed** XY centroid, and measures height from the observed
board. Coordinates use a fixed0.1m scale; object size is not normalized away.
The network receives no absolute XY, object ID, full mesh or target. A shared
pointwise MLP3-32-64-128 with mean/max pooling and256-64-1 head predicts an
offset from visible centroid along fixed +base Y. The output is bounded to
+-50mm. This is a small local model, not a full PointNet++ reproduction.

Full rendered-surface bounds generate offline labels; "middle" means the
midpoint of the complete bounds along the fixed cut normal, not equal volume.
The three input surfaces are the existing settled cucumber003, half-potato and
prism studio snapshots.900 cases =300 unique static XY placements per shape
on a20x15 grid covering +-15mm, with no rotation. They are rendered from the same
front camera. These translated surfaces are perception data, not900 newly
settled physical simulations. No real depth noise or robot occlusion is modeled
in this dataset; the physical handoff masks the known parked robot projection.

A shared placement-ID partition gives200/50/50 train/validation/test cases per
object (600/150/150 total). No placement's repeated view crosses splits.
The test contains seen shapes and nearby unseen positions, not unseen objects.
CPU training120 epochs, Adam.001, batch32, seed20261117; validation MAE selects
epoch117. The test was opened only after selection. No test-driven retuning.
Online W&B run:
[abn7qktr](https://forge.coreweave.com/wandb/ju-dong6276-technical-university-of-munich/Cutting_DEV/runs/abn7qktr).
Remote state finished and120 logged epochs were verified.

Held-out midpoint MAE: learned0.210mm, fixed-heading visible-extent midpoint
5.499mm, constant world coordinate7.685mm, visible centroid15.129mm.
The learned predictor can exploit these three known shape/view priors.
These numbers do not demonstrate novel geometry reasoning or real localization.

### Commands

Run from the prepared GPU7 development checkout, preserving the source assets:

~~~bash
PYTHONPATH=.:scripts /dataHDD/dong/Envs/render-20261108/bin/python scripts/generate_visual_center_data.py --scenes runs/shared_scene_views_20261116/disect --output runs/NEW_CENTER/dataset --wall-seconds 1800
source runs/sac_env.sh
PYTHONPATH=. /dataHDD/dong/Envs/cutting-mppi-20261108/bin/python scripts/train_visual_center.py --data runs/NEW_CENTER/dataset --output runs/NEW_CENTER/learning --epochs 120 --wall-seconds 900 --wandb-mode online
PYTHONPATH=. .venv/bin/python scripts/plan_visual_cut.py --selection learned-midpoint --center-checkpoint runs/NEW_CENTER/learning/center.pt --depth RUN/depth.npz --camera RUN/camera.json --config RUN/visual_config.json --output RUN/learned_plan.json
~~~

No retraining of WM/policy is required for this interface trial; that is not
a guarantee that an arbitrary new visual goal is executable by the policy.
The approved physical trial used the same source policy66 as the earlier route1
trial, explicitly transferred to native BladeMaster, with nominal cucumber
and fixed heading. Reset/IK places the start; real approach planning is absent.

Add --selection fixed-midpoint to evaluate_visual_cut.py for the geometric
fixed-heading reference. For the learned route add --selection learned-midpoint
--center-checkpoint MODEL. Both reuse the same camera, task checks,8D goal,
policy history, physical loop, recording and shared studio rendering. The original
--selection visible-midpoint remains the geometric PCA-heading prototype.
The fair fixed-heading comparison does not silently rename it as learned vision.

GPU9 results: runs/visual_center_closedloop_20261117.
Both180-step episodes timed out; neither exceeded force/workspace limits.
Geometric/learned plane errors8.852/0.688mm, final goal-height errors2.496/2.890mm
against a2mm tolerance, and peak material forces2.747/8.605N.
Both retained one connected material component. Improved localization did not
make this frozen cross-backend controller complete the task. This is one nominal
case per route, not a success-rate estimate or causal claim about force changes.

The dataset/checkpoint stay on GPU7 under runs/visual_center_20261117.
The full shape-source hashes, split membership, depth/RGB, input point clouds,
labels, validation history and final test predictions are retained there.
No active long-training configuration or low-level model was changed.

## DP3 replacement result — 2026-11-17

New training defaults to --architecture dp3_xyz_v1; old checkpoints still load
local_v1. The point encoder is pinned official DP3 code (see METHOD/SOURCE.json),
with a task-specific midpoint regression head, not an action diffusion network.
Inputs/outputs and local-base-axis coordinate convention remain unchanged.
On three matched seeds, mean midpoint MAE improved0.2013 ->0.1370mm.
This is known-shape/small-translation development evidence, not real deployment.
The seed20261117 frozen-policy66 physical handoff still timed out; plane error
0.8453mm vs previous local0.6880mm. Improved aggregate perception does not imply
better control. See RESULTS for the complete negative outcome and limits.

~~~bash
source runs/sac_env.sh
PYTHONPATH=. /dataHDD/dong/Envs/cutting-mppi-20261108/bin/python scripts/train_visual_center.py --data runs/visual_center_20261117/dataset --output runs/NEW_DP3 --architecture dp3_xyz_v1 --seed 20261117 --epochs 120 --wall-seconds 900 --wandb-mode online
PYTHONPATH=. /dataHDD/dong/Envs/cutting-mppi-20261108/bin/python scripts/compare_visual_centers.py --runs runs/dp3_center_20261117
~~~

Existing plan_visual_cut/evaluate_visual_cut commands accept the new center.pt
without changing the policy or goal interface. Do not load DP3 weights into the
legacy local network; architecture dispatch is explicit in the checkpoint.

## DP3 controller video commands and result

Use the same --policy goal-convention reference, --backend-checkpoint,
--center-checkpoint, --render-python and --selection learned-midpoint arguments
as the frozen RL trial. For WM planning add:

~~~bash
--method wm_mppi --model-checkpoint FULL_MATCHING_POLICY66_CHECKPOINT --allow-backend-transfer --planner-horizon 8 --planner-samples 512 --planner-refinements 1
~~~

For nonlearned control add --method analytic_mpc --analytic-substeps 5.
It uses the reference checkpoint only for goal convention/task metadata;
learned weights are not used for action prediction. Both planners use MPPI
with no terminal critic. Do not call it a reproduced cutting-paper MPC.

GPU9 runs/dp3_controllers_20261117 retains full runs/frames/force traces.
The RL video reuses runs/dp3_center_closedloop_20261117/episode (policy66).
Both RL and WM planning explicitly transfer DiSECt-trained weights to native
BladeMaster; this is not the newer policy527 and not matched training across
backends. DP3 is the pinned point encoder plus our midpoint head.

One case per controller: RL and WM+MPPI timed out at180steps; analytic MPC
reached pose success at105steps. Peak material forces10.84/10.62/10.81N.
The analytic case has two material components; this count alone is not a
validated intact intended-slice metric. Planning averaged0.469/0.460s per
step, exceeding the50ms physical control period. Offline simulation videos do
not demonstrate real-time control. See RESULTS for full limitations.
