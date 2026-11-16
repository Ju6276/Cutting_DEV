# Vision route 1 — prescribed cut, calibrated depth (2026-11-17)

## Implemented scope

User requested the existing prescribed cut, without enlarging the task or retraining
the low-level policy. This implementation produces a **pre-cut plan**, not a
hardware driver or a completed vision-conditioned physical rollout.

`cutting_world_model/visual_cut.py` accepts metric optical-axis depth,
pinhole intrinsics, camera-to-robot-base calibration, calibrated board geometry,
known robot self-occlusion, and the existing mounted blade / prescribed EE goal.
It estimates the board height offset and visible food top, checks that the
prescribed cut crosses the visible food footprint and fits within the blade,
then returns a starting EE pose and the current eight-dimensional policy goal.
It never uses the object mesh/category/material as estimator inputs.
Mesh surface truth is used separately in the synthetic audit.

The policy goal remains `[x,y,z,qx,qy,qz,qw,target_speed]` in **robot base**
coordinates. A goal-only coordinate change does not make the policy invariant
to arbitrary object positions. The policy still needs real joint/motion/wrench
observations, with exactly the trained normalization and ten-frame history.
No image is added to its observation.

## Camera and cut selection

The proposed fixed camera is beyond the workpiece in +base Y, 320 mm in front
of the board centre and 550 mm above it, looking back and down. Its optical axis
is about 30 degrees from vertical. At 640 by 480 pixels, focal length is 700 px.
These are synthetic design values, not measured camera calibration.

The existing single metallic physical blade and original robot/handle visual
assets are reused. There is no blue overlay knife or invented mounting rod.
The camera preview is a sensor view; it does not replace the studio video renderer.

Current route: **prescribed XY cut plane and orientation**. Vision estimates
height and checks whether that plane intersects the observed object. If it
misses, reject the plan; do not secretly relocate the DiSECt cut interface.
Start clearance is 20 mm plus an explicit 1 mm height-estimation margin, within
the trained 5–30 mm range. Speed is a caller command, default 3 cm/s, within
1–5 cm/s. The margin and clearance together must fit within the allowed range.
Finite blade mounting, not just the EE origin, determines the goal at the board.

Route 2 would learn to predict a cut goal from an image/point cloud. The label
must come from a specified cutting objective: currently the same prescribed
plane; later, for example, a specified slice thickness from a chosen end.
It does not discover the desired cut solely from recognising a vegetable.
Training that predictor is separate from retraining the low-level policy;
deployment outside the low-level task distribution still requires validation.

## Commands

From the GPU7 repository, render a calibrated camera from an existing recording:

```sh
PYTHONPATH=.:scripts /dataHDD/dong/Envs/render-20261108/bin/python scripts/capture_visual_cut_camera.py \
  --recording runs/shared_scene_views_20261116/disect/geometry/0000/group_00 \
  --appearance-scene runs/shared_scene_views_20261116/disect/cucumber.npz \
  --source-root /opt/ros/noetic/share/ur_description \
  --source-root /dataHDD/dong/Projects/SliceIt/catkin_ws/src \
  --camera configs/vision/front_camera_v1.json \
  --config configs/vision/prescribed_cut_v1.json \
  --output runs/NEW_CAMERA_CAPTURE
```

Offline planning also accepts real camera exports after calibration:

```sh
PYTHONPATH=. .venv/bin/python scripts/plan_visual_cut.py \
  --depth runs/NEW_CAMERA_CAPTURE/depth.npz \
  --camera configs/vision/front_camera_v1.json \
  --config configs/vision/prescribed_cut_v1.json \
  --output runs/NEW_CAMERA_CAPTURE/plan_checked.json
```

NPZ fields: `depth_m`, optional boolean `robot_mask`. Depth means camera-Z,
not Euclidean ray range; convert sensor units explicitly. Optional
`--checkpoint PATH` validates goal/observation schema, coordinate convention,
task version and blade/board geometry against the frozen checkpoint, on CPU.
It neither runs that policy nor synthesizes its observation history.

Outputs: `plan.json`, visible point cloud, RGB/depth, diagnostic overlay and
separate geometry audit. The yellow overlay is the prescribed blade line at
the board, not a visually discovered cut or material-separation prediction.
The teal overlay marks selected visible foreground.

## Recorded-scene verification

GPU7: `/dataHDD/dong/Projects/Cutting_DEV_board/runs/vision_route1_20261117/`.
Three existing, settled DiSECt-material scene recordings; current shared UR5e
and blade. Ideal depth includes occlusion by the recorded robot and blade.
Self masking uses known robot projection, not food instance labels.

| Object | Visible top estimate | Physical surface top | Approximate absolute error |
|---|---:|---:|---:|
| Cucumber | 38.35 mm | 38.69 mm | 0.34 mm |
| Half potato | 25.89 mm | 26.03 mm | 0.14 mm |
| Prism | 49.73 mm | 49.76 mm | 0.03 mm |

All three passed visible-footprint and workspace checks. These are single
recorded frames, not sensor accuracy statistics or policy completion rates.
The 1 mm engineering margin covered these observed top errors; it is not a
guarantee for occluded/noisy real objects.

Current frozen policy checkpoint128 (sha256
`3bf6d7017bcfe31b5e38da68d61372156c2c85a77e8328b84989e126d138a8fd`)
passed the CPU goal-interface check. No checkpoint weights were changed.

## Execution boundary and remaining work

A robot application must first move to the planned starting pose through its
checked pre-cut motion, update the controller target consistently, and initialize
history from actual measurements. Only then can the frozen policy execute.
This task has implemented and tested perception/goal conversion; **a physical
episode initialized from this visual plan has not yet been run**. Current long
runs keep their original simulation reset and never consume these plans.

Real use needs measured camera intrinsics/extrinsics, tool/board calibration and
visibility/noise validation. The estimator knows the board normal and only fits
a small height correction; it does not estimate an arbitrary tilted board.
A largely hidden top or multiple foreground objects can require another view.
The board stopping margin remains the simulation contract, not a real-robot
contact tolerance. Visible footprint coverage does not prove full material
separation. No claims of real deployment or arbitrary cut generalization.

## Regression verification

GPU9 RTX 4090: root suite **589 passed, 3 skipped**; bundled backend **41 passed**.
The new vision module contributes 12 passing geometry/interface tests.
Both suites include unchanged GPU paths; these checks shared GPU9 with the
already-running SAC job and are not throughput benchmarks. New depth rendering
was verified on GPU7. Running training checkouts were not modified.
