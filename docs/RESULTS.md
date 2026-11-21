# Results

Measured results whose conclusions still hold for the current code and task
(pose_depth_v3, `project_continue_v2`, 500 steps). Each entry gives the result
and its server evidence path. Superseded entries (free_cut, pose-v1, v2 interface
checks, latch-era V3 checks, fixed-prism development) were removed; the full
previous record is [RESULTS at d7e05a2](https://github.com/Ju6276/Cutting_DEV/blob/d7e05a2/docs/RESULTS.md).
[PLAN](PLAN.md) says what happens next. UR10e integration evidence is in
[UR10E_INTEGRATION_REVIEW](hardware/UR10E_INTEGRATION_REVIEW.md).

## Board continuation (`project_continue_v2`)

**Physical probe, 2026-11-19** (`scripts/check_board_continue.py`, GPU7, two
cucumber worlds per backend, normal force limits, explicit FDCC targets: press,
hold, 1 s of pure inward target motion at each speed, 10 mm lateral slide, lift,
partial reset). Clean checkouts: 08bb59f (v2) vs f12ca04 (v1). EE tangential
drift / rotation over each 1 s inward phase (DiSECt / BladeMaster):

| Speed | f12ca04 drift | f12ca04 rotation | 08bb59f drift | 08bb59f rotation |
|---:|---:|---:|---:|---:|
| 0.5 cm/s | 0.80–0.81 mm / 0.80 mm | 0.0122–0.0123 rad / 0.0122 rad | 0.068–0.070 mm / 0.069–0.070 mm | 0.84–1.17 mrad / 0.61 mrad |
| 2 cm/s | 4.44–4.46 mm / 4.42 mm | 0.047 rad / 0.047 rad | 0.26–0.27 mm / 0.30–0.31 mm | 1.4–1.9 mrad / 2.4 mrad |
| 5 cm/s | 17.9 mm / 17.9 mm | 0.110–0.111 rad / 0.111 rad | 0.74–0.76 mm / 1.01–1.04 mm | 5.1–6.1 mrad / 6.1 mrad |

The projection itself moved the EE <= 3.5e-12 m and rotated it <= 5.6e-13 rad at
08bb59f (f12ca04: up to 81 mm path length and 0.34 rad). The remaining drift is in
the FDCC proposals (controller response to the unreachable target and material
wrench), not added by the projection; the controller mechanism was not analysed
separately. Final gaps 1.17–1.53 um, no force-limit event. The lateral phase moved
9.985–9.998 mm for the commanded 10 mm (measured from the phase start; reanalysis
2026-11-20). Hold, lift (contact cleared) and partial reset passed; first
boundary step DiSECt 38–39, BladeMaster 36. Material particles, sensor queue and
simulation time kept advancing at the boundary.
Evidence: GPU7 `runs/board_normal_20261119/audited/{after,before}_{disect,blademaster}/`;
slide reanalysis `runs/review_reset_contact_20261120/slide_reanalysis.json`.

**Kinematic test** (no controller/material; synthetic 10 cm edge on the UR5e
chain, 10 ms command steps): v1 drift 2.0/8.0/19.8 mm and 0.010/0.040/0.100 rad
at 0.5/2/5 cm/s; v2 < 1e-12 m and < 2e-13 rad (tests/test_board_normal_projection.py).
One v2 projection + velocity projection took 55 ms vs 44 ms for v1 on the 4090
(microbenchmark, not an end-to-end real-time validation).

**Implementation corrections found during the continuation work:** a position
command must not be injected as MuJoCo generalized velocity (only the inward
component is removed); the FK second derivative needs the analytic
squared-angle series near zero angle (the norm-based form gave NaN). Both have
regression tests.

**Reset contact, 2026-11-20:** both physical adapters now recompute
`board_contact` from the restored rigid-blade FK at reset (a 1–10 um reset
previously disagreed with the WM); the two-backend regression failed before and
passed after the fix. Evidence: GPU7 `runs/review_reset_contact_20261120/`.

**Review of 9ea91b2 (2026-11-20):** no additional blocking defect confirmed;
spatial-6R gradcheck and gradgradcheck pass for the normal projection. Not
covered: projection timing, extreme singular cases, long-run memory, training
quality, hardware contact. Final validation at 51b2c0b: root 773 passed /
5 skipped, backend 44 passed. Evidence: GPU7 `runs/review_9ea91b2_20261120/`,
`runs/review_reset_contact_20261120/`.

## Shared scene interface (2026-11-16)

- Shared knife: three-object reference probe, maximum physical-tool/FK vertex
  discrepancy 1.02e-8 m (DiSECt) and 6.43e-9 m (BladeMaster); all 18 blades of a
  later three-object x three-world scene matched recorded EE kinematics within
  3.163e-9 m. Evidence: GPU7 `runs/tool_analysis_20261116/`,
  `runs/interface_recheck_20261116/geometry_check.json`.
- Start clearance: requested and actual per-world gaps differ by < 2e-7 m in both
  backends; partial reset leaves non-reset worlds' frames identical. Evidence:
  GPU7 `runs/shared_scene_20261116/`, `runs/interface_recheck_20261116/`.
- Paired materials: corresponding initial/reset material, speed and gap cases
  agree between backends to 5.83e-8. Evidence: GPU7 `runs/paired_material_20261116/`.
- Exactly 60 mm can compute as 0.06000000000000005 m after robot-frame subtraction;
  boundary comparisons now allow eight epsilons at metre scale. Counterfactual
  60 mm fails 0/9 and 61 mm 9/9 worlds in path mode only. Evidence: GPU7
  `runs/interface_motion_review_20261116/tool_*_final.json`.
- Position-mode demonstration (one cucumber case, scripted 75 mm lateral
  excursion and return, 180-step horizon at the time): path mode ends at step 25
  (tool_motion failure), final-only returns and succeeds at step 58; maximum
  planar deviation 73.65 mm, wrench utilization 0.318. Evidence: GPU7
  `runs/tool_position_demo_20261116/recording_final/comparison.json`.

## Training lifecycle (2026-11-16)

Pending unstarted learning rounds are checkpointed and resumed before further
collection; a saved WM milestone with two pending rounds resumes both before any
new physical step, with update counts equal to the uninterrupted runner.
Environment shutdown precedes tracker finalization and a forced close failure
reaches progress and W&B. Evidence: GPU7 `runs/runner_audit_20261116/`.

## GPU memory headroom, 3072-world DiSECt (2026-11-17)

GPU7 RTX 4090 (24564 MiB), real v3 config, 3072 transitions per update, 24
WM-only rounds, 10 joint rounds; no episode ended, so no masked reset was measured.

| Run | Torch reserved in physics, WM-only / joint | critic_data peak reserved / allocated | Free at physics start, WM-only / joint |
|---|---:|---:|---:|
| baseline (no release) | 16618 / 19034-19038 | 19034-19038 / 18401-18408 | 6776 / 4352-4356 |
| release before physics/reset | 16168-16238 / 16324-16532 | 19002-19122 / 18404-18409 | 7156-7246 / 6858-7066 |
| release + expandable segments | 16126-16188 / 16272-16398 | 18766-18930 / 18335-18339 | 7206-7288 / 6992-7118 |

(MiB.) Warp pool high-water was 841 MiB, reached during initialization. Release
returned 2.7–3.0 GiB of idle Torch cache before each physics step after joint
rounds; with expandable segments the physics peak stayed within 16272–16398 MiB.
Expandable segments with DiSECt CUDA graphs reproduced warmup rewards within
run-to-run nondeterminism (4.2e-3 vs 3.7e-3 between two baseline runs). Default
since then: release always + expandable segments + 1 GiB headroom guard. The
2026-11-16 long run's growth from 16.7 to 23.8 GB was not reproduced; behaviour
over hours with resets is not established.
Evidence: GPU7 `runs/mem_headroom_20261117/`.

## Evidence validators and reference audit (2026-11-19, a1852dc)

`scripts/verify_entry.py` requires explicitly selected stages, checks diagnostic
completion/content and artifacts, and exits nonzero on failure;
`scripts/verify_resume.py` checks the full learner tree (including world_model
tensors and optimizer state) and replay rows. Both reproduced defects of the
earlier temporary checkers (a NaN injected into the WM passed the old check) and
now fail them. Explicit DiSECt reference trajectories take precedence over the
shared descent; training, SAC, evaluation and diagnostics use action-driven steps.
DiSECt reference reward audit at nominal material: 30/30 cases terminated,
equation errors <= 3.7509e-6; stall times out, normal press scores 107.98–108.47,
fast press 105.26–105.83; the prism force attempt triggers a force failure. Not
covered: BladeMaster reference audit, physical pending-round resume.
Evidence: GPU7/A6000 `runs/review_a1852dc_20261119/`, A6000
`runs/audit_pose_reward_362d15e_20261119/`.

## BladeMaster cut-topology diagnostics (2026-11-17)

Scripted reference descent, 2 worlds per object, 2 episodes each, all pose
successes. Every episode had two native components split by side codes while no
particle reached damage 1: with the external tool, side codes follow edge passage,
not damage, so a two-component count records where the edge went, not fracture.
Component sizes repeated exactly across worlds despite randomized material/speed.
`separated_bm_candidate` was true for all 12; it is not a separation result.
Evidence: GPU7 `runs/mem_headroom_20261117/bm6_separation.json`.

## Historical pose_depth_v2 long runs (evaluate-only)

- GPU7 DiSECt, 3072 worlds: failed at 22.4 h (CUDA allocation failure inside a
  physics reset), 933,888 transitions, 1406 WM / 906 policy updates; last named
  checkpoint 00000288. A6000 BladeMaster, 510 worlds: 24 h cap, 482,460
  transitions, 705 WM / 205 policy updates. GPU9 SB3 SAC: 1,281,024 transitions.
  Same cap, different interaction; compare only at equal transition counts.
- Training-log audit: recent successes arrive in ~20 steps with planar XY offsets
  of ~19–27 mm (inside the 40 mm radius, except GPU7 cucumber); GPU7 half potato
  and prism never completely separated among recent successes.
- policy527 frozen evaluation (DiSECt checkpoint 224): pose success on all three
  objects in 16 steps, no material separation. Blade y stays within 0.2 mm of
  the cut plane; the final offset is along the blade edge. Descent ~55 mm to ~3 mm
  in ~5 steps (fastest ~0.43 m/s vs 0.03–0.046 m/s targets); on potato and prism
  the blade then slides ~2 mm/step along +x to +28 mm and drags the object
  (~20–23 mm). Early snapshots show no such drag: the behaviour appears with
  training and the pose region admits it.
- Evidence: frozen checkouts `/dataHDD/dong/Projects/Cutting_DEV_shared_long_20261116/runs/shared_long_20261116/`
  (GPU7, A6000); GPU7 `runs/final_eval_20261117/`, `runs/policy_video_20261117/`.
  The final-checkpoint frozen evaluation statistics in `final_eval_20261117` are
  not yet verified.

## Vision (2026-11-17)

- Prescribed-plane depth estimator: visible-top errors 0.34/0.14/0.03 mm
  (cucumber/half potato/prism), ideal single-frame depth. GPU7 `runs/vision_route1_20261117`.
- Visible-major-axis selector on seven OmniObject3D scans: five pass, potato003
  and apple002 reject as ambiguous. GPU7 `runs/vision_variants_20261117`.
- Fixed-heading midpoint learning (900 cases, 600/150/150 split by placement):
  held-out MAE learned 0.210 mm, visible-extent midpoint 5.499 mm, constant
  coordinate 7.685 mm, visible centroid 15.129 mm. Known shapes only.
  GPU7 `runs/visual_center_20261117`.
- Official DP3 encoder vs local network, three paired seeds: mean MAE
  0.2013 -> 0.1370 mm; bootstrap mean difference -0.0643 mm, 95% [-0.0872, -0.0433]
  (reused development test set). GPU7 `runs/dp3_center_20261117`.
- Physical handoffs (native BladeMaster nominal cucumber, frozen DiSECt policy66
  transferred across backends, 180 steps, v2): geometric, learned and DP3
  selectors with RL all timed out (final height error 2.5–2.9 mm vs 2 mm
  tolerance); paired WM566+MPPI timed out (only 5 qualifying samples); nonlearned
  FDCC+MPPI reached pose success at step 105. Planning took ~0.46 s per step, above
  the 50 ms period. Single illustrations, not a ranking. GPU9
  `runs/vision_closedloop_20261117`, `runs/visual_center_closedloop_20261117`,
  `runs/dp3_controllers_20261117`.

## 2026-11-21 — UR10e V3 candidate foods and calibration documentation

Current corrected FDCC bundle, actual IKEA CAD blade and virtual ATI frame were
used for one nominal BladeMaster reference press per candidate. Conditions:
10 mm start clearance, 30 mm/s requested feed, original homogeneous material
presets, unchanged V3 force/workspace/arrival/horizon rules. No policy was loaded.
Source-at-start is a70a6d5 with presentation-only edits; the experiment directory
preserves the exact recorder and configs.

Banana / tofu / luncheon reached pose-task success at 39 / 67 / 51 steps,
returns 108.0911 / 108.8698 / 108.5837, final blade gaps 1.472 / 1.432 / 1.467 mm.
No force-limit or workspace failure or nonfinite state was observed. Largest
absolute sensor Fz component during each episode was 1.992 / 0.801 / 0.663 N;
these are simulated components, not real-food calibrated forces.
All terminal component diagnostics report one component; these runs do not
establish complete material separation. Only one nominal condition was run per
food, not a generalization or policy-performance assessment.

Evidence: /dataHDD/dong/Experiments/Cutting_DEV/runs/ur10e_candidate_foods_20261121/
(record.py, individual configs, report.json, geometry, summary.json).
The three presets are candidates, not added to formal two-food training defaults.
Tofu/luncheon are names in our reproduction of paper deformable-block scenarios,
not proof of real-food parameter identification in the BladeMaster paper.

Banana appearance is peeled flesh, consistent with the single-material preset;
the downloaded skin-bearing scan remains separate. The common MPM surface display
uses additional sinc smoothing with unchanged mesh connectivity and at most
0.5 mm extra display-vertex displacement relative to the previous reconstruction
(up to float32 rounding). The banana check measured 0.500032 mm and identical faces.
This is not physical mesh refinement or evidence about submillimetre fracture gaps.
No particles, contact parameters or task rules were changed for appearance.

The real-to-sim plan is rewritten in hardware/REAL2SIM.md, including existing-tool
data collection, identifiability limits, separate support/blade friction and
primary-source calibration references. The user subsequently explicitly confirmed Mini45 SI-580-20 on 2026-11-21.
Official capacities are 580/580/1160 N and 20/20/20 N m; serial-specific
calibration and acquisition interface remain to verify. Task force limits remain unchanged.

Verification: root suite 771 passed, 25 skipped; backend suite 44 passed.
These tests and three reference episodes do not certify absence of all bugs or
real-hardware readiness. No long training or real robot operation was performed.

## 2026-11-21: UR10e five-food candidate and baseline adapters

Candidate contract: configs/shared/ur10e_five_food_v1.json. The original two-food
contract is retained unchanged. These checks do not establish five-food long-run
physical validity or learned success.

- Randomized 15-world checks: both backends passed reward/done agreement and
  partial-reset isolation. Maximum errors: DiSECt 1.4833e-8, BladeMaster 1.5715e-8.
- Nominal reference probe: one world per food, 30 mm/s, 10 mm initial clearance,
  120 control-step inspection budget; the task horizon remained 500.
- DiSECt all five reached pose success: cucumber step 51, half potato 53,
  peeled banana 40, tofu 115, luncheon 52. Nine natural ends/resets were recorded,
  including another physical step after the final reset. Reward error <=3.71e-6.
- BladeMaster cucumber/banana/tofu/luncheon succeeded at steps 50/39/67/51.
  Half potato had not finished at inspection step 120: not a 500-step timeout.
  Eight ends/resets were recorded; reward error <=3.71e-6.
- Corrected the tofu ground diagnostic: the previously reported 6.740 mm
  depth was over ALL FEM nodes, including empty-side auxiliary nodes, not the
  represented material surface. These nodes still participate in FEM/cut-edge
  interpolation and must not be deleted or arbitrarily made contact-active.
  A dedicated GPU9 nominal tofu run reconstructed tri + cut_tri surface vertices
  at every 50 ms control boundary: maximum surface penetration 0.296627 mm,
  versus 11.010555 mm over all nodes. Pose success occurred at step 141;
  terminal data were saved before reset. This is a new single-world run, not
  a bit-identical replay of the five-world step-115 result, and does not bound
  unrecorded material substeps. No physics parameter was changed.
  Evidence: GPU9 /dataHDD/dong/Experiments/Cutting_DEV/runs/tofu_surface_20261121/report.json.
- SAC and frozen WM+MPPI/non-learning MPC interface checks passed on both backends
  with five food groups; see methods/CURRENT_BASELINES.md. These were tiny fresh
  fits and execution tests, not comparative trained results. Planner runtime
  remains above the 50 ms control interval.
- GPU7 final integrated suite: 797 passed / 9 skipped; backend 44 passed.
  Logs: final_root_env.log and final_backend.log. An earlier launch overwrote
  the environment script's PYTHONPATH and failed collection because SciPy's
  extra-module path was removed; it is retained as final_root.log, not counted
  as a pass. The final launch preserved the official environment setup.
  Five canonical sensor-view captures passed separately; no learning semantics
  were changed. Skipped tests are not counted as passes.

Evidence: /dataHDD/dong/Experiments/Cutting_DEV/runs/ur10e_five_food_20261121
on GPU7 (DiSECt) and A6000 (BladeMaster), including check JSON, nominal config,
reference/report.json, transitions.pt, snapshot_initial/final, source status and
probe SHA. Source base b501c28 with the recorded candidate working-tree changes;
not represented as a clean main run. No long training was started.


## 2026-11-21: FDCC reset tare-frame correction before the A6000 run

The BladeMaster half-potato stop above target was traced to a shared FDCC bug:
reset stored a base-frame gravity vector as sensor-frame tare compensation and
omitted the COM torque. With a zero tared sensor reading the UR10e controller
therefore produced a false base-Z force of 6.8768 N. At recorded step120 the
combined controller force/Kzz predicted 2.7523 mm static height error, versus
the observed 2.7584 mm target error (2.7594 mm geometric board clearance).

FDCC.reset now uses its existing _gravity_sensor_wrench(frames), including the
sensor rotation and payload COM moment. Robot/payload mass, inertia, gravity,
material contact, task, stiffness and limits are unchanged. This is a shared
controller implementation correction, not subtraction from ATI observations.
Old checkpoints/records retain their original code provenance.

GPU9 corrected nominal half-potato: success step50, arrival count11, target
height error1.8317 mm, XY error0.0330 mm, no force/workspace/timeout failure.
Final sensor Fz=-8.739 N: physical contact feedback remains present.
Initialization and full reset give zero controller external force for zero
tared readings. Runtime211 s. No broad random-material success claim follows.

Regression: new tests fail twice on the old implementation; all four pass after
the fix. GPU7 base-environment root suite781 passed/25 skipped; backend48 passed.
Evidence: runs/fdcc_tare_20261121 and runs/blademaster_potato_tare_fixed_20261121
under /dataHDD/dong/Experiments/Cutting_DEV on GPU7.
