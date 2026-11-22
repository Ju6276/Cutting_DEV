# Configurations

Current training: [shared/ur10e_five_food_v1.json](shared/ur10e_five_food_v1.json),
UR10e + ATI + IKEA CAD knife, task pose_depth_v3/project_continue_v2.
Retained profiles: UR5e [shared/scene_v1.json](shared/scene_v1.json) and the
earlier UR10e [two-food contract](shared/ur10e_ati_ikea_v1.json).
These platforms/distributions have distinct checkpoint metadata.
All four controller entries use this same current task; backend choice is
independent of controller choice. Old checkpoint loading must retain the task
metadata embedded in that checkpoint. See the [controller ledger](../docs/methods/CURRENT_BASELINES.md).

[evaluate_only/scene_v1_pose_depth_v2_20261116.json](evaluate_only/scene_v1_pose_depth_v2_20261116.json)
is the exact scene_v1 used by the 2026-11-16 shared runs (pose_depth_v2, three objects,
180 steps). It exists only to read and evaluate those checkpoints; training on it is refused.
Generate both backend inputs with scripts/prepare_shared_scenes.py; see
[SHARED_SCENE](../docs/SHARED_SCENE.md). Scene capture and reward checks consume
these same inputs. World count in the shared file is a bounded-check resource
setting, not the budget for a long run.

Historical trained-task configs remain under free_cut and blademaster to load
recorded checkpoints. Their task, knife and randomization settings are versioned;
they are not alternatives to the current shared scene.

Visual inputs are under vision/: camera calibration, the historical
prescribed-plane mode and current visual-midpoint selector. They do not
replace the physical task or the generated material-backend configs.
free_scenes/ and food_scenes/ contain source scene-export settings;
blademaster_disect/ contains candidate asset conversions, not approved long-run
tasks. Keep their validation status from DATA/RESULTS when reusing them.
