# Current and historical configurations

Current shared scene: [shared/scene_v1.json](shared/scene_v1.json), task pose_depth_v3.
Its pose_depth_v2 form used by the 2026-11-16 runs is archived under
archive/shared_scene_v1_pose_depth_v2_20261116 (read/evaluate only).
Generate both backend inputs with scripts/prepare_shared_scenes.py; see
[SHARED_SCENE](../docs/SHARED_SCENE.md). Scene capture and reward checks consume
these same inputs. World count in the shared file is a bounded-check resource
setting, not the budget for a long run.

Historical trained-task configs remain under free_cut and blademaster to load
recorded checkpoints. Their task, knife and randomization settings are versioned;
they are not alternatives to the current shared scene.
Superseded intermediate depth/knife/motion checks are under
archive/scene_checks_20261116. They remain fixtures for migration tests.

Visual inputs are under vision/: camera calibration, the historical
prescribed-plane mode and current visual-midpoint selector. They do not
replace the physical task or the generated material-backend configs.
free_scenes/ and food_scenes/ contain source scene-export settings;
blademaster_disect/ contains candidate asset conversions, not approved long-run
tasks. Keep their validation status from DATA/RESULTS when reusing them.
