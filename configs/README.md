# Current and historical configurations

Current shared scene: [shared/scene_v1.json](shared/scene_v1.json).
Generate both backend inputs with scripts/prepare_shared_scenes.py; see
[SHARED_SCENE](../docs/SHARED_SCENE.md). Scene capture and reward checks consume
these same inputs. World count in the shared file is a bounded-check resource
setting, not the budget for a long run.

Historical trained-task configs remain under free_cut and blademaster to load
recorded checkpoints. Their task, knife and randomization settings are versioned;
they are not alternatives to the current shared scene.
Superseded intermediate depth/knife/motion checks are under
archive/scene_checks_20261116. They remain fixtures for migration tests.
