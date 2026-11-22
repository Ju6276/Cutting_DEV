# Historical free-cut configurations

These files preserve recorded task versions and checkpoint compatibility.
**None of the pose_*.json files in this directory selects the current
pose_depth_v3**: they still contain free_cut_pose_v1.
Generate current paired backend inputs from
[shared/ur10e_five_food_v1.json](../shared/ur10e_five_food_v1.json), following
[场景](../../docs/METHOD.md#scene). Use the generated files recorded
in the run manifest for training/evaluation.

| Files | Historical purpose |
|---|---|
| `pose_cucumber.json`, `pose_prism.json`, `pose_potato.json` | Pose-v1 development and material/reset checks |
| `pose_three_object_*.json` | Pose-v1 integration, capacity and recorded training; world counts are not current launch instructions |
| `interface_check.json` | v1 rejected reward candidate |
| `review_v2.json`, `review_v3.json` | Bounded earlier reward/interface checks |
| `short_train_cucumber.json`, `eval_cucumber.json` | v3 short training and frozen development conditions |
| `eval_board_cucumber_v4.json` | Retired board/coverage evaluation |

The historical potato pose-v1 entry uses the unheld flat-bottom half and 1 s
per-material settling. Its randomized reset checks remain evidence for that
version, not validation of arbitrary new inputs. Tests and old checkpoint
reproduction still reference these files, so they are retained intact.
Never overwrite an old config to relabel an old checkpoint as a new task.
Simulation limits/material parameters are not real-robot calibration.
