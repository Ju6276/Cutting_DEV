# SuperMimic-Reproduced components

Source: https://github.com/CuboiLeo/SuperMimic-Reproduced
Commit: `ba6105349069bf040f1fdad60873d13138d505a2`.
License: [MIT](LICENSE).

| Local file | Source path | Changes |
|---|---|---|
| replay_buffer.py | src/supermimic/data/replay_buffer.py | None |
| dynamics.py | src/supermimic/world_model/dynamics.py | None |
| schedule.py | src/supermimic/training/schedule.py | None |
| objective.py | src/supermimic/imagination/objective.py | Move the rollout type import under TYPE_CHECKING; equations unchanged |
| schedule_config.py | ScheduleCfg in src/supermimic/config.py | Extract the dataclass; defaults unchanged |

| trainer_config.py | WMTrainerCfg and PolicyTrainerCfg in world_model/wm_trainer.py and imagination/policy_trainer.py | Extract dataclasses; defaults unchanged |

Task adapters in `cutting_world_model/world_model.py` and `trainer.py` follow
`WorldModel.window_loss`, `WorldModelTrainer.update`, `imagine` and
`PolicyTrainer.update` at the same revision. Effective batches are accumulated
in microbatches; rollout steps use the source compile setting. Task-specific
state and analytic heads replace the G1 interfaces. The source MIT license
applies to these adaptations as well.
