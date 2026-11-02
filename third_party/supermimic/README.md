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
