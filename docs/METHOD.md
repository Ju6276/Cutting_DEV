# Source mapping

Source revision: SuperMimic-Reproduced `ba610534`; SliceIt `4efdf37`;
robot utilities `a14c856`; DiSECt `db7de0f`.

| Module | Source design | Cutting implementation |
|---|---|---|
| Replay | Episode-aware ring, 1000 rows/environment, clean/executed actions, terminal row | Original replay code with a recorded-episode importer; existing task configuration is fixed for this scene |
| Schedule | Warmup 40 environment steps; UTD 0.4; one actor update per world-model update | Original defaults and update cadence; current physical sampler has one environment, source default has 4096/GPU |
| Objective | Survival-weighted 32-step return, terminal critic, GAE lambda 0.95 | Original objective and GAE functions; only their type-only import is changed |
| Dynamics MLP | Three 2048-unit SiLU layers, zero final layer | Original source module; cutting StateSpec and trainer integration are next |
| Policy | Goal encoder, 32-D latent, decoder; widths 2048/1024/512/256; decoder gain 0.01 | Same architecture with a 7-D target pose, ten frames of SliceIt's 24-D observation, and six task actions |
| Policy input scaling | G1 observations and reference motions use robot-specific scales | SliceIt observations retain their existing normalization; target pose uses its SI/quaternion representation |
| Actions | Gaussian execution noise sigma 0.1 | Same noise; execution respects SliceIt's existing action-space bounds; both clean and executed actions are recorded. SliceIt's last-action observation contains the executed action |
| Observation/reward | Analytic task functions, including robot FK | Torch translation of original SliceIt normalization, reward and termination formulas. No learned reward network |
| Termination | Learned probability head trained from simulator terminal labels | Task labels and hard formulas are connected; the cutting probability head is next |
| Physical state | Simulator state read at the transition boundary | Gazebo joint/controller state plus DiSECt position, velocity and spring history, recorded during observation generation, before terminal controller shutdown |
| Force history | Observation reads the sensor history | Record that exact sensor read, rather than rereading a moving queue after env.step |
| Geometry/material | Source humanoid task | SliceIt's public prism configuration and aligned initial knife pose; physical equations and task thresholds unchanged |
| Reference library | Motion-tracking goals, reward targets and reset frames | SliceIt's fixed cutting target and original task reward replace the humanoid motion library |

The preceding offline full-packet predictor, learned reward head and added damage
projection have been removed. Their implementations and experiment revisions
remain in Git history. New architecture changes require a source mapping before
implementation; a short fit is a data/interface check, not a model-selection result.

Sources: [world model](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/src/supermimic/world_model/world_model.py),
[training loop](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/src/supermimic/training/loop.py),
[SliceIt](https://github.com/omron-sinicx/sliceit).
