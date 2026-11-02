# Migration order

Target: the existing SliceIt robot/compliance task, Gazebo + DiSECt, compared
with its unchanged SAC baseline.

1. **Task/data interfaces:** reuse original replay, schedule, objective and MLP;
   translate SliceIt observation/reward/termination functions; verify on a whole
   recorded episode. Use the original initialized policy architecture and noise
   for new interface collection. Record actual sensor reads at observation time.
2. **Cutting StateSpec:** separate physical state, task context and observation;
   replace G1 FK/integration with the robot/object interfaces required by this
   task. Derive observation and reward from predicted state using original task
   formulas. Keep source loss reductions and explicit scaling provenance.
3. **Online loop:** collect with the current policy, retain recent replay,
   warm up, fit eight-step windows, then apply the original 32-step actor/critic
   update and return to physical collection. Keep clean/executed actions and
   termination/truncation separate. Use source hyperparameters; document necessary
   task/hardware interface differences before running.
4. **Learning curves:** report physical transitions, wall time, prediction errors,
   original task return and physical replay outcomes. Judge model performance
   over training, instead of modifying its structure after a tiny initial fit.

Item 1 passed the recorded-episode observation/reward/termination and replay
alignment checks (41 transitions; RESULTS). Next is item 2. Interface collection
has a 600-second wall cap, per-step progress and automatic child cleanup.
It performs no policy updates.
The original SAC continues independently. Physics collection and neural fitting
are scheduled sequentially within this branch to limit GPU contention.

The source's 40-step warmup yields about 163840 transitions at 4096 environments.
One Gazebo/DiSECt environment yields 40. The initial three-episode collection
recorded 104 transitions in 495 s while SAC ran concurrently. Throughput and data
requirements must be measured before allocating a long online-training run.
