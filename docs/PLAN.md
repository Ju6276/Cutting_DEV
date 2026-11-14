# Current plan — 2026-11-14

## Decisions

One cut of freely placed food. Prescribed pose completion earns +100 only without
force/workspace failure; prevent knife penetration into the cutting board.
Retain the existing v3 running rewards. No spring-release/coverage completion
gate, new penetration penalty or sliding penalty. Contract: [TASK_FREE_CUT](TASK_FREE_CUT.md).

The active task is `free_cut_pose_v1`. Existing v3 learned policies are historical
separation-task checkpoints and must not silently resume into this task.

## Completed authorized work

- Default potato: flat-bottom half, free placement, 1 s per-material settle.
  128 distribution draws plus 16 E/density/spring/clearance corners passed the
  unchanged stability windows during a 0.5 s follow-up; no initial contact or damage.
  The whole asset remains as historical preparation evidence.
- Mixed-object reset metadata bug fixed: heterogeneous settle durations are
  recorded per group while force/reward/robot definitions must agree.
  Cucumber/half-potato/prism actually collected 36 transitions and completed
  finite WM plus actor/critic updates. Runnable development config:
  `configs/free_cut/pose_three_object_check.json`, one world per object.
- All 30 nominal reference comparisons completed under the training pose reward:
  press, saw, fast, extreme-fast, slow, stall, retract, slow-press,
  hold-in-material and force-attempt, for all three objects. Reward decomposition
  and WM equations agree on actual transitions. Results include failures/timeouts;
  material separation is recorded separately. [RESULTS](RESULTS.md) has every case.
- Four BladeMaster-derived DiSECt assets generated with prescribed interfaces.
  They remain candidate scenes, not automatically added to long training.
  In particular tofu penetrates the inherited soft ground; longer settling does
  not correct this contact-model limitation. [Mapping and checks](BLADEMASTER_DISECT_ASSETS.md).
- Independent BladeMaster physical backend connected to UR5e/FDCC and the normal
  replay/WM/BPTT training entry. Actual reference, updates, restore, local reset
  and swept-board checks passed. It uses a separate depth-progress task identity;
  this is not a trained policy or a matched-reward comparison. [Review](BLADEMASTER_BACKEND_REVIEW.md).
- Integrated tests: 366 root passed / 5 skipped, 38 backend passed.

## Next step, not automatically launched

Prepare a bounded multi-object training budget using the checked original three
objects. Reassess horizon/throughput for that declared task distribution without
silently changing task/reward definitions. No bulk multi-object replay or
long-trained pose policy has yet been produced. Do not wait for every residual
spring mechanism to be explained before planning; report actual separation
alongside control-task completion.

Do not add the four new candidates to that distribution merely because they
load or reach the goal. Their contact assumptions and preparation limits remain
explicit. BladeMaster batches currently use serial independent material solvers;
no large-batch throughput claim is made.

## Execution and evidence

GPU7 via tams61: `/dataHDD/dong/Projects/Cutting_DEV_board`,
branch `codex/board-cut-v4` (historical branch name). Assets, runs and weights
stay on servers. [DATA](DATA.md) lists locations. Main evidence:
`runs/half_potato_randomized_20261114/`,
`runs/half_potato_multi_update_20261114/`,
`runs/pose_reward_audit_20261114/`.
No new long training was started.
