# Current plan — 2026-11-14

## Decisions

One cut of freely placed food. Prescribed pose completion earns +100 only without
force/workspace failure; prevent knife penetration into the cutting board.
Retain existing v3 running reward terms for this authorized change. No spring
release/coverage completion gates, no new penetration penalty or sliding penalty.
Task contract: [TASK_FREE_CUT](TASK_FREE_CUT.md).

## Current work

- Implemented `free_cut_pose_v1`: board-aligned goal, physical board stop,
  pose/counter settlement, WM task semantics and versioned configuration.
  Physical cucumber/prism arrival and actual WM/actor/critic updates plus optimizer/replay restart passed; see RESULTS. No learned-policy performance claim.
- Prepared a natural potato support pose from the uncut tetrahedral geometry,
  preserving vertices/tetrahedra/material parameters. Re-exported the cutting
  interface. Three material pairs were stable after 8 s in the production reset
  check (COM movement <0.01 mm over the following 0.25 s).
- Cucumber/prism development interactions already exist. A stationary potato
  dataset requires its own checked settling duration; the old 0.75 s remains
  inadequate. No friction/damping tuning or artificial velocity reset.

## Completed preparation comparison

The authorized whole/half potato comparison is complete; see RESULTS 2026-11-14.
Both assets are unheld, with the same material/contact parameters. Across three
material pairs, the flat-bottom half met the common settling-window check at
0.55 s and remained stable through 8 s; whole required 4.65/5.35 s or did not
remain within the window by 8 s. Nominal references: half pose-success in 39 steps,
whole timeout at 180. Neither fully separated; no force or board violations.

Recommendation: use the flat-bottom half as the next preparation candidate, with
1 s settling as a candidate supported by these traces. The reference cuts still
used 8 s, and full randomized-density training has not been tested at 1 s. Active
pose_potato.json remains whole/8 s; no training was started. Recorded separately:
JSON ground_radius maps to contact_closeness but the kernel reads contact_distance;
this comparison preserved the actual 0.01 mm contact distance, not the JSON's 1 mm.

## Next steps

1. Use the versioned pose configs and checked settling durations for development collection. No bulk multi-object replay has yet been collected.
2. Write a bounded
   multi-object training plan. Re-evaluate time horizon against the selected
   objects; do not silently copy a longer horizon or retune reward weights.
3. Do not wait for every remaining spring mechanism to be explained. Report
   material separation separately from control-task completion.

No new long training is started in this work. Existing policies were trained on
v3 separation labels and are not checkpoints of the new pose task.
Execution: GPU7 via tams61, `/dataHDD/dong/Projects/Cutting_DEV_board`,
branch `codex/board-cut-v4` (historical branch name). Assets, runs and weights stay
on servers. Review evidence: `runs/pose_task_20261114/`.
