# Free-object single cut — current direction

Updated 2026-11-14 after the user's simplification request. **Use target-pose completion with a board execution boundary. Do not turn geometric coverage or complete spring separation into additional training gates.**

## Three distinct things

1. **Control-task completion:** reach the prescribed knife/robot target within the configured position/orientation tolerances and arrival count, without force/workspace failure. Reuse the existing SliceIt pose-error/termination interface. Its configured `target_dims` select the checked axes; do not describe this as checking every Cartesian component unless configured that way. Its arrival counter is the existing accumulated counter, not a newly claimed consecutive hold.
2. **Execution boundary:** the actual finite knife must not pass below the board. Stop unsafe downward/rotational motion. The board stop is a software interlock; it does not synthesize board reaction forces. A normal board stop earns no new penetration penalty.
3. **Physical evaluation:** separately report material separation, residual springs, object motion and whether the knife missed the intended material. These measurements do not supply new policy inputs or extra WM progress/completion heads.

Reaching a target is a valid RL task definition. It does not alone prove material separation. A controller can reach the target after an object moves away; report that physical outcome rather than quietly renaming it a completed cut.

## Numerical handling

The existing board interlock aims for +1 micrometre clearance. A knife clearance below -2 micrometres triggers a simulator error; it must not generate a successful or penalized learning transition. Small errors within that numerical allowance are not a new RL cost. These values are local numerical settings, not published Mini45 tolerances or parameters copied from the papers. They do not authorize millimetres of penetration. Material-ground overlap is measured separately from knife clearance.

## What is implemented, what is not

- Source SliceIt target-pose reward/termination exists and remains available.
- The board interlock is implemented and physically exercised on three free-object scenes.
- `free_cut_v4` is the **retired coverage-task experiment**, preserved to reproduce the 2026-11-13 audit. It is not the selected long-training task, and WM construction rejects it.
- The simplified pose task still needs an explicit board-aligned goal and versioned integration into the training entry. Do not launch v4 or silently relabel a v3 separation checkpoint as this new task.
- This cleanup does not change reward weights, the WM network, BPTT, the critic or the action mapping. A learned coverage head, new penetration reward, and a requirement to break every spring are not part of the selected direction.

The existing default source path has no board interlock. Its old reproduction command is therefore not the new safe task launch. A board-aligned target must be checked against the actual material knife, including the robot/material coupling offset; the old lower target must not be copied without checking.

## Historical implementations

- [v1-v3 equations and checkpoint contract](archive/TASK_FREE_CUT_V1_V3.md): separation-based rewards, progress state and learned completion head. These remain for reproducing existing policies and measurements.
- [v4 board/coverage audit](BOARD_CUT_REVIEW.md): shows why knife-path completion and physical separation must be reported separately. The 64-case outcomes remain historical data, not the current task specification.
- [PLAN](PLAN.md): actual next steps and object readiness. [RESULTS](RESULTS.md): dated measurements.

## Relationship to the papers

[SliceIt, section III-C.4](https://arxiv.org/html/2404.02569v1#S3.SS3.SSS4) describes target-pose completion and collision penalties based on excessive contact force. It does not specify our micrometre geometric-error penalty; none is being added.

[DiSECt's official optimization example](https://github.com/NVlabs/DiSECt/blob/main/examples/optimize_slicing.py) constrains final blade height to zero and lateral motion by blade length. It is constrained trajectory optimization, not an RL reward for numerical penetration. Our hard execution stop is an engineering safeguard, not a claim to reproduce its optimizer or SliceIt's compliant board-contact dynamics.
