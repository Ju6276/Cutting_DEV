# Board-stop cutting task and frozen-policy audit (2026-11-13)

User-authorized scope: implement effective knife-path coverage plus board arrival,
with no force/workspace failure; enforce no downward passage through the board.
Evaluate the latest 543-update policy on the same 64 development conditions.
No new training, fracture changes, material tuning, or checkpoint deletion.

## Task version

Opt-in `free_cut_v4` preserves v1-v3. The board is DiSECt y=0 (the actual
particle-ground convention), not the moving object's minimum particle height.
The material knife's actual finite vertex geometry and xyzw rotation determine
its lowest point. The leading edge is the pair of extreme-z lowest-local-y
vertices; the implementation rejects a degenerate edge.

Coverage witnesses are every cut-interface spring midpoint, reconstructed from
current material particles with the consistent edge interpolation coordinates.
They are geometry samples, not stiffness or fracture labels. A witness is covered
only when the finite leading edge sweeps down past it in relative motion, while
it lies within the blade's z span at both sample boundaries and the knife plane
intersects the actual particle transverse footprint. There is no fixed permitted
sliding distance. This is projected-section coverage, not a continuum geometric
intersection proof. The 0.5 mm tolerance applies to swept height, not to permitted
object sliding. Coverage is accumulated at 500 Hz. No witness is discarded
because it is near the board. 0.5 mm is an explicit geometric tolerance (including
material penalty-contact penetration), not a fitted material constant or a
claim of continuum intersection accuracy. Sliding outside the blade cannot
receive coverage credit. Large motion between samples remains a discretization
limitation; tests include missed-width and translated-object cases.

Success = all geometric witnesses covered AND board stop reached AND no
force/workspace failure or board penetration. Stop at the board without full
coverage is a separate terminal `board_incomplete`, with no success bonus.
Material separation remains a separate all-springs-zero measurement. A successful
knife path is explicitly NOT labelled full physical separation.

## Execution stop

At each 2 ms coupling step, bound the continuous translated/rotated knife sweep
against the board, including a rotation-curvature bound. Shorten an unsafe
exchange motion towards a 1 micrometre positive clearance by bisection, advance the
material with this shortened twist. Only actual clearance <=10 micrometres
counts as board arrival; a conservative rotational bound that stops farther away
can continue approaching next interval. Stop that world at arrival. The robot joint motion
is shortened by the same fraction and its velocity stopped. Completed worlds
are frozen while other worlds continue. Per-sample and minimum gap are recorded.
This is a software motion interlock, NOT a compliant knife-board contact solver;
it generates no invented board reaction wrench. Hence these runs cannot assess
board contact-force control or certify a hardware stop.

## Reward / learned model compatibility

v4 uses geometric path coverage increments in place of spring-release increments.
The existing v3 weights, gamma=1, finite deadline and speed/force/jerk costs are
unchanged. Original release fraction is saved as separation_progress. Thus v3/v4
returns have different meanings and are not directly ranked.

The frozen policy retains identical observation/action dimensions and goal. Its
weights are untouched. The v3 WM completion head and critic were trained against
different labels and are NOT silently reused: constructing a v4 WM is explicitly
rejected. This delivery is the physical task and frozen-policy evaluation. A
matching trainable v4 WM state/termination model requires a separate change and
new data; no claim of training readiness is made.

## Bounded evaluation protocol

GPU7, maximum 60 GPU minutes after compilation for two 64-world first-episode
runs plus one small integration check. Use eval_cucumber.json seed 100, latest
free_cut_joint_20261113/training/last.pt. Preserve material, start, target speed,
force limits, horizon and policy weights. Check conditions pairwise.

1. Original v3 behaviour, plus read-only knife/coverage observer: measure old
   successes, timeouts and whether/when the knife crosses the board.
2. v4 execution stop + task: same policy and cases; measure path success, actual
   separation, incomplete board arrival, force/workspace/time endings.
3. Pair by case. Distinguish changed labels from changed trajectories. Do not
   claim success-rate improvement is learning improvement. Old successful
   policies may continue into unfamiliar states when v4 requires board arrival.
4. Preserve raw transitions, actual minimum knife height, conditions, source and
   checkpoint hashes. Use finally cleanup. No automatic budget extension.

Tests: geometry stop with translation/rotation and high velocity, finite-edge
coverage/sliding, separation vs success, failure precedence, and old checkpoint
incompatibility; root + backend regression suites. Actual evaluation supplies
the integrated knife-height evidence, beyond pytest.

## Reproduce the frozen-policy transfer

Run from the repository root with prepared backend assets and the shared runs directory mounted. The input conditions retain the checkpoint's v3 task; explicit board mode derives and saves the v4 configuration in its manifest. Do not pass the v4 template as if the checkpoint had been trained on it.

```bash
source runs/sac_env.sh
PYTHONPATH=.:backends/mujoco .venv/bin/python scripts/evaluate_board_cut.py \
  --checkpoint runs/free_cut_joint_20261113/training/last.pt \
  --conditions configs/free_cut/eval_cucumber.json \
  --mode board --max-seconds 1800 --output runs/NEW_BOARD_AUDIT
```

Use a new output directory. Repeat with mode legacy for the read-only observer comparison. No checkpoint is overwritten. The v4 JSON template describes the derived physical task; training against it is intentionally rejected until the learned task model has matching semantics.

## Evaluation outcome

Completed both 64-case physical runs on GPU7. Legacy: 53 successes, 52 episodes below board (41 successful episodes), minimum -9.119 mm. v4: 38 path successes, 26 incomplete board arrivals, zero knife penetrations, minimum +0.825 micrometres; only 19/64 fully separated. None of the 11 paired legacy timeouts became a path success. See RESULTS dated 2026-11-13 for interpretation, historical replay variation and material-below-board witnesses. These are different task outcomes, not evidence of improved learning.

## Validation completed

- GPU7 root suite, with CUDA backend assets and the MPPI environment enabled:
  **337 passed, 5 skipped** (224.31 s).
- Bundled backend suite: **38 passed** (5.16 s).
- Actual physical audit: 64 legacy and 64 board-stop episodes, not pytest-only
  evidence. All new-task knife minima positive; finished-world physical states
  remained unchanged while other worlds advanced.
- No new training, no checkpoint mutation, no generated run data committed.
- Final v4 runtime source hashes exactly match the committed implementation.
  The legacy replay manifest records the earlier stop-only implementation;
  the stop refinement is inactive on that legacy path. Exact manifests are
  retained, not relabelled as if both runs loaded identical files.
