# Archived action-perturbation evaluation

Historical reproducibility only. This is not a current task or manuscript protocol.
Current task definitions: [BENCHMARK](../BENCHMARK.md).

## Archived action-perturbation protocol — excluded from the paper

**Decision (2026-11-10): stop these evaluations and exclude their results from
the paper.** Retain the exact protocol for historical reproducibility only.
The amplitudes were project choices without measured hardware-error calibration.
The normalized action range is [-1, 1], so sigma 1.0 is large relative to that
range. This changes six compliance/reference commands, not material properties
or an applied physical push. Training exploration is separate; the new
material-distribution training plan does not disable it.

An evaluation condition, not part of the task. `scripts/legacy/evaluate_perturbed.py run
--checkpoint CKPT --cases 256 --seed-base 3000000 --output DIR` executes a frozen
controller on 256 cases fixed before any controller runs. Each case adds
`sigma * z` to the executed six-dimensional action from its first perturbed step
on: the controller's action is clipped to the bounds, the perturbation is added,
and the result is clipped again (rule `executed_action`, since `580597b`; earlier
runs, including the A pair, added it to the raw policy output before clipping and
are not compared with the new rule). sigma in {0.3, 0.6, 1.0}, first perturbed step
in {0, 5, 10}, `z` drawn per step (i.i.d.) or once per episode (persistent
direction), each case with its own seeds (`scripts/legacy/perturbed_eval.py`).
Every case runs to its own end; success, force limit, workspace and time limit are
all kept. Mean material force divides the summed impulse by the actual simulated
time (a force-limit stop ends inside a control step). `--method sac|wm-mppi|supermimic|constant-zero`
uses the `evaluate_benchmark.py` controller interfaces. SAC supports batched
deterministic evaluation (32 worlds in the completed comparison); MPPI and
constant-zero use the single-world path. Evaluation parallelism does not change
the training history of the loaded checkpoint. `evaluate_perturbed.py compare --a DIR_A --b DIR_B --output
FILE` refuses runs with unfinished cases, different case configurations or
different perturbation rules and reports case-paired differences with 95%
bootstrap intervals over cases, plus steps and force on the cases both complete
successfully. These intervals describe two trained policies on this perturbation
distribution, not training-seed variation. Report results as robustness under
action perturbations; proportions are not task success rates.
