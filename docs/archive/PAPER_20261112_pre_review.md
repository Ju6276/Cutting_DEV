> Archived copy of docs/PAPER.md at a8b5083 (2026-11-12, before the task/platform review rewrite in 0ca52c3).
> Proposals in it are not approved decisions unless the current PLAN says so.

# Paper

Updated 2026-11-12. **The topic is being re-decided; nothing in this file is a
settled claim yet.** [PLAN](../PLAN.md) is the task list, [RESULTS](../RESULTS.md) the
measurements, [literature search 2026-11-12](../literature/force_adaptive_search_20261112.md)
and [NOVELTY](../NOVELTY.md) the prior-art evidence. The previous version of this file
(problem: "the world model under-predicts the force spike at contact onset") is
archived as [PAPER_20261111](../archive/PAPER_20261111.md) and is **retired**.

## 1. Why the previous problem statement was retired

It was derived from failures observed in our own experiments, not from the task.
The 2026-11-12 diagnosis (RESULTS) shows those failures are at least partly produced
by the task definition: with the inherited reward the trained policy drives the
knife down at 20-27 cm/s against a 1-5 cm/s speed command, and force-limit hits
happen at that speed. Whether a force-spike problem remains once the task is
corrected is unknown. A paper cannot rest on it.

## 2. Candidate topic (proposed 2026-11-12, not decided)

**Force-adaptive robotic cutting of unknown objects under one hard force limit.**
One policy and one force/torque limit for objects of unknown shape and material.
The policy receives no object identity or material label. It tracks a commanded
cutting speed when the material allows it and slows down or changes compliance
from its own force feedback when the force approaches the limit. Exceeding the
limit is a failure, not a cost.

The problem follows from the task (unknown material, hard hardware limit, speed
command), not from an observed failure of a particular model.

Candidate method: the working base (history-conditioned world model of robot state
and sensor wrench including the within-interval peak; policy trained by
backpropagation through short imagined rollouts with behaviour regularisation)
with a constraint on the predicted peak force. The force prediction makes the
speed-force trade-off differentiable.

## 3. Prior art (2026-11-12 search; evidence levels in the search record)

The task framing alone is **not** new:

| Precedent | What is already done | What it does not do |
|---|---|---|
| Hathaway et al. (T-ASE 2023, IROS 2024, arXiv 2026) | One model-free policy adapts feed rate and stiffness from wrench on materials hidden from the agent; sim-to-real (milling saw) | No hard force limit (quadratic penalty only), no history, no learned model for policy gradients, not knife cutting of deformables |
| FORGE (2024) | Recurrent policy conditioned on a maximum force, hidden dynamics inferred from force; sim-to-real assembly | Soft penalty: real peak 12.8 N against a 7.5 N threshold; no material removal, no speed command |
| SRL-VIC (RA-L 2024) | Force threshold ends the episode; safety critic + variable stiffness; sim-to-real | Environment stiffness not varied; model-free |
| Luo & Yuan robocutting; SliceIt; Liu et al. (RA-L 2025) | Real knife cutting of varied foods with force feedback (RL per food / SAC per material / demonstration primitives) | No single frozen learned policy with a formal force limit across unseen objects |
| Peng et al. (TNNLS 2022); CaT; LAMBDA; SafeDreamer | Chance-constrained / terminating / Lagrangian safe (model-based) RL | No contact force, no manipulation |
| Classical adaptive feed control | "Slow the feed when force nears a limit" as a hand-designed governor | Needs tuning per process; no learned adaptation |

Not found in the search (38 queries; "not found", not "does not exist"): knife
cutting of deformable objects with a hard, episode-ending force limit handled as
a formal constraint across unknown materials; a wrench-predicting world model used
for backpropagated policy learning under a peak-force constraint; prediction of the
within-interval force peak (policy at 20 Hz, sensor faster).

A reviewer will ask: why not a model-free force-penalty policy (FORGE), a
terminating safe-RL policy (SRL-VIC/CaT), or a hand-tuned force-feedback speed
governor? The paper is defensible only as a **method** paper that beats these three
in the same task, compared on force-limit violations at matched cutting speed.

## 4. What has to be true for this paper (to be tested, in this order)

1. The task is well defined for every object (PLAN step 1): success means the object
   is cut, the reward makes speed tracking under the limit optimal.
2. The base learns the corrected task on several objects.
3. The constraint on predicted peak force lowers force-limit violations at matched
   speed against the three baselines above, on held-out material/geometry/start.
4. Real-robot validation with the external Mini45 ([REAL2SIM](../REAL2SIM.md), draft).

## 5. What exists

- Base method, multi-object environment, per-object logging, deterministic
  evaluation, resume with replay: implemented and tested (METHOD, tests).
- Risk head + adaptive penalty (`--risk-constraint`) and the official SuperMimic
  adaptation: implemented; both failed in their first run for identified reasons
  (RESULTS 2026-11-12). Kept as code; not results.
- Fixed-prism development results (archived PAPER §5) remain valid for the fixed
  prism only.
