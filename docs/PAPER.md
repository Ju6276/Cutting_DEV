# Paper

Updated 2026-11-12. **The topic is being re-decided; nothing in this file is a
settled claim yet.** [PLAN](PLAN.md) is the task list, [RESULTS](RESULTS.md) the
measurements, [literature search 2026-11-12](literature/force_adaptive_search_20261112.md)
and [NOVELTY](NOVELTY.md) the prior-art evidence. The previous version of this file
(problem: "the world model under-predicts the force spike at contact onset") is
archived as [PAPER_20261111](archive/PAPER_20261111.md) and is **retired**.

## 1. Why the previous problem statement was retired

It was derived from failures observed in our own experiments, not from the task.
The 2026-11-12 diagnosis (RESULTS) shows those failures are at least partly produced
by the task definition: with the inherited reward the trained policy drives the
knife down at 20-27 cm/s against a 1-5 cm/s speed command, and force-limit hits
happen at that speed. Whether a force-spike problem remains once the task is
corrected is unknown. A paper cannot rest on it.

## 2. Task-first research scope (open)

A candidate application is cutting unknown deformable objects with force feedback.
First specify the real operation, holding arrangement, desired outcome and
measurable limits; then identify what practical limitation existing methods leave
unresolved. No particular modality or learned architecture is required by this.

Force-adaptive speed/compliance control is a candidate, not a settled topic.
The historical/default material-feedback scalar is not a Mini45 contact wrench.
Optional net feedback is now implemented and locally checked (METHOD), but has
not been calibrated against a real sensor and mount. Force-limit
termination detects a modeled violation; it does not guarantee a real robot
stays below a bound.

WM+BPTT, a risk head and conservative planning are available ingredients. Their
implementation or failure on the old task does not establish a real need for
them. Predicted force peaks are not an adopted core contribution.

## 3. Preliminary prior art (requires primary-source verification)

This table summarizes the linked preliminary search, including second-hand
summaries, abstracts and snippets. Negative claims in "What it does not do"
are provisional, not established exclusions. The candidate-oriented search
cannot substitute for reading the practical problems the papers solve.

The task framing alone is **not** new:

| Precedent | What is already done | What it does not do |
|---|---|---|
| Hathaway et al. (T-ASE 2023, IROS 2024, arXiv 2026) | One model-free policy adapts feed rate and stiffness from wrench on materials hidden from the agent; sim-to-real (milling saw) | No hard force limit (quadratic penalty only), no history, no learned model for policy gradients, not knife cutting of deformables |
| FORGE (2024) | Recurrent policy conditioned on a maximum force, hidden dynamics inferred from force; sim-to-real assembly | Soft penalty: real peak 12.8 N against a 7.5 N threshold; no material removal, no speed command |
| SRL-VIC (RA-L 2024) | Force threshold ends the episode; safety critic + variable stiffness; sim-to-real | Environment stiffness not varied; model-free |
| SliceIt (primary text checked 2026-11-14) | SAC trained with randomized loading of calibrated tomato, cucumber and potato; carrot evaluated as an unseen food | Do not claim that one policy across materials or adaptation to unseen food is absent from this precedent; distinguish the model/planning method and actual evidence |
| Luo & Yuan robocutting; Liu et al. (RA-L 2025) | Other real food-cutting RL / demonstration precedents | Their exact training and transfer protocols require separate primary-source comparison; no blanket exclusion is established here |
| Peng et al. (TNNLS 2022); CaT; LAMBDA; SafeDreamer | Chance-constrained / terminating / Lagrangian safe (model-based) RL | No contact force, no manipulation |
| Classical adaptive feed control | "Slow the feed when force nears a limit" as a hand-designed governor | Needs tuning per process; no learned adaptation |

Not found in the search (38 queries; "not found", not "does not exist"): knife
cutting of deformable objects with a hard, episode-ending force limit handled as
a formal constraint across unknown materials; a wrench-predicting world model used
for backpropagated policy learning under a peak-force constraint; prediction of the
within-interval force peak (policy at 20 Hz, sensor faster).

A force-feedback controller, model-free methods and learned-model planning
are comparison candidates. The final set follows the task and claim; this is
not a requirement to implement all of them now. Candidate comparison from the
search (not decided): a recurrent force-penalty policy (FORGE style), terminating
safe RL (SRL-VIC/CaT style) and a force-feedback speed governor, compared on
force-limit violations at matched cutting speed.

## 4. Evidence needed before a method claim

1. A real cutting requirement and credible completion/force definitions.
2. A platform demonstration representing the required interactions, including
   the fixture and sensor meaning, without success shortcuts.
3. A literature-grounded limitation that matters in that task.
4. A method addressing it, and comparisons capable of testing the claim.
5. Real-robot evidence if the claim concerns physical cutting or force safety.

Additional modalities and objects do not automatically provide novelty.
A simulator-only claim must be scoped to the represented physics and outcomes.

## 5. What exists

- Base method, multi-object environment, per-object logging, deterministic
  evaluation, resume with replay: implemented and tested (METHOD, tests).
- Risk head + adaptive penalty (`--risk-constraint`) and the official SuperMimic
  adaptation: implemented; both failed in their first run for identified reasons
  (RESULTS 2026-11-12). Kept as code; not results.
- Fixed-prism numbers (archived PAPER §5) retain their historical goal-pose
  definitions; they do not alone validate separation, physical wrist-force
  prediction or general cutting performance.

2026-11-14 correction: the earlier grouped row incorrectly described SliceIt as SAC per material. Section IV-A.3 explicitly describes loading different calibrated foods during training: [primary text](https://arxiv.org/html/2404.02569v1#S4.SS1.SSS3). The selected engineering direction is target-pose completion with a board boundary; physical separation is separately reported. This does not establish a new paper contribution.
