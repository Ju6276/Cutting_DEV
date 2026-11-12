# Literature search — force-adaptive cutting under a hard force limit (2026-11-12)

Purpose: prior art for the candidate topic (defined in
[archive/PAPER_20261112_pre_review](../archive/PAPER_20261112_pre_review.md) §2; PAPER now keeps it open), searched
outside the existing library. 38 web-search queries by a search agent (arXiv,
publisher and project pages; IEEE Xplore and Google Scholar not searched directly).
**Evidence levels:** *summary* = the agent queried the paper's full HTML through a
summarising reader (second-hand, not a first-hand read); *abstract*; *snippet* =
search-result snippet only. Nothing here has been read first-hand by the main
session; verify before citing. "Not found" means not found in these queries.

## Most relevant works

| Work | Venue / year | Link | What it does | Difference from the candidate | Evidence |
|---|---|---|---|---|---|
| Hathaway, Rastegarpanah, Stolkin: milling strategies with passive variable interaction control | T-ASE 2023 | https://arxiv.org/abs/2304.01000 | PPO outputs inverse stiffness, feed time-offset and depth offset; cutting constants randomised and hidden | Model-free; force as quadratic penalty, no hard limit; no history; milling saw; policy results in simulation | summary |
| Hathaway et al.: imitation learning, residual GP force model | IROS 2024 | https://arxiv.org/abs/2311.04096 | Mechanistic + GP force model corrects the simulator; policy transferred to a real robot | Force model is not used for policy gradients; no force constraint | abstract |
| Hathaway et al.: neural-stylisation sim-to-real for cutting | arXiv 2026 | https://arxiv.org/abs/2601.20846 | Style transfer from unlabelled real trajectories; real cutting of five materials with stiffness/feed actions | No force limit; contribution is transfer | summary |
| FORGE | arXiv 2024 | https://arxiv.org/abs/2408.04587 | Recurrent PPO conditioned on a maximum force; penalty above threshold; hidden controller gains | Soft penalty (real max 12.84 N vs 7.5 N threshold); model-free; assembly | summary |
| SRL-VIC | RA-L 2024 | https://arxiv.org/abs/2406.13744 | CMDP with safety critic and recovery policy; force threshold ends the episode; variable stiffness; sim-to-real | Stiffness/material not varied; model-free; no speed objective | summary |
| Constraint-grounded RL for VIC insertion | arXiv 2026 | https://arxiv.org/abs/2609.13516 | Policy takes the force limit, outputs insertion rate and gain | Simulation only; limit satisfaction empirical | abstract |
| CoRMA | arXiv 2026 | https://arxiv.org/abs/2605.22082 | RMA-style context from force/proprioception history | No material inference, no limit, no cutting | abstract |
| Xue et al.: implicit motor adaptation | IJRR 2025 | https://arxiv.org/abs/2412.11829 | Parameter distribution from history, parameter-conditioned policy | Wrench input unverified; no limit | snippet |
| Ding et al.: transferable F/T dynamics model | IROS 2019 | https://arxiv.org/abs/1912.00260 | Learned wrench dynamics across pegs; MPC and MBRL | No constraint; not cutting | abstract |
| Yi et al.: action-conditional GNN contact dynamics | arXiv 2025 | https://arxiv.org/abs/2509.12151 | Predicts motion and reaction wrench; MPC in simulation | No constraint; no policy gradient | abstract |
| Anand et al.: deep model-predictive VIC | RAS 2023 | https://arxiv.org/abs/2209.09614 | Learned impedance model + CEM-MPC over stiffness | Force is an input, not predicted | summary |
| Hachimine et al.: learning to shape by grinding | RA-L 2023 | https://arxiv.org/abs/2308.02150 | Model-based RL for grinding | Retrain when hardness changes (snippet) | abstract |
| DecompGrind | arXiv 2026 | https://arxiv.org/abs/2603.22859 | Imitation-learned contact-force policy across hardness | No limit formulation | abstract |
| Liu, Verleysen, wyffels: one-shot demonstration slicing | RA-L 2025 | DOI 10.1109/LRA.2025.3606310 | Position/force primitives from one demo, 14 unknown foods, real robot | Not RL, no model, no explicit limit or speed command | snippet |
| Zhang, Iida, Forni: rock chop via virtual model control | arXiv 2025 | https://arxiv.org/abs/2508.02604 | Non-learning controller cuts five vegetables without object information | A "why learn?" baseline | abstract |
| Padalkar et al.: task frame + RL vegetable cutting | ICINCO 2020 | https://www.scitepress.org/Papers/2020/95906 | RL learns the cutting-force part of a compliant skill | Per-vegetable learning | snippet |
| Peng et al.: model-based chance-constrained RL, separated PI Lagrangian | TNNLS 2022 | https://arxiv.org/abs/2108.11623 | Chance constraint, model gradient of safe probability, PI multiplier | No force, no manipulation | snippet |
| CaT: constraints as terminations | arXiv 2024 | https://arxiv.org/abs/2403.18765 | Violations become stochastic terminations | Model-free, locomotion | snippet |
| State-wise safe RL survey | IJCAI 2023 | https://arxiv.org/abs/2302.03122 | Per-step hard-constraint RL | Framing reference | snippet |
| Safe learning for contact-rich tasks survey | arXiv 2025 | https://arxiv.org/abs/2512.11908 | Survey | Field is established | abstract |

Also seen: SHaRe-RL (2509.13949), ContactRL (2512.03707), "How to Peel with a Knife"
(2603.03280), DexNinja (2610.17208), Jia group fracture-mechanics cutting control,
Amigo et al. first-order RL through world models (2602.06219), TacForeSight (2606.11184).

## Density by area

1. Cutting with force adaptation to unknown material: moderately crowded (Hathaway
   line, Liu et al., SliceIt, Mitsioni, robocutting). Sparse: explicit hard force
   limit; commanded cutting speed as input.
2. Force/compliance-adaptive policies with implicit identification: crowded, almost
   all rigid assembly (FORGE, CoRMA, Xue).
3. Force-predicting learned models: crowded for imitation/VLA "world action models";
   sparse for RL/MPC (Ding, Yi, Mitsioni, DeepMPC). Policy gradients through a learned
   wrench model with a force constraint: not found.
4. Safe RL with hard/terminal/peak constraints: crowded in general; force peaks in
   manipulation handled by penalty, controller or safety critic.
5. Feed-rate adaptation under a force limit: classical control is crowded; learned
   versions sparse (Hathaway, constraint-grounded RL).

## Assessment

- Already done: one policy adapting feed and compliance from wrench on hidden
  materials with sim-to-real (Hathaway); force-threshold-aware policies inferring
  hidden dynamics (FORGE); force violation as termination with variable impedance
  (SRL-VIC); chance-constrained model-based RL (Peng); learned wrench models for MPC.
- Not found: knife cutting of deformables under one hard, episode-ending force limit
  across unknown materials with a speed command; a wrench world model used for
  backpropagated policy learning under a peak-force constraint; within-interval peak
  prediction.
- As a task claim the topic is incremental over Hathaway and SliceIt. It is defensible
  as a method claim only with three baselines in the same task: a FORGE-style
  recurrent force-penalty policy, terminating model-free safe RL, and a hand-tuned
  force-feedback speed governor, compared on violation rate at matched cutting speed.
- Narrower open angles: peak-force constraint under sensor-rate vs policy-rate
  mismatch; speed lost relative to an oracle that knows the material; real-robot
  demonstration that the limit holds on unseen materials (FORGE's real peak exceeds
  its threshold).
