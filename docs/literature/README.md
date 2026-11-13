# Literature review index — 2026-11-10

This is the per-paper evidence ledger requested by the user, produced by Codex
with one literature subagent. The former READING_LIST is consolidated below;
individual method, simulator and data notes retain their original reading scope.
[Synthesis](../NOVELTY.md), [paper argument](../PAPER.md) and
[current task list](../PLAN.md) have different roles.

Each record states **what was actually read**, method and data, relevant
experiments/metrics, limitations, and inspected code/version where available.
A scoped method/experiment read is not a complete proof or dependency audit.
A downloaded PDF, an abstract, a README or an advertised Code button is not
counted as an implementation review. None of these external algorithms was
trained or numerically reproduced in this review. No sealed test was opened.

The 2026-11-10 synthesis was conditional: the inherited WM+BPTT framework and a
multi-object benchmark do not themselves establish a new method. The then-proposed route B needed
a distinction from existing conservative world-model control, force MPC and
contact-conditioned calibration. That review suggested a research question concerning
useful action changes under changing cutting interaction; it is a hypothesis,
not a claim that these papers prove an untouched gap.

## Storage and version provenance

PDF/text/source files and code snapshots stay on **GPU7** under
`/dataHDD/dong/Research/cutting_literature_20261110/`:
parent review material in the root / `parent-code/`, subagent material in
`contact_risk/`. Notes alone enter Git. No source/data/PDF copies were
downloaded to the Mac. Individual notes name pinned commits and retrieval
limitations. Current pages can differ from an older manuscript.

## Reading coverage

### Cutting, simulation and datasets

| Record | Paper reading scope | Code / resource scope |
|---|---|---|
| [sliceit](sliceit.md) | Main paper + task/bridge source | Author source, pinned; task-version differences recorded |
| [disect](disect.md) | Journal method/experiments + optimization source | Author source, pinned; RSS version not a separate full read |
| [deepmpc](deepmpc.md) | Extended author paper, model/controller/experiments | No author algorithm/data release verified |
| [mitsioni 2019](mitsioni_2019.md) | Dynamics, control and robot experiments | No author training implementation verified |
| [mitsioni 2021](mitsioni_2021.md) | Recurrent model, control and experiments | No author training implementation verified |
| [mitsioni safeddc](mitsioni_safeddc.md) | Accessible preprint method/results | Author pendulum source inspected; final T-RO cutting section unavailable |
| [robocutting](robocutting.md) | System, PPO, data and robot results | No released training repository retrieved |
| [roboninja](roboninja.md) | Optimization teacher, student, real system/results | Author core source, pinned |
| [topocut](topocut.md) | Main method/results + technical appendix sections | Official code still marked Coming Soon |
| [dexninja](dexninja.md) | Data engine, policy and real experiments | Assets released; algorithm source not retrieved |
| [blademaster](blademaster.md) | Forward mechanics and experiments | Paper evidence; local reproduction is not author code |
| [cressim](cressim.md) | MPM, cutting/contact, coupling, demos/limits | Author contact/coupling source, pinned |
| [sashimi bot](sashimi_bot.md) | Shape RL, cutting/tactile method and results | Publisher manuscript says full source not public |
| [grinding](grinding.md) | Model, MPC and material-removal experiments | No author algorithm repository verified |
| [zhang multimodal cutting](zhang_multimodal_cutting.md) | Multimodal phase inference, DMP, experiments | No author algorithm release verified |
| [playing food](playing_with_food.md) | Dataset, collection and representation experiments | Public dataset/linked resources; scope in note |
| [culinarycut](culinarycut.md) | v1 paper/appendices + current release check | Version and live code/data accessibility distinguished |

### World models, policy/value learning

| Record | Paper reading scope | Code / resource scope |
|---|---|---|
| [supermimic](supermimic.md) | Paper method/results + observation/architecture details | Community reproduction core source, pinned; not author code |
| [maac](maac.md) | Pathwise model policy update and experiments | No author implementation verified |
| [dreamerv3](dreamerv3.md) | Core method + termination/actor source | Author core source, pinned |
| [harmony](harmony.md) | Loss balancing and experiments | Author loss source, pinned |
| [segmentation dreamer](segmentation_dreamer.md) | Model, experiments and relevant appendix | Algorithm code not retrieved |
| [visuotactile wm](visuotactile_wm.md) | v2 method, all experiments, calibration/actions | Code advertised; retrieved source archive lacked implementation |
| [leq](leq.md) | Critic/actor method, experiments and failure discussion | Author core source, pinned |
| [td3 bc](td3_bc.md) | Objective and experiments | Author actor/train source, pinned |
| [mbpo](mbpo.md) | Branched rollouts and experiments | Author core source, pinned |
| [mve](mve.md) | Method, error argument, TD-k, main experiments | Author target construction, pinned |
| [steve](steve.md) | Method, main experiments and ablations | Author archived TensorFlow source, pinned |
| [mopo](mopo.md) | Theory assumptions, implementation and main results | Author uncertainty/rollout source, pinned |
| [learning on fly](learning_on_fly.md) | Hybrid model/derivative method and results | Author custom backward and model fit, pinned |

### Contact models and force-aware control

| Record | Paper reading scope | Code / resource scope |
|---|---|---|
| [kuo contact safe](kuo_contact_safe.md) | Model, safety formulation and robot results | Paper evidence; no author code verified |
| [contactnets](contactnets.md) | Contact identification, solver and experiments | Author core source, pinned |
| [cd lagrange](cd_lagrange.md) | Structured contact dynamics and experiments | Author train/inference source, pinned |
| [hybrid contact gp](hybrid_contact_gp.md) | Hybrid modes/reset map and robot experiments | Author Python 2 implementation inspected |
| [parmar contact learning bias](parmar_contact_learning_bias.md) | Full paper and appendices | Author predictor/rollout/data-generation source, pinned |
| [ahac](ahac.md) | Full paper and appendices | Author actor/horizon source, pinned |
| [hewing cautious gp mpc](hewing_cautious_gp_mpc.md) | Full main paper and appendix | No author implementation specifically reproducing this paper verified |
| [matschek force mpc](matschek_force_mpc.md) | Full paper and appendices | No independent author algorithm release verified |
| [haninger gp impedance](haninger_gp_impedance.md) | Full main paper | Author GitLab endpoint unreachable |
| [passivity centric](passivity_centric.md) | Full paper and appendix | Official data/weights archive; no source in retrieved archive |
| [srl vic](srl_vic.md) | Full paper and appendix | RecoveryRL dependency inspected; not SRL-VIC author implementation |
| [contact survey](contact_safety_survey.md) | All body sections and end tables; cited works not individually read | Author literature/resource scripts inspected; not a control implementation |

### Constrained learning and calibration

| Record | Paper reading scope | Code / resource scope |
|---|---|---|
| [safe pilco](safe_pilco.md) | Full main paper | Official source inspected; scope in note |
| [charvet robust pilco](charvet_robust_pilco.md) | Full main paper + appendix | No author algorithm repository verified |
| [lambda](lambda.md) | Full main paper + appendices A–J | Author WM/actor/critic/SWAG source, pinned |
| [safedreamer](safedreamer.md) | Full main paper + appendices A–C | Author WM/actor/planners source, pinned |
| [cerl](cerl.md) | Full main paper and appendices | Linked repository inaccessible (403) |
| [cgpo](cgpo.md) | Full paper + appendices A–F | No attributable author repository retrieved |
| [cap](cap.md) | Full main paper + appendices A–E | Author constrained planners/penalty source, pinned |
| [smbpo](smbpo.md) | Full main paper + appendices A.1–A.3 | Author rollout/SAC/dynamics source, pinned |
| [cpo](cpo.md) | Full main paper and appendix | Author trust-region/constraint source, pinned |
| [dalal safety layer](dalal_safety_layer.md) | Full main paper and appendix | Third-party safe-explorer inspected; not official |
| [chee conformal control](chee_conformal_control.md) | Full paper + appendix | No author algorithm source retrieved |
| [capture](capture.md) | Full main paper + appendices A/B | Project released; algorithm source not retrieved |
| [cqr](cqr.md) | Full paper + appendices A/B | Author quantile/calibration source, pinned |
| [conformal tutorial](conformal_tutorial.md) | Full tutorial; relevant guarantees examined | Theory reference; tutorial companion status in note |

## Explicit gaps

- The final T-RO 2023 cutting section remains unavailable. Its accessible
  pendulum preprint and SafeDDC source were read and recorded separately.
- The contact-safety survey body and end tables were read; its bibliography
  entries are not implicitly treated as individually read papers.
- Original RSS DiSECt is covered here through the extended journal version;
  this is not a claim of two independent full-paper reviews.
- Unavailable code, advertised-but-inaccessible releases and third-party
  reproductions stay distinct. No claim of repository-wide correctness follows
  from reading its relevant core paths.
- Appendix sections outside the stated note scope and systematic citation
  tracing are not marked complete. These gaps do not authorize new training
  jobs or an indefinite review queue.


## Paper and data entry points

Consolidated from READING_LIST on 2026-11-13. Availability and reading-scope claims
are dated observations, not fresh checks of external releases. The current
research scope is [PAPER](../PAPER.md); this bibliography does not schedule work.

Priority: **A** decides novelty or the paper's framing; **B** method background;
**C** context.

### Robotic cutting: closest competitors

| P | Paper | Why read |
|---|---|---|
| A | Luo, Yuan. *A Perception-Manipulation Robotics System for Food Cutting*. IROS 2026. [arXiv:2607.04367](https://arxiv.org/abs/2607.04367), [project](https://robotproject8.github.io/robocutting/) | Real UR5e food cutting. PPO tunes target downward force and sawing speed. Reward is downward progress minus mechanical work. Ten foods, one policy per food, plus knife selection from trial cuts. Their actions bound force to [-30, 0] N. Closest real-robot RL work. |
| A | Beltran-Hernandez, Erbetti, Hamaya. *SliceIt! A Dual Simulator Framework for Learning Robot Food Slicing*. ICRA 2024. [arXiv:2404.02569](https://arxiv.org/abs/2404.02569) | Our task, reward and SAC baseline. Paper force-penalty formula differs from the code; see the pinned SliceIt discussion in [NOVELTY](../NOVELTY.md). |
| A | Wang, Bian, Heiden, Garg. *TopoCut: Learning Multi-Step Cutting with Spectral Rewards and Discrete Diffusion Policies*. arXiv 2025. [arXiv:2509.19712](https://arxiv.org/abs/2509.19712) | Cutting benchmark (MLS-MPM, Taichi): slice, stick and dice goals, seen and novel shapes. Kinematic knife; no sensor-force prediction target in the inspected method. Goal-conditioned diffusion behavior cloning; code still advertised as forthcoming. See the individual note; this does not by itself establish our novelty. |
| A | Xu et al. *RoboNinja: Learning an Adaptive Cutting Policy for Multi-Material Objects*. RSS 2023. [arXiv:2302.11553](https://arxiv.org/abs/2302.11553) | Differentiable MPM; closed-loop policy trained from optimized trajectories; collision and energy metrics. |
| A | Heiden et al. *DiSECt: A Differentiable Simulator for Parameter Inference and Control in Robotic Cutting*. Autonomous Robots. [arXiv:2203.10263](https://arxiv.org/abs/2203.10263) | Our material model; gradient trajectory optimization baseline; Ansys force data for sim validation. |
| B | Heiden et al. *DiSECt: A Differentiable Simulation Engine for Autonomous Robotic Cutting*. RSS 2021. [arXiv:2105.12244](https://arxiv.org/abs/2105.12244) | Original DiSECt; mean-force and time objective. |
| B | Lou et al. *DexNinja: Learning Robust Dexterous Cutting Policy with a Real-to-Sim-to-Real Data Engine*. arXiv 2026. [arXiv:2610.17208](https://arxiv.org/abs/2610.17208) | Dexterous-hand cutting with RGB-D and fingertip tactile input; ACT-style behavior cloning; differentiable MPM data engine built on RoboNinja; category-specific policies and real-robot generalization; assets release differs from algorithm code. |
| B | Yang et al. *BladeMaster: Real-Time Robotic Cutting Simulation with Online-Generated Persistent Discontinuities*. arXiv 2026. [arXiv:2609.27342](https://arxiv.org/abs/2609.27342) | GPU TLMPM cutting with tool reaction forces and persistent cuts; no learning. Forward mechanics reference; no new simulator migration is planned. |
| C | *CRESSim-MPM: A Material Point Method Library for Surgical Soft Body Simulation with Cutting and Suturing*. IROS 2025. [arXiv:2502.18437](https://arxiv.org/abs/2502.18437) | GPU MPM cutting, surgical domain. |
| C | *Sashimi-Bot: autonomous tri-manual advanced manipulation and cutting of deformable objects*. npj Robotics 2026. [link](https://www.nature.com/articles/s44182-026-00098-9) | SAC transfers shape-pushing control; cutting uses geometric trajectories with tactile board-contact correction. Do not call cutting itself an RL policy. |
| C | *Learning to Shape by Grinding: Cutting-surface-aware Model-based Reinforcement Learning*. [arXiv:2308.02150](https://arxiv.org/abs/2308.02150) | One of few model-based RL papers on material removal. |

### Safe and force-constrained learning for contact

| P | Paper | Why read |
|---|---|---|
| A | Kuo et al. *Uncertainty-aware Contact-safe Model-based Reinforcement Learning*. RA-L 2021. [arXiv:2010.08169](https://arxiv.org/abs/2010.08169) | Closest prior work to force-constrained MBRL: model uncertainty for contact safety, mixing and scooping on real robots. Must state our difference. |
| A | *Passivity-Centric Safe Reinforcement Learning for Contact-Rich Robotic Tasks*. [arXiv:2503.00287](https://arxiv.org/abs/2503.00287) | Model-free constrained RL with a measured-force constraint and energy tanks. |
| A | *Safe Learning for Contact-Rich Robot Tasks: A Survey from Classical Learning-Based Methods to Safe Foundation Models*. [arXiv:2512.11908](https://arxiv.org/abs/2512.11908) | Map of the field; check for cutting-specific work we missed. |
| B | As et al. *Constrained Policy Optimization via Bayesian World Models* (LAMBDA). ICLR 2022. [arXiv:2201.09802](https://arxiv.org/abs/2201.09802) | Constrained MBRL with world-model uncertainty. |
| B | Huang et al. *SafeDreamer: Safe Reinforcement Learning with World Models*. ICLR 2024. [arXiv:2307.07176](https://arxiv.org/abs/2307.07176) | Lagrangian constraints in a world-model agent. |
| B | Achiam et al. *Constrained Policy Optimization*. ICML 2017. [arXiv:1705.10528](https://arxiv.org/abs/1705.10528) | Constrained MDP baseline formulation. |
| B | Dalal et al. *Safe Exploration in Continuous Action Spaces*. [arXiv:1801.08757](https://arxiv.org/abs/1801.08757) | Safety layer; the usual alternative to a learned constraint. |
| C | *SRL-VIC: A Variable Stiffness-Based Safe Reinforcement Learning for Contact-Rich Robotic Tasks*. [arXiv:2406.13744](https://arxiv.org/abs/2406.13744) | Safe RL through variable impedance, like our compliance actions. |

### Model-based RL and value-learning precedents

| P | Paper | Why read |
|---|---|---|
| A | *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*. CoRL 2026. [arXiv:2610.28480](https://arxiv.org/abs/2610.28480) | Community reproduction is the active base; the separately inspected official implementation is mapped in [SOURCES](../SOURCES.md#official-supermimic-adaptation). |
| B | Janner et al. *When to Trust Your Model: Model-Based Policy Optimization* (MBPO). [arXiv:1906.08253](https://arxiv.org/abs/1906.08253) | Short model rollouts augment SAC replay; this is not our BPTT actor implementation. |
| B | Feinberg et al. *Model-Based Value Estimation* (MVE). [arXiv:1803.00101](https://arxiv.org/abs/1803.00101) | Model-expanded targets and TD-k at imagined states; not an implementation or justification of our `real`/`real_wm` targets. |
| B | Buckman et al. *STEVE*. [arXiv:1807.01675](https://arxiv.org/abs/1807.01675) | Ensemble-weighted value expansion. |
| B | Fujimoto, Gu. *A Minimalist Approach to Offline RL* (TD3+BC). [arXiv:2106.06860](https://arxiv.org/abs/2106.06860) | Precedent for actor BC regularization; original offline objective includes Q scaling and is not identical to our fixed online coefficient. |
| C | Yu et al. *MOPO: Model-based Offline Policy Optimization*. [arXiv:2005.13239](https://arxiv.org/abs/2005.13239) | Uncertainty-penalized model rewards. |
| C | *Learning on the Fly: Rapid Policy Adaptation via Differentiable Simulation*. RA-L. [arXiv:2508.21065](https://arxiv.org/abs/2508.21065) | Differentiable simulation plus residual dynamics; RA-L format reference. |

### Object and material data

DiSECt material parameters are not only Young's modulus and Poisson ratio; the
cut-spring parameters must be fitted to a knife force profile. Literature moduli
alone therefore do not define a DiSECt food.

| Source | Contents | Status |
|---|---|---|
| DiSECt dataset ([repo](https://github.com/NVlabs/DiSECt), 96 MB, CC BY-NC 4.0) | Meshes plus ANSYS LS-DYNA simulated knife forces and nodal motion: prism, cylinder (jello), sphere (apple). Configs in `examples/config/`. | Used; prism was the historical L1 object; the current free cucumber is a separate scanned mesh. |
| SliceIt OSX dataset (link in `SliceIt/disect/README.md`, CC BY-NC 4.0) | Real knife-force profiles (`calibrated/*.npy`) referenced by `examples/config/osx/` for cucumber (E 2.6 MPa), potato (2.0 MPa) and tomato (40 kPa). The three configs share default cut-spring parameters (ke 200, softness 200, surface softness 300), so they are not fitted DiSECt materials. | Google Drive link returned "page not found" on 2026-11-07; ask the authors. A cucumber mesh is in `catkin_ws/src/o2ac_gazebo/models/cucumber`. |
| Sawhney et al. *Playing with Food*. [arXiv:2101.02252](https://arxiv.org/abs/2101.02252) | Real robot cutting of multiple foods with force, audio, images and proprioception; publicly downloadable. | See the data/module review in playing_with_food.md; dataset access and task labels differ from our online transitions. |
| Luo, Yuan (robocutting). [arXiv:2607.04367](https://arxiv.org/abs/2607.04367) | Force, velocity and displacement from trial cuts on 14 foods with a UR5e. | Release not confirmed. |
| Lenz et al. *DeepMPC*. RSS 2015 | 1488 PR2 cutting examples over 20 materials. | Release and modalities not confirmed. |
| Zhang et al. *Leveraging Multimodal Haptic Sensory Data for Robust Cutting*. [arXiv:1909.12460](https://arxiv.org/abs/1909.12460) | 25 materials/conditions, force and audio fusion for six cutting phases; DMP execution. | Release not confirmed. |
| *CulinaryCut-VLAP*. [arXiv:2601.06451](https://arxiv.org/abs/2601.06451) | Simulation cutting demonstrations and force annotations; versions differ. | See culinarycut.md for v1 versus current project release, and actual code/data accessibility. |
| Jia lab, Iowa State (via DiSECt) | Real apple and potato knife forces with laser-scanned meshes. | Provided to DiSECt; not public. |
| GenVegeFruits3D ([LIRIS](https://datasets.liris.cnrs.fr/genvegefruits3d-version1)) | Meshes for 100 fruit and vegetable categories. | Geometry only; partial release. |
