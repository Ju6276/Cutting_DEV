# Literature review index — 2026-11-10

This is the per-paper evidence ledger requested by the user, produced by Codex
with one literature subagent. It covers the named method papers in READING_LIST
and NOVELTY, with separately identified simulator and data references.
[Synthesis](../NOVELTY.md), [paper argument](../PAPER.md) and
[current task list](../PLAN.md) have different roles.

Each record states **what was actually read**, method and data, relevant
experiments/metrics, limitations, and inspected code/version where available.
A scoped method/experiment read is not a complete proof or dependency audit.
A downloaded PDF, an abstract, a README or an advertised Code button is not
counted as an implementation review. None of these external algorithms was
trained or numerically reproduced in this review. No sealed test was opened.

The main conclusion is conditional: the inherited WM+BPTT framework and a
multi-object benchmark do not themselves establish a new method. B also needs
a distinction from existing conservative world-model control, force MPC and
contact-conditioned calibration. The recommended research question concerns
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
