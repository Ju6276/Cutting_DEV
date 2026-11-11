# Novelty check — force-aware world model for force-limited cutting

Status: paper and source review, updated 2026-11-10; contact-dynamics source inspection
included. Literature evidence and candidate gaps only; it does not add
development tasks. The research argument and algorithm are in [PAPER](PAPER.md);
current work is in [PLAN](PLAN.md). The proposed setting is now one shared WM
and policy trained across materials and prepared geometries (single-cut episodes).
The question remains how learning robot/force response helps physical cutting
control. Multiple objects do not by themselves establish novelty. Action-noise
evaluations are retired and excluded from the paper. Candidate uncertainty-aware
mechanisms below are literature directions, not adopted algorithms or new runs.

Labels: **T** paper text; **C** code; **I** inference; **A/S** abstract or search
snippet only. Claims marked "spot-checked" were re-read in the source by the
main session. Current reviewed texts and source snapshots stay on GPU7 (not in Git):
`/dataHDD/dong/Research/cutting_literature_20261110/`.
Individual scopes and code pins are indexed in [literature/README](literature/README.md).
Code findings below name the inspected functions and link to pinned revisions.
**C** records source inspection; execution results belong in [RESULTS](RESULTS.md).

> 2026-11-12: the problem statement this review served is retired (PAPER §1). Prior art for the
> candidate topic is in [literature/force_adaptive_search_20261112](literature/force_adaptive_search_20261112.md).
> The per-paper evidence below stays valid.

## 1. SliceIt (ICRA 2024, [2404.02569](https://arxiv.org/abs/2404.02569); code omron-sinicx/sliceit)

- Same setting: UR5e + FDCC, DiSECt + Gazebo, SAC outputs FDCC gains and pose
  increments (6-D), F/T history encoder [T, C]. No learned dynamics, no force
  prediction [T].
- Reward: paper Eq. 1 is inconsistent with the code (force penalty decreases with
  force as written); the code uses L1-normalized weights 0.25,
  `−σ(‖F̄‖/2 − 3)`, `−tanh(5|e_z|)`, step cost −0.25, +100 goal, −100 collision,
  −100 out of workspace [C]. Collision: any hand-frame wrench component above
  50 N / 10 N·m, checked at control rate [C]. Success: knife at board height for
  a cumulative >10 steps, not material separation [C].
- Training: 80K steps, about 40 h with DiSECt; no seed count [T].
- Randomization: paper says calibrated foods plus small noise, no ranges [T];
  the public code's parameter noise defaults to off, and the material appears
  fixed per run [C/I]. The inspected released bridge contains an unresolved function call; this
  establishes a release inconsistency, not the identity of the code used for
  the paper experiments [C/I].
- Unseen object: one carrot, real robot, **contact-force distributions only** — no
  success rate, time, completion or simulation evaluation [T, spot-checked].
  Approximate Fig. 7 medians 12/15/20/24 N (DiSECt+Gazebo) vs 18/22/30/31 N
  (Gazebo only); carrot maxima near 50 N [Fig].
- Stated limitation: compute time only [T].

## 2. DiSECt (Autonomous Robots, [2203.10263](https://arxiv.org/abs/2203.10263); code NVlabs/DiSECt)

- Trajectory optimization: 5 RBF keyframes, mean-knife-force objective, final
  height equality and lateral bound via MDMM; **no force-limit constraint**;
  open-loop, prescribed knife motion without robot, per object [T, C].
- Simulated cylinder 89.7 → 76.7 N (−14.5%); real Panda (glued fruit) mean force
  −42.7/−46.5/−46.7%, peak −26.5/−20.9/−43.9% for apple/banana/cucumber [T].
- Parameter inference: real apple 0.253 N, potato 0.379 N MAE; real-robot NMAE
  0.05–0.16 after inference [T].
- Our prism/jello/apple meshes are upstream `ansys_*` configs, but their
  materials differ from the paper's evaluated objects (prism E 40 kPa, jello
  43 kPa, apple 3 MPa) [C]: do not cite DiSECt numbers as ours.
- Inherited limitations we cannot claim as advances: predefined cut plane,
  static vertices, scalar knife force (zero torque in SliceIt's bridge).

## 3. Learned-dynamics MPC for cutting

| Work | Model predicts | Force/safety | Evaluation |
|---|---|---|---|
| DeepMPC, Lenz et al., RSS 2015 | Position; action is EE force | Hard input bound 30 N per axis [T, spot-checked] | 1488 cuts / 20 materials; held-out prediction and robot cutting-rate results, plus temperature/knife transfer; no interval-peak guarantee |
| Mitsioni et al., Humanoids 2019, [1903.03831](https://arxiv.org/abs/1903.03831) | Position only; reference-force input/proxy [T] | Only the sampling bound on reference force (8 N) | 5 trials/class, cutting rate; potato unseen |
| Mitsioni et al., CASE 2021, [2003.09179](https://arxiv.org/abs/2003.09179) | Position only (LSTM, horizon curriculum) [T, spot-checked] | Torque-limit shutdowns repeated and discarded; some foods pre-slit | Prediction MSE; mean MPC cost; success = reached board |
| Mitsioni et al., T-RO 2023 | State trajectories; β-VAE safe set | Soft cost on distance to safe set, plus abort | Accessible preprint: pendulum only; the published cutting experiment was **not read** — get the IEEE version |

No author cutting implementation was verified for these learned-MPC papers;
SafeDDC releases pendulum code [C]. DeepMPC and the 2019/2021 studies include
real-robot cutting; the retrieved T-RO preprint cannot establish its final
cutting experiment. Details and versions are in the individual reading records.

## 4. Constrained world-model RL and force-predicting world models

- **LAMBDA** (ICLR 2022, [2201.09802](https://arxiv.org/abs/2201.09802), code yardenas/la-mbda):
  Bernoulli cost head sharing gradients with the RSSM; augmented Lagrangian with
  a cost critic; **pathwise backprop through the world model** — i.e. WM + BPTT +
  constraint already exists. SWAG pessimism for cost. CMDP budget, violation not
  terminal [T, C]. §6 names integrating prior knowledge of the state→cost mapping
  into the world model "a notable open question" [T, spot-checked].
- **SafeDreamer** (ICLR 2024, [2307.07176](https://arxiv.org/abs/2307.07176)): separate cost head
  sharing gradients; Lagrangian variants; code does not stop the gradient of the
  cost return (pathwise). A `use_cost_model=False` variant computes cost from the
  decoded state with a non-differentiable threshold [C]. Safety-Gym navigation;
  violation not terminal; authors note safety depends on predicting
  safety-relevant states [T].
- **Visuotactile WM for lifting** ([2609.09597](https://arxiv.org/abs/2609.09597)): predicts endpoint
  and substep-peak forces with independent heads; peak MAE equals a persistence
  baseline; WM-RL with a predicted-peak penalty violates the force budget in
  57.8–93.3% of episodes and underperforms IQL (11.9% vs 25.0%); authors do not
  identify the mechanism [T]. Rigid boxes, simulation only.
- **Kuo et al.** (RA-L 2021, [2010.08169](https://arxiv.org/abs/2010.08169)): GP pMPC on joint
  states; action limits shrink with uncertainty; no force model [T].

## 5. Termination heads and gradient flow

- **SuperMimic** ([2610.28480](https://arxiv.org/abs/2610.28480)): heads are "either learned or
  computed analytically" (method §3.2) [T, spot-checked]; loss is an unweighted
  sum incl. termination BCE; soft survival through d̂ in BPTT; critic on imagined
  GAE; no BC. The paper does not discuss termination gradients into the
  dynamics. The reproduction feeds the termination head from the predicted state
  without detach [C, spot-checked], as ours does; its features scale height by
  1/0.25 m, ours divide target error by 2 mm (~125× larger Jacobian) [I].
- **DreamerV3** ([2301.04104](https://arxiv.org/abs/2301.04104), code danijar/dreamerv3): continue
  head on the full latent, gradient flows into the RSSM, no stop-gradient switch
  for it (there is one for reward) and no continue-head ablation [C]. Its state
  is a learned latent; ours is physical and supervised, so a pull away from the
  true state is bias [I].
- Precedents for repairs: MBPO applies hand-coded termination to the predicted
  next state [C]; LEQ ([2407.00699](https://arxiv.org/abs/2407.00699)) reports a large drop with a
  learned instead of true termination [T, partial]; Segmentation Dreamer
  ([2410.09972](https://arxiv.org/abs/2410.09972)) shows stopping an auxiliary binary head's
  gradient into the world model helps (mask head, not termination) [T, partial];
  HarmonyDream (2310.00344) balances WM loss terms [A].

## 6. Other learned models in cutting and material removal

- **Luo & Yuan, robocutting** ([2607.04367](https://arxiv.org/html/2607.04367),
  §§III–V): real UR5e + wrist F/T sensor; PPO changes desired downforce and
  sawing speed. The six observations are planar velocity, planar force and the
  two current targets. Reward trades downward progress against mechanical work
  [T]. §V-B trains five runs separately for each of eight foods; deployment
  starts from two knife-specific policies and adapts to the food [T]. §V-B.2
  reports frequent saturation of desired force at −30 N: progress reward can
  outweigh the extra work. This is a bound on the force command [T]. Greater
  executed force can therefore result from the objective even without a WM;
  our action-ranking comparison must distinguish this from model error [I].
  The [project page](https://robotproject8.github.io/robocutting/) links the
  paper/video; its GitHub link is the website template, with no algorithm-code
  or dataset download linked there at this review.
- **TopoCut** ([2509.19712, §7.9](https://arxiv.org/html/2509.19712)) predicts next
  particle positions and cluster membership from current topology and a cut
  mask. A GCN/Graph-Transformer encoder is trained with geometric losses
  (Chamfer/EMD/Hausdorff) and a Hungarian-matched topology loss, then reused by a
  goal-conditioned discrete-diffusion BC policy. MPPI/teleoperation supply
  demonstrations [T]. This model supervises cutting geometry and topology;
  its targets do not include the sensor-force waveform or a force-limit event
  [T]. The [official site](https://topocut.github.io/) says **Code (Coming Soon)**
  on 2026-11-08; code inspection remains pending publication.
- **RoboNinja** ([2302.11553, §III](https://arxiv.org/html/2302.11553), official
  code `real-stanford/roboninja` at `0d90994`): differentiable MPM generates
  expert trajectories; `OptimizationWorkspace.backward()` propagates through
  the physical rollout. `CloseLoopPolicyWorkspace.run()` instead trains the
  deployed policy with supervised action MSE [T, C; [optimization][rn-opt],
  [policy fitting][rn-fit]]. `CloseLoopPolicyDataset.__getitem__()` constructs
  three 256×256 channels: core mask/SDF, drawn knife pose and a constant
  tolerance map; targets are two angles (knife orientation and travel
  direction) with a configured step length [C; [dataset][rn-data],
  [kinematics][rn-dynamics]]. During execution, sparse collision locations
  update the core estimate, and collisions increase the policy's tolerance
  [C; [evaluation][rn-eval], [adaptive policy][rn-policy]]. RealSense records
  video in `RealEnv`; camera frames are absent from this policy input path
  [C; [real environment][rn-real]]. The paper converts measured force to binary
  collision feedback using a manually chosen threshold for each material [T].
  Its estimation target is hidden core geometry; continuous force-peak
  forecasting is a different target [I].
- **Learning to Shape by Grinding** ([2308.02150](https://arxiv.org/abs/2308.02150)): cutting-surface-aware
  model-based RL in robotic grinding (material removal, not food slicing) [A].
- With DeepMPC and Mitsioni (section 3), learned-model control of cutting exists;
  "no world model or MBRL in cutting" is not a valid claim.

## 7. Learning contact and impact dynamics

- **ContactNets** (CoRL 2020, [2009.11193, §4](https://arxiv.org/html/2009.11193)),
  official `DAIRLab/contact-nets`, revision `c3dc21c`: learns signed distances
  and contact Jacobians; rigid-body contact mechanics produces impulses [T].
  `StructuredLearnable.compute_phi/compute_Jn` ties the normal Jacobian to the
  distance network [C; [model][cn-model]]. `PolyGround3DSurrogate.compute_loss`
  infers net impulse from observed velocity change and known non-contact
  dynamics, then fits complementarity, impulse matching and friction terms
  through a differentiable QP [C; [loss][cn-loss]]. The provided prediction
  path calls `solve_lcp_batch`; `solve_lcp` detaches tensors, uses NumPy/Lemke,
  and constructs a new tensor. Its QP fitting path and LCP rollout have
  different gradient behavior [C; [solvers][cn-solvers]]. The useful precedent
  is coupling contact geometry, impulses and state change. Using this rollout
  inside actor BPTT would require a gradient implementation for contact
  resolution; a rigid impulse also differs from our filtered sensor peak [I].
- **Learning Contact Dynamics using Physically Structured Neural Networks**
  (AISTATS 2021, [2102.11206, §3](https://arxiv.org/html/2102.11206)), official
  `libeanim/contact-symplectic-integrator-network`, revision `1a7f27d`: learns
  potential energy and contact activation, with smooth and impact velocity
  updates [T, C]. `TRAIN()` passes contact labels into the model;
  `BaseNetwork.forward()` supplies each labelled contact to the rollout;
  `predict_forward()` instead passes `None`. The contact model then uses
  `ctf.numpy() > 0.5` for a hard decision. Its fitting objective combines state
  MSE and contact BCE [C; [training][cd-train], [rollout][cd-base],
  [contact model][cd-model]]. Experiments cover a pendulum, bouncing ball and
  Newton's cradle, with idealised touch feedback [T]. This supplies a concrete
  contact-label supervision precedent; its prediction contact decision breaks
  the gradient through activation, so it is not a ready BPTT replacement [I].
- **Khader et al., Data-efficient Model Learning and Prediction for
  Contact-rich Manipulation Tasks** (RA-L 2020,
  [1909.04915, §§III–IV](https://arxiv.org/html/1909.04915)): clusters contact
  modes, fits a GP within each mode and a reset map across modes, and learns
  switching with an SVM. Uncertainty in impact time produces multiple possible
  future trajectories [T]. Its robot formulation assumes rigid, static
  surroundings so joint position/velocity can represent state [T]. It motivates
  separating timing error from response error; applying the same state
  assumption to deforming, damaged material would omit history [I]. Author implementation is now inspected; see
  [the hybrid GP record](literature/hybrid_contact_gp.md) for its pin and scope.

## 8. Our own implementation, for comparison with the above

- Events from predicted state already exist: `imagine()`
  (`cutting_world_model/trainer.py:164`) ends imagination when the predicted
  `peak_wrench` exceeds the limit or the task predicates fire on the predicted
  state (absorbing; reward counted once), and also keeps the learned terminal
  probability. Hard predicates give no gradient across the boundary, and an
  under-predicted force still yields a wrong event; removing the classifier alone
  does not address that.
- Task definitions (force-limit check on the filtered sensor wrench, material
  force reported separately, success = goal-pose counter) are in
  [BENCHMARK](BENCHMARK.md) "Task definitions"; this document does not redefine them.

### Implications for the current force-prediction failure

- The new contact papers establish precedents for event-aware dynamics and
  physics-linked state/contact predictions. These general ideas already exist;
  the candidate contribution concerns their effect on compliant cutting and
  BPTT policy learning [T/I].
- The A comparison in [RESULTS](RESULTS.md#2026-11-08--wm-reliability-check-steps-1-and-3-route-a-vs-original)
  separates two observations: removing BCE gradients from dynamics improves
  position prediction, while force-limit misses remain. The fixed-fit training
  split contains no force-limit failures. Current evidence leaves both coverage
  and model representation as explanations for poor force prediction [I].
- Compare contact timing, peak amplitude and the sensor/filter response on the
  recorded physical time series. A per-control-step peak summarizes a whole
  interval, whereas an endpoint force describes one instant. The 50 ms policy
  interval and 2 ms coupling checks in BENCHMARK make this distinction relevant
  [I]. Early episode termination alone does not identify which physical or
  controller event produced the peak; use the contact and force trace [I].
- The completed Step 2 model-versus-physics branches capture the higher force
  on the five checked successful trajectories. Missed success events instead
  cause critic queries beyond the physical ending. The compared score is a
  proxy for the actor objective; a causal control effect remains open (RESULTS,
  2026-11-08) [I]. The paired A experiment is now complete: its pre-stated
  final-success comparison did not detect a clear benefit. This historical
  motivation does not add another prediction-diagnostic stage.

## 9. Deep-reading update: what constrains route B

Updated 2026-11-10. [Individual records](literature/README.md) replace the
earlier abstract-only B survey. Each gives the retrieved paper version, actual
read scope, algorithm, data/metrics, relevant limitation, code provenance and
inspected implementation paths. Code reading is not an executed reproduction.

| Established idea | Direct precedents | Consequence here |
|---|---|---|
| Learned cutting dynamics and online MPC across materials | DeepMPC; Mitsioni 2019/2021 | Neither a learned cutting WM nor WM+MPC is new. |
| Short learned-model rollout, endpoint value, analytic policy update | MAAC; SuperMimic; constrained LAMBDA/CGPO | WM+BPTT or adding a constraint alone is insufficient. |
| Uncertainty-aware force/compliance control | Kuo; Haninger; Matschek | Force uncertainty and GP impedance MPC already exist. |
| Contact-mode dynamics and structured impulse/state updates | Hybrid contact GP; ContactNets; CD-Lagrange | Mode splitting or a physics residual alone is insufficient. |
| Conservative model costs or prediction bounds | CAP; MOPO; CQR; CaPTURe | Calibration/upper bounds are inherited machinery, not a new method. |
| Endpoint and interval-peak force prediction in a WM | Visuotactile lifting (2609.09597) | Retract the earlier search-based suggestion that peak-force WMs were not found. |
| Learned action sensitivity of constraints and action correction | Dalal safety layer; safe/force MPC | Action gradients are not themselves the contribution. |
| Value targets on imagined states and model/critic uncertainty | MVE; STEVE; LEQ | Accurate physical-return estimates do not establish imagined-state reliability. |

Important source details: DreamerV3's actor uses REINFORCE in the inspected
implementation, whereas our actor uses BPTT. Its stop-gradient choices cannot
be transplanted by name. Learning on the Fly uses a hybrid forward model but
a custom analytical backward derivative; it is not exact differentiation of
the complete learned forward map. STEVE's original code remains available at
a historical TensorFlow Models commit. MVE's author implementation explicitly
does not predict new terminal states. These are adaptation boundaries, not
reasons to replace our stack.

Some close priors have limited releases: Haninger's linked author GitLab was
unreachable, CERL's code endpoint returned 403, and the visuotactile arXiv source
archive did not contain the advertised implementation. Passivity-Centric
releases data and weights, which must not be mislabeled source code.
Third-party Dalal and SuperMimic implementations are separately labeled.
The T-RO final cutting section remains a coverage limitation; its pendulum
preprint and SafeDDC release cannot stand in for it.

Two resource corrections also matter. [Sashimi-Bot](literature/sashimi_bot.md)
uses RL for shape manipulation; its cutting is a geometry-based planner with
tactile board-contact correction. [CulinaryCut](literature/culinarycut.md)
has materially different manuscript/project versions and a currently
inaccessible advertised code link; public samples do not establish access to
the algorithm. Its older force-response speed limiter is relevant prior art.
[Playing with Food](literature/playing_with_food.md) provides real multimodal
interaction data, not just visual objects.

The earlier general claim that other papers do not use perturbation evaluations
is not supported by the contact-safety literature. Our old suite remains
retired because it does not serve the chosen task/claim; this correction does
not reactivate it.

## Assessment and recommended focus

**How precedents are judged (user, 2026-11-10).** A related paper does not mean
the problem is solved. Each precedent is judged by its setting, assumptions and
measured results (e.g. the visuotactile WM's BPTT agent underperformed IQL;
Dalal's sensitivity layer was shown on toy position constraints). Where the
precedents stop is summarized in [PAPER](PAPER.md) section 3; precedent
approaches become baselines when the paper claims to do better. The text below
is the earlier synthesis and keeps its evidence; read its "not new" wording
through this rule.


**Method novelty is unresolved. It depends on B or another justified mechanism,
not on S0–S4 integration succeeding.** B's +4-event pilot gate decides whether
to spend the already capped control-screen budget; it does not establish novelty,
a significant population improvement or safe deployment.

The recommended candidate question is whether a history-conditioned cutting
model can support useful action changes near changing contact, rather than
merely predicting recorded force values. A possible mechanism concerns the
region of candidate compliance/reference actions for which force-risk
predictions remain trustworthy under changing cutting history. This is an
inference and formulation direction, not a verified literature gap or a new
implemented module. Local action sensitivity, trust regions, contact-dependent
calibration and hybrid controller models all have precedents above.

The required distinction would be a concrete treatment of changing interaction
during compliant cutting, with action-dependent force risk and demonstrated
physical control benefit. Compare against a calibrated ensemble and the
unmodified working procedure; do not require that every standard method fail.
The current within-interval peak output already exists. Adding it again, adding
a constant safety margin, or giving the whole method a new name is not a method.

Keep cutting as the task domain. Non-cutting papers provide formulation
evidence, not a requirement to add insertion/pushing/grinding benchmarks.
S0–S2 build the shared task distribution independently; S3 trains fresh networks.
Historical B fits and old-checkpoint MPPI are bounded screens, not the main
multi-condition learning evidence.

## 10. Method precedents and scope correction — 2026-11-10

These precedents still constrain novelty after broadening the proposed training
task. The earlier nominal-only recommendation has been replaced by shared
multi-condition training in PAPER/PLAN. The ten-episode zero-shot material pilot
remains withdrawn; it is not the new training-distribution proposal. A's control
comparison is complete and did not establish its pre-stated final-success
benefit. This review does not authorize a new run.

- **MAAC**, [2005.08068](https://arxiv.org/pdf/2005.08068): §§2, 4.1 and 5.4
  describe differentiating through a short learned-model rollout with an
  endpoint value and comparing policy execution with model-predictive control
  [T; targeted paper reading this review, not a code audit]. This is a direct
  precedent for the generic framework and comparison. Our cutting interface,
  training data and evidence differ; a priority claim for WM+BPTT or same-WM
  planning comparison is unsupported [I].
- **TD3+BC**, [official repository](https://github.com/sfujim/TD3_BC):
  its actor implementation and training entry document weighted behavior cloning
  and state normalization [C; see the pinned [TD3+BC record](literature/td3_bc.md)]. Our METHOD already attributes the
  replay-action regularizer to this work. We use it during online continuation;
  that does not turn the procedure into the original offline-RL algorithm [I].

The remaining claim must concern a justified cutting-control formulation and
its physical effects. Existing method ingredients can form a task-specific
algorithm, but neither renaming them nor reporting their integration makes a
new generic algorithm. See PAPER's evidence table and review questions.

## Candidate comparisons (depend on the final claim; not current tasks)

Current tasks are only those in [PLAN](PLAN.md). Candidates, each needing its own
plan and an estimate of integration and validation cost:

- SAC with the SliceIt code reward (state any multi-material randomization that
  goes beyond their release).
- The same WM + MPC (CEM/MPPI) with the predicted force event as constraint or
  penalty; optionally a Mitsioni-style position-only LSTM + shooting MPC.
- A LAMBDA-style augmented-Lagrangian force constraint on our WM.
- Ablations: learned event head vs events derived from predicted state/force;
  with vs without event-loss gradient into the dynamics; fixed −100 vs adaptive λ.
- DiSECt trajectory optimization as a per-object open-loop reference.
- Metrics follow BENCHMARK; possible additions: WM force-peak error with
  persistence as a reference point, event calibration (Brier/AUROC).

## Still open

- IEEE version of Mitsioni T-RO 2023 (cutting section).
- Google Scholar "cited by" pass on SliceIt and DiSECt.
- Unread sections are listed explicitly in individual reading records; do not
  promote a scoped method/code read into a complete appendix/proof audit.
- Inspect TopoCut code after publication; check for a robocutting algorithm/data
  release beyond its current project-page links.

[rn-opt]: https://github.com/real-stanford/roboninja/blob/0d90994d1a23d5f595b53f83ba0abff3643bd848/roboninja/workspace/optimization_workspace.py
[rn-fit]: https://github.com/real-stanford/roboninja/blob/0d90994d1a23d5f595b53f83ba0abff3643bd848/roboninja/workspace/close_loop_policy_workspace.py
[rn-data]: https://github.com/real-stanford/roboninja/blob/0d90994d1a23d5f595b53f83ba0abff3643bd848/roboninja/dataset/close_loop_policy_dataset.py
[rn-dynamics]: https://github.com/real-stanford/roboninja/blob/0d90994d1a23d5f595b53f83ba0abff3643bd848/roboninja/utils/dynamics.py
[rn-eval]: https://github.com/real-stanford/roboninja/blob/0d90994d1a23d5f595b53f83ba0abff3643bd848/roboninja/workspace/eval_workspace.py
[rn-policy]: https://github.com/real-stanford/roboninja/blob/0d90994d1a23d5f595b53f83ba0abff3643bd848/roboninja/cutting_policy/cutting_policy_adaptive.py
[rn-real]: https://github.com/real-stanford/roboninja/blob/0d90994d1a23d5f595b53f83ba0abff3643bd848/roboninja/env/real_env.py
[cn-model]: https://github.com/DAIRLab/contact-nets/blob/c3dc21c94577ff4ed99123f0b00bfe57bb2d1d90/contactnets/experiments/block3d/structured_learnable.py
[cn-loss]: https://github.com/DAIRLab/contact-nets/blob/c3dc21c94577ff4ed99123f0b00bfe57bb2d1d90/contactnets/train/poly_ground_3d_surrogate.py
[cn-solvers]: https://github.com/DAIRLab/contact-nets/blob/c3dc21c94577ff4ed99123f0b00bfe57bb2d1d90/contactnets/utils/solver_utils.py
[cd-train]: https://github.com/libeanim/contact-symplectic-integrator-network/blob/1a7f27d5821549c91ef7bed818364c6f6535b36a/utils/training.py
[cd-base]: https://github.com/libeanim/contact-symplectic-integrator-network/blob/1a7f27d5821549c91ef7bed818364c6f6535b36a/models/base.py
[cd-model]: https://github.com/libeanim/contact-symplectic-integrator-network/blob/1a7f27d5821549c91ef7bed818364c6f6535b36a/models/cd_lagrange_network.py
