# Novelty check — force-aware world model for force-limited cutting

Status: deep read, 2026-11-08 (first pass the same day). Framing under test
(user, 2026-11-08): a reliable force-aware world model (state, force and
cutting-event predictions consistent) with BPTT policy learning; multi-material
evaluation shows that one policy cuts several objects under the 50 N limit.
Robot-methods paper (RA-L/ICRA/IROS).

Labels: **T** paper text; **C** code; **I** inference; **A/S** abstract or search
snippet only. Claims marked "spot-checked" were re-read in the source by the
main session. Downloaded texts and clones (not in Git):
`/private/tmp/claude-501/-Users-judong/4c5bdeab-87d4-4a26-b99c-e7d9a689c037/scratchpad/{src,pdf}`.

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
  fixed per run [C/I]. The released bridge is not the code that produced the
  results (calls an undefined function) [I].
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
| DeepMPC, Lenz et al., RSS 2015 | Position; action is EE force | Hard input bound 30 N per axis [T, spot-checked] | 1488 cuts / 20 materials; 1.78 mm error at 0.5 s; no unseen-material robot test, no events |
| Mitsioni et al., Humanoids 2019, [1903.03831](https://arxiv.org/abs/1903.03831) | Position only; force is an input [T] | Only the sampling bound on reference force (8 N) | 5 trials/class, cutting rate; potato unseen |
| Mitsioni et al., CASE 2021, [2003.09179](https://arxiv.org/abs/2003.09179) | Position only (LSTM, horizon curriculum) [T, spot-checked] | Torque-limit shutdowns repeated and discarded; some foods pre-slit | Prediction MSE; mean MPC cost; success = reached board |
| Mitsioni et al., T-RO 2023 | State trajectories; β-VAE safe set | Soft cost on distance to safe set, plus abort | Accessible preprint: pendulum only; the published cutting experiment was **not read** — get the IEEE version |

No code for the cutting papers; SafeDDC (T-RO) code is pendulum only [C].
All are real-robot and argue cutting simulation is unavailable [T].

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

## Assessment

Already done (cannot be claimed): the setting and force-limit collision terminal
(SliceIt); learned cutting dynamics + MPC on many and unseen materials (DeepMPC,
Mitsioni); WM + BPTT + learned cost head + Lagrangian (LAMBDA); force-peak
prediction in a WM (visuotactile lifting).

Not found in any of the above:
1. A world model whose force and event predictions are **consistent with its
   state predictions by construction** (all prior: independent heads, or a
   non-differentiable hand-coded threshold). LAMBDA states the state→cost prior
   as an open question.
2. A **hard terminal** force constraint in world-model policy learning (all prior:
   CMDP budgets or input bounds; Mitsioni discards limit hits).
3. A mechanism for why WM policies violate force budgets despite force heads
   (the lifting paper shows the symptom, no mechanism); our measured BCE gradient
   dominance (RESULTS 2026-11-08) is a candidate mechanism, untested causally.
4. One amortized policy across materials in cutting with violation rate, success
   and time over seeds (SliceIt: one carrot, force only; Mitsioni: per-episode
   planning).

Risks: reads as "SliceIt + SuperMimic" unless the consistency mechanism is
principled and beats a LAMBDA-style Lagrangian and a same-WM MPC; simulation-only
results invite "why not use the differentiable simulator"; the T-RO cutting
experiment is unread.

## Baselines and ablations this implies

- SAC (SliceIt code reward, our multi-material randomization stated as stronger
  than their release).
- Same WM + MPC (CEM/MPPI on GPU) with predicted force event as constraint or
  penalty; optionally a Mitsioni-faithful position-only LSTM + shooting MPC.
- LAMBDA-style augmented-Lagrangian force constraint on our WM (estimated
  ~100–150 LOC).
- Ablations: learned event head vs events derived from predicted state/force;
  with vs without event-loss gradient into the dynamics; fixed −100 vs adaptive λ.
- DiSECt trajectory optimization as a per-object open-loop reference (mean and
  peak force).
- Metrics: WM force-peak error vs a persistence baseline, event Brier/AUROC and
  calibration, success = cut AND max force < 50 N, violation rate, time, seeds.

## Still open

- IEEE version of Mitsioni T-RO 2023 (cutting section).
- Google Scholar "cited by" pass on SliceIt and DiSECt.
- Full reads of LEQ and Segmentation Dreamer appendices.
