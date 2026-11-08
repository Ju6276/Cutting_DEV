# Novelty check — force-aware world model for force-limited cutting

Status: first pass, 2026-11-08. Framing under test (user, 2026-11-08): a reliable
force-aware world model (state, force and cutting-event predictions consistent)
with BPTT policy learning; multi-material evaluation shows that one policy cuts
several objects under the 50 N limit. Robot-methods paper (RA-L/ICRA/IROS).

Evidence labels: **T** read in the paper text; **A** abstract or listing page only;
**S** search snippet or secondary page; **M** from memory, unchecked. Most entries
are A/S: nothing below may be cited as a claim about a paper before a full read.

## Closest prior work

| Work | Does | Does not (relative to our framing) | Label |
|---|---|---|---|
| SliceIt, ICRA 2024, [2404.02569](https://arxiv.org/abs/2404.02569) | UR5e + FDCC, DiSECt + Gazebo; SAC outputs FDCC gains and reference; F/T history encoder; +100 completion, −100 force-limit collision; tomato/cucumber/potato with parameter noise, carrot held out (5 slices) | No learned dynamics or world model | T |
| DeepMPC, Lenz et al., RSS 2015, [RSS](https://www.roboticsproceedings.org/rss11/p12.html) | Recurrent latent dynamics for real-time MPC on PR2; 1488 cuts, 20 materials | Policy learning, force limit, event prediction | A |
| Mitsioni et al., Humanoids 2019, [1903.03831](https://arxiv.org/abs/1903.03831) | RNN dynamics with F/T in MPC; training and unknown objects | Policy learning, completion/terminal prediction, hard force limit | A |
| Mitsioni et al., CASE 2021, [2003.09179](https://arxiv.org/abs/2003.09179) | LSTM multi-step cutting dynamics with horizon curriculum, in MPC; unseen object classes | (to read) | S |
| Mitsioni et al., T-RO 2023, DOI 10.1109/TRO.2023.3266995 | Learned safe-state sets from safe demonstrations, distance-to-safe-set in MPC cost; real food cutting | (to read) — closest to "reliable under a constraint with a learned model" | S |
| Rezaei-Shoshtari et al., [2007.11167](https://arxiv.org/abs/2007.11167) | Latent space infers cutting/not cutting, material, geometry | Control | S |
| Luo & Yuan, [2607.04367](https://arxiv.org/abs/2607.04367) | Knife choice from trial-cut force; RL tunes target force and sawing speed | World model; peak force in reward | A |
| RoboNinja, [2302.11553](https://arxiv.org/abs/2302.11553) | Multi-material (rigid core), differentiable MPM demonstrations, core estimation from collisions | World model, force-magnitude limit | A |
| DexNinja, [2610.17208](https://arxiv.org/abs/2610.17208); TopoCut, [2509.19712](https://arxiv.org/abs/2509.19712) | Imitation / diffusion policies on differentiable or MPM simulators | World model, force limit | A |
| Zhang et al., [1909.12460](https://arxiv.org/abs/1909.12460) | Vibration + F/T networks detect contact events, adapt slicing, >20 foods | World model, MBRL | A |
| Kuo et al., RA-L 2021, [2010.08169](https://arxiv.org/abs/2010.08169) | GP probabilistic MPC, acceleration limits shrink with uncertainty | Force prediction or bound; policy learning | A |
| SafeDreamer, [2307.07176](https://arxiv.org/abs/2307.07176); LAMBDA, [2201.09802](https://arxiv.org/abs/2201.09802) | Constrained world-model RL with cost models (Safety-Gym) | Contact force, physical event consistency | A |
| Visuotactile WM for lifting, [2609.09597](https://arxiv.org/abs/2609.09597) | WM predicts force, force-budgeted success; imagination RL underperforms reactive IQL (11.9% vs 25.0%) | — evidence that imagined and real returns diverge | A |
| ForceLDM [2611.03294](https://arxiv.org/abs/2611.03294), FAWAM, CW-WAM, TacForeSight | Force-aware world/latent models | Imitation/VLA; no BPTT, event consistency or hard limit | A/S |
| DreamerV3, [2301.04104](https://arxiv.org/abs/2301.04104) | Continue head trained jointly with dynamics; stop-gradient ablations on reward/value heads | Continue-head gradient balance | M/S |
| Task-relevant reconstruction, [2410.09972](https://arxiv.org/abs/2410.09972) | Stopping a mask head's gradient into the WM helped | — closest analogue of the BCE-gradient repair | S |

## Assessment (first pass)

Already done: the setting itself (SliceIt); learned cutting dynamics + MPC with
unseen objects (DeepMPC, Mitsioni 2019/2021); safe learned-model MPC for cutting
(Mitsioni T-RO 2023); constrained world-model RL in general (SafeDreamer, LAMBDA).

Not found (searches, not a full citation crawl):
1. A world model trained with a BPTT policy for force-limited cutting.
2. Consistency of force-peak and cut-event predictions with state predictions,
   including the termination-loss vs dynamics gradient imbalance measured here
   (RESULTS, 2026-11-08).
3. One policy over several geometries and randomized materials, judged on
   reliability under a hard force limit, against SAC, learned-model MPC and
   DiSECt trajectory optimization.

Risks:
- The paper reads as "SliceIt plus a SuperMimic-style world model". A standard
  trick (stop-gradient, reweighting, separate event network) will look like
  engineering; the method needs a principled mechanism and clear gains over SAC
  (SliceIt) and Mitsioni-style learned-model MPC under the force limit.
- DiSECt is differentiable; without real-robot results reviewers may ask why
  the simulator is not used directly.

## Must read in full before any novelty claim

1. SliceIt (2404.02569) — reward, force threshold, results.
2. Mitsioni et al. T-RO 2023 — safe learned-model MPC for cutting.
3. Mitsioni et al. CASE 2021 (2003.09179).
4. SuperMimic (2610.28480) — critic and termination handling.
5. DreamerV3 (2301.04104) — continue head, gradient stopping.
6. SafeDreamer (2307.07176) and LAMBDA (2201.09802).
7. Visuotactile WM for lifting (2609.09597).
8. DiSECt (2203.10263) — trajectory-optimization baseline.

Still open: Google Scholar "cited by" pass on SliceIt and DiSECt.
