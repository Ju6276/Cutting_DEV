# Official SuperMimic on the cutting prism

Reference: [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic) at
`5efbf11` (Apache-2.0), read in full for the learners, world-model env, runner, models and
replay. The project's earlier base (`third_party/supermimic`, "SuperMimic-Reproduced
ba610534") is a third-party reproduction, not this code. The adaptation lives in
`cutting_world_model/official/` and `scripts/train_official_supermimic.py`; tests pin the
official defaults (`tests/test_official_supermimic.py`).

**Rule.** Algorithmic parts follow the official code; only robot/task interfaces (state
representation, prediction integration, reward, termination rules, action processing) are
replaced by the cutting task's. Every replacement is listed below. Re-check this table
whenever the adaptation changes.

## Component mapping

| Official (`src/supermimic/...`) | Adaptation | Same / changed |
|---|---|---|
| `runners/supermimic_runner.run`: collect one step, then `Ratio(num_envs/num_batches/batch_size/horizon*replay_ratio)` updates, each = 1 WM update + 1 policy update; no WM-only stage | `train_official_supermimic.py` | Same (replay_ratio 16; 1024 worlds → 0.1 updates per step) |
| Exploration `actions + N(0, 0.1)`; raw action stored | same | Same |
| `replay_buffer.ReplayBuffer`: pre-step frames, buffer 1000/env, windows of L frames with no done in the first L−1 (terminal post-step states never stored), uniform `sample(1, B)` starts | `official.components.ReplayBuffer` | Same algorithm (dict of tensors instead of TensorDict) |
| Normalizers updated from every collected step (`_flush_normalization`) | same | Same |
| `models.build_mlp`: Linear → LayerNorm → GELU | same | Same |
| `models.Dynamics`: EmpiricalNormalization(state), EmpiricalNormalization(proc(action)), MLP(2048×3) → `representation.apply_prediction` | `official.components.Dynamics` | Changed (task interface): state = control-history state of `ControlStateSpec` (the cut material is not observed, so history stands in for it); `proc` = clamp to SliceIt action bounds (no scale/offset); `apply_prediction` = `spec.integrate_physical` + `spec.derive`; no sim context (fixed task) |
| `learners/wm_learner`: lr 2e-4, 10 × 2048 windows of 8, rollout of L−1 steps, Bryson-weighted state loss only (no termination head), norm clip 1.0, one optimizer step per batch | `official.components.WorldModelLearner` | Same, with loss groups = physical fields and nominal errors = the representation's per-channel `err_scale` |
| `learner_utils.build_optimizers`: Muon (torch 2.10 defaults, `adjust_lr_fn='match_rms_adamw'`, weight decay 0.1) on hidden Linear weights, Adam on the rest | `official.components.build_optimizers` + vendored `official/muon.py` | Same (Muon vendored from torch v2.10.0; Adam not fused/capturable — same update) |
| `models.MultiEncoderAE` (one task): goal normalizer → encoder MLP → tanh latent 32; obs normalizer → decoder MLP (2048,1024,512,256) | `official.components.Actor` | Same; single task so latent-BC is zero; actor obs = `rep_state` (official default), goal = target pose |
| `envs/world_model_env`: starts from replay, reward from predicted state, soft termination `sigmoid((err/threshold − 1)/0.05)` | `official.components.WorldModelEnv` | Changed (task rules): failure rules = force limit (max |peak wrench|/limit) and workspace (max |pose error|/limit); a predicted goal success terminates with probability 1 (cutting ends at the goal; motion tracking has no success termination); reward = cutting task reward |
| `learners/policy_learner.ActorCriticPolicyLearner`: horizon 32, γ 0.99, λ 0.95, batch 4096, lr 2e-4, betas (0.9, 0.95), norm clip 1.0, value clip 0.1, survival masking, bootstrap γ^H·V·survival, critic clamped at v_max = max_step_reward/(1−γ), critic fitted to clamped GAE returns 5 × 4 minibatches | `official.components.ActorCriticPolicyLearner` | Same; max_step_reward = goal bonus 100 (all other cutting reward terms ≤ 0); batch split into chunks with accumulated gradients (memory only; same gradient) |
| Critic input `[rep_state, goal, sim_ctx]`, critic MLP (256,128), no normalizer | same without sim_ctx | Same |
| `torch.compile`, CUDA-graph capture | not used | Performance only |
| Return-based reset sampling over motion bins | not applicable | Fixed task: uniform environment resets |

## Consequences worth watching in this comparison

- **No terminal transitions in WM training.** The force-limit step's post-step state is never
  stored, so the WM is trained only on non-failing transitions; force-limit risk enters
  imagination only through the soft termination of the mean prediction.
- **Imagined-return critic with an analytic cap**, not the project's real-data critic.
- **Fewer updates per transition** than the project's runs (0.1 vs 0.4 per step at 1024 worlds).
