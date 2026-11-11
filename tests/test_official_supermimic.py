"""Pins the official SuperMimic (HybridRobotics/supermimic @ 5efbf11) defaults and semantics
kept by the cutting adaptation; see docs/OFFICIAL_SUPERMIMIC.md."""
from dataclasses import asdict

import torch
from torch import nn

from cutting_world_model.official import components as c


def test_official_defaults():
    assert asdict(c.WorldModelLearnerCfg()) == {'lr': 2e-4, 'num_batches': 10, 'batch_size': 2048, 'horizon': 8}
    assert asdict(c.ActorCriticPolicyLearnerCfg()) == {
        'lr': 2e-4, 'num_batches': 1, 'batch_size': 4096, 'noise': 0.1, 'horizon': 32, 'gamma': 0.99,
        'lam': 0.95, 'survival_masking': True, 'beta2': 0.95, 'grad_clip_value': 0.1}
    r = c.RunnerCfg()
    assert (r.buffer_size, r.actor_obs, r.history_length, r.termination_softness, r.replay_ratio,
            r.latent_dim) == (1000, 'rep_state', 1, 0.05, 16, 32)
    assert (r.actor_hidden_dims, r.critic_hidden_dims, r.dynamics_hidden_dims) == (
        (2048, 1024, 512, 256), (256, 128), (2048, 2048, 2048))
    assert c.OFFICIAL_COMMIT == '5efbf11'


def test_mlp_and_optimizer_split():
    mlp = c.build_mlp(5, (8, 8), 2)
    assert [type(m) for m in mlp] == [nn.Linear, nn.LayerNorm, nn.GELU, nn.Linear, nn.LayerNorm, nn.GELU, nn.Linear]
    adam, muon = c.build_optimizers([mlp], 2e-4)
    assert len(muon.param_groups[0]['params']) == 1 and muon.param_groups[0]['params'][0] is mlp[3].weight
    assert muon.defaults['weight_decay'] == 0.1 and muon.defaults['adjust_lr_fn'] == 'match_rms_adamw'
    assert muon.defaults['momentum'] == 0.95 and muon.defaults['nesterov']


def test_ratio_matches_official():
    ratio = c.Ratio(1024 / 10 / 2048 / 8 * 16)
    counts = [ratio(step) for step in range(100)]
    assert counts[0] == 1 and sum(counts) == 1 + 9


def test_replay_windows_never_cross_a_done():
    schema = {'x': ((), torch.long), 'terminated': ((), torch.bool), 'truncated': ((), torch.bool)}
    replay = c.ReplayBuffer(schema, 20, 1, 'cpu')
    for t in range(20):
        replay.insert({'x': torch.tensor([t]), 'terminated': torch.tensor([t == 5]), 'truncated': torch.tensor([False])})
    for _ in range(50):
        window = replay.sample(4, 32)['x']
        # frames 0..2 of a window must not contain the done frame (5) before the end
        assert not ((window[:, :3] == 5).any(dim=1) & (window[:, 1:] > 5).any(dim=1)).any()
        assert ((window[:, 1:] - window[:, :-1]) == 1).all()


def test_gae_soft_done():
    values = torch.tensor([[1.], [2.]])
    returns = c.gae_returns(values, torch.tensor([3.]), torch.tensor([[1.], [1.]]), torch.tensor([[0.], [1.]]), .9, .5)
    # last step: done=1 removes the bootstrap -> return = reward
    assert torch.allclose(returns[1], torch.tensor([1.]))
