from copy import deepcopy

import pytest
import torch

from cutting_world_model.policy import SliceItCritic
from cutting_world_model.trainer import OnlineLearner
from cutting_world_model.world_model import CuttingWorldModel
from third_party.supermimic.dynamics import MLP
from third_party.supermimic.trainer_config import PolicyTrainerCfg, WMTrainerCfg
from test_online_training import replay_fixture
from test_state_spec import fixture


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP(24 + 7, (16,), 6, last_gain=.01)

    def forward(self, history, goal):
        return self.net(torch.cat((history[-1], goal), -1))


def make_learner(wm_only_updates=500):
    torch.set_num_threads(2)
    torch.manual_seed(81)
    metadata, frame = fixture()
    model = CuttingWorldModel(metadata, hidden=(16, 16))
    replay = replay_fixture(model, frame)
    learner = OnlineLearner(
        model, TinyPolicy(), SliceItCritic(model.spec.state_dim), microbatch=2,
        wm_only_updates=wm_only_updates,
        wm_cfg=WMTrainerCfg(compile=None, window=3, batch_size=4, batches_per_update=2),
        policy_cfg=PolicyTrainerCfg(compile=None, horizon=3, batch_size=4,
                                   critic_epochs=1, critic_minibatches=2))
    return learner, replay


def weights(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def assert_unchanged(module, before):
    for actual, expected in zip(module.parameters(), before):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def assert_changed(module, before):
    assert any(not torch.equal(actual, expected) for actual, expected in zip(module.parameters(), before))


def test_two_world_model_updates_then_joint_update_changes_only_allowed_weights():
    learner, replay = make_learner(wm_only_updates=2)
    before_policy, before_critic = weights(learner.policy), weights(learner.critic)
    assert learner.optimizer_steps == {'world_model': 0, 'policy': 0, 'critic': 0}
    for update in (1, 2):
        before_model = weights(learner.wm)
        learner.fit_world_model(replay)
        assert_changed(learner.wm, before_model)
        assert_unchanged(learner.policy, before_policy)
        assert_unchanged(learner.critic, before_critic)
        assert learner.wm_updates == update and learner.updates == 0
        assert learner.stage == 'world_model_only'
        assert not learner.should_update_policy
        assert learner.optimizer_steps == {'world_model': update * 2, 'policy': 0, 'critic': 0}
        assert not learner.popt.state and not learner.copt.state
        with pytest.raises(RuntimeError, match='world-model-only'):
            learner.improve_policy(replay)
    learner.fit_world_model(replay)
    assert learner.stage == 'joint' and learner.should_update_policy
    learner.improve_policy(replay)
    assert_changed(learner.policy, before_policy)
    assert_changed(learner.critic, before_critic)
    assert learner.wm_updates == 3 and learner.updates == 1
    assert learner.optimizer_steps == {'world_model': 6, 'policy': 1, 'critic': 2}


def test_checkpoint_preserves_both_sides_of_world_model_stage_boundary():
    learner, replay = make_learner(wm_only_updates=2)
    for count in (1, 2):
        learner.fit_world_model(replay)
        resumed, _ = make_learner(wm_only_updates=2)
        resumed.restore(deepcopy(learner.checkpoint()))
        assert resumed.wm_updates == count and resumed.updates == 0
        assert not resumed.should_update_policy
        assert resumed.optimizer_steps == learner.optimizer_steps
        assert_unchanged(resumed.wm, weights(learner.wm))
        learner = resumed
    learner.fit_world_model(replay)
    joint, _ = make_learner(wm_only_updates=2)
    joint.restore(deepcopy(learner.checkpoint()))
    assert joint.wm_updates == 3 and joint.should_update_policy
    joint.improve_policy(replay)
    resumed, _ = make_learner(wm_only_updates=2)
    resumed.restore(deepcopy(joint.checkpoint()))
    assert resumed.wm_updates == 3 and resumed.updates == 1
    assert resumed.stage == 'joint'
    assert resumed.optimizer_steps == joint.optimizer_steps
    assert_unchanged(resumed.policy, weights(joint.policy))
    assert_unchanged(resumed.critic, weights(joint.critic))


def test_resume_rejects_critic_trained_with_previous_terminal_rule():
    learner, _ = make_learner()
    saved = learner.checkpoint()
    saved.pop('imagination_terminal_rule')
    with pytest.raises(ValueError, match='imagination terminal rule'):
        learner.restore(saved)


def test_interrupted_world_model_block_records_optimizer_steps_without_completed_credit():
    learner, replay = make_learner(wm_only_updates=2)

    def interrupt_second_batch(**progress):
        if progress['minibatch'] == 2:
            raise InterruptedError('interrupted after one optimizer step')

    learner.progress = interrupt_second_batch
    with pytest.raises(InterruptedError):
        learner.fit_world_model(replay)
    assert learner.wm_updates == learner.updates == 0
    assert learner.optimizer_steps == {'world_model': 1, 'policy': 0, 'critic': 0}
    resumed, _ = make_learner(wm_only_updates=2)
    resumed.restore(deepcopy(learner.checkpoint()))
    assert resumed.wm_updates == 0 and not resumed.should_update_policy
    assert resumed.optimizer_steps == learner.optimizer_steps
    resumed.fit_world_model(replay)
    assert resumed.wm_updates == 1
    assert resumed.optimizer_steps == {'world_model': 3, 'policy': 0, 'critic': 0}


def test_default_stage_boundary_and_explicit_original_update_mode():
    learner, _ = make_learner()
    assert learner.wm_only_updates == 500 and not learner.should_update_policy
    learner.wm_updates = 500
    assert not learner.should_update_policy
    learner.wm_updates = 501
    assert learner.should_update_policy
    original, replay = make_learner(wm_only_updates=0)
    assert original.stage == 'joint'
    original.improve_policy(replay)
    assert original.wm_updates == 0 and original.updates == 1


@pytest.mark.parametrize('count', [-1, 1.5, True, None])
def test_invalid_world_model_stage_configuration_is_rejected(count):
    with pytest.raises(ValueError, match='nonnegative integer'):
        make_learner(wm_only_updates=count)


@pytest.mark.parametrize('change, message', [
    ('missing', 'lacks training-stage'),
    ('duration', 'world-model-only configuration differs'),
    ('negative', 'wm_updates must be'),
    ('trainer', 'trainer configuration differs'),
    ('optimizer', 'optimizer step count differs'),
    ('early_policy', 'completed policy updates exceed'),
    ('inflated_world_model', 'completed world-model updates exceed'),
])
def test_incompatible_stage_checkpoint_rejected_before_loading_weights(change, message):
    learner, replay = make_learner(wm_only_updates=2)
    learner.fit_world_model(replay)
    checkpoint = deepcopy(learner.checkpoint())
    if change == 'missing':
        checkpoint.pop('training_stage')
    elif change == 'duration':
        checkpoint['training_stage']['wm_only_updates'] = 1
    elif change == 'negative':
        checkpoint['training_stage']['wm_updates'] = -1
    elif change == 'trainer':
        checkpoint['training_stage']['world_model_config']['batches_per_update'] = 10
    elif change == 'optimizer':
        checkpoint['training_stage']['optimizer_steps']['world_model'] += 1
    elif change == 'early_policy':
        checkpoint['updates'] = 1
    elif change == 'inflated_world_model':
        checkpoint['training_stage']['wm_updates'] = 501
    resumed, _ = make_learner(wm_only_updates=2)
    before = weights(resumed.wm)
    with pytest.raises(ValueError, match=message):
        resumed.restore(checkpoint)
    assert_unchanged(resumed.wm, before)
    assert resumed.wm_updates == resumed.updates == 0
    assert not resumed.wopt.state


@pytest.mark.parametrize('change, message', [
    ('policy', 'completed policy updates exceed'),
    ('critic', 'completed critic updates exceed'),
    ('early_optimizer', 'optimizer steps precede the joint'),
])
def test_completed_and_partial_policy_updates_obey_recorded_optimizer_steps(change, message):
    learner, replay = make_learner(wm_only_updates=2)
    for _ in range(3):
        learner.fit_world_model(replay)
    learner.improve_policy(replay)
    checkpoint = deepcopy(learner.checkpoint())
    if change == 'policy':
        checkpoint['updates'] += 1
    elif change == 'critic':
        for state in checkpoint['critic_optimizer']['state'].values():
            state['step'].fill_(1)
        checkpoint['training_stage']['optimizer_steps']['critic'] = 1
    elif change == 'early_optimizer':
        checkpoint['updates'] = 0
        checkpoint['training_stage']['wm_updates'] = 2
    resumed, _ = make_learner(wm_only_updates=2)
    with pytest.raises(ValueError, match=message):
        resumed.restore(checkpoint)


def test_partial_joint_update_remains_resumable_without_complete_update_credit():
    learner, replay = make_learner(wm_only_updates=2)
    for _ in range(3):
        learner.fit_world_model(replay)

    def interrupt_critic(**progress):
        if progress['phase'] == 'critic':
            raise InterruptedError('interrupted after policy optimizer step')

    learner.progress = interrupt_critic
    with pytest.raises(InterruptedError):
        learner.improve_policy(replay)
    assert learner.updates == 0
    assert learner.optimizer_steps == {'world_model': 6, 'policy': 1, 'critic': 0}
    resumed, _ = make_learner(wm_only_updates=2)
    resumed.restore(deepcopy(learner.checkpoint()))
    assert resumed.should_update_policy and resumed.updates == 0
    assert resumed.optimizer_steps == learner.optimizer_steps
