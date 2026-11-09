import copy

import pytest
import torch

from cutting_world_model.checkpoint import checkpoint_identity, check_checkpoint_pair


def snapshot():
    return {
        "metadata": {"task": "prism"},
        "iteration": 1561,
        "learner": {
            "updates": 62,
            "training_stage": {"wm_updates": 562, "policy_config": {"horizon": 32}},
            "world_model": {"weight": torch.tensor([1., 2.])},
            "policy": {"weight": torch.tensor([3., 4.])},
            "critic": {"weight": torch.tensor([5., 6.])},
        },
    }


def test_identity_separates_policy_wm_and_critic_training():
    saved = snapshot()
    assert checkpoint_identity(saved)["critic"]["target"] == "imagined"
    saved["learner"]["training_stage"]["wm_updates"] = 563
    saved["learner"]["critic_variant"] = {
        "target": "real", "reset_update": 62, "prefit_rounds": 100, "prefit_pending": False}
    saved["learner"]["actor_horizon"] = 8
    identity = checkpoint_identity(saved)
    assert identity["policy_updates"] == 62 and identity["wm_updates"] == 563
    assert identity["critic"]["target"] == "real"
    assert identity["critic"]["prefit_rounds"] == 100
    assert identity["actor_horizon"] == 8


@pytest.mark.parametrize("group", ["world_model", "policy", "critic"])
def test_equal_counters_do_not_hide_changed_weights(group):
    original = snapshot()
    changed = copy.deepcopy(original)
    changed["learner"][group]["weight"][0] += .001
    with pytest.raises(ValueError, match=group + " differs"):
        check_checkpoint_pair(changed, original, use_critic=True)


def test_unused_critic_can_differ_but_wm_and_policy_must_match():
    original = snapshot()
    changed = copy.deepcopy(original)
    changed["learner"]["critic"]["weight"] *= 2
    assert check_checkpoint_pair(changed, original) == ["world_model", "policy"]
    with pytest.raises(ValueError, match="critic differs"):
        check_checkpoint_pair(changed, original, use_critic=True)


def test_pair_rejects_different_task_and_missing_weights():
    original = snapshot()
    changed = copy.deepcopy(original)
    changed["metadata"]["task"] = "apple"
    with pytest.raises(ValueError, match="metadata"):
        check_checkpoint_pair(changed, original)
    changed = copy.deepcopy(original)
    del changed["learner"]["world_model"]
    with pytest.raises(ValueError, match="missing world_model"):
        check_checkpoint_pair(changed, original)
