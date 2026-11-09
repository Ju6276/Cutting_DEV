"""Checkpoint identity and exact weight pairing for frozen evaluations."""
import torch


def checkpoint_identity(saved):
    learner = saved["learner"]
    stage = learner.get("training_stage", {})
    variant = learner.get("critic_variant", {})
    return {
        "iteration": saved.get("iteration"),
        "policy_updates": learner["updates"],
        "wm_updates": stage.get("wm_updates"),
        "optimizer_steps": stage.get("optimizer_steps", {}),
        "critic": {
            "target": variant.get("target", "imagined"),
            "reset_update": variant.get("reset_update"),
            "prefit_rounds": variant.get("prefit_rounds", 0),
            "prefit_pending": variant.get("prefit_pending", False),
        },
        "actor_horizon": learner.get(
            "actor_horizon", stage.get("policy_config", {}).get("horizon")),
        "source_git": saved.get("git"),
    }


def check_checkpoint_pair(planner, paired, use_critic=False):
    """Compare loaded tensors; filenames and update counters do not identify weights."""
    if planner["metadata"] != paired["metadata"]:
        raise ValueError("Paired checkpoint task metadata differ")
    groups = ("world_model", "policy") + (("critic",) if use_critic else ())
    for group in groups:
        left = planner["learner"].get(group)
        right = paired["learner"].get(group)
        if not left or not right:
            raise ValueError(f"Paired checkpoint is missing {group} weights")
        if left.keys() != right.keys():
            raise ValueError(f"Paired checkpoint {group} parameter names differ")
        for name, value in left.items():
            other = right[name]
            if value.dtype != other.dtype or not torch.equal(value, other):
                raise ValueError(f"Paired checkpoint {group} differs at {name}")
    return list(groups)
