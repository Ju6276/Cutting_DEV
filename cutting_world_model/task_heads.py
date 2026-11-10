"""Torch translations of SliceIt's observation, slicing reward and termination formulas."""
# Adapted from ur_python_utilities, copyright 2018-2021 Cristian Beltran.
# MIT license: third_party/sliceit/LICENSE.
import torch

TASK_INTERFACE_REVISION = "sliceit-177cf21"

# 'sliceit': the source saturating term -sigmoid(s/2 - 3); 'linear': -s / force_penalty_scale.
FORCE_PENALTIES = ("sliceit", "linear")
# Environment keys that select the reward; absent keys mean the unchanged SliceIt reward.
REWARD_KEYS = ("force_penalty", "force_penalty_scale")


def reward_settings(environment):
    """Reward selection of an environment config; empty for the source SliceIt reward."""
    return {key: environment[key] for key in REWARD_KEYS if key in environment}


def without_reward_settings(metadata):
    """Task metadata with the reward selection removed, to compare the physical task alone."""
    environment = {key: value for key, value in metadata["environment"].items()
                   if key not in REWARD_KEYS}
    return {**metadata, "environment": environment}


def pose_error(target, current):
    desired, actual = target[..., 3:], current[..., 3:]
    ne = (actual * desired).sum(dim=-1, keepdim=True)
    orientation = actual[..., 3:] * desired[..., :3] - desired[..., 3:] * actual[..., :3]
    orientation = orientation + torch.linalg.cross(actual[..., :3], desired[..., :3])
    orientation = orientation * torch.sign(ne)
    return torch.cat((target[..., :3] - current[..., :3], orientation), dim=-1)


class SliceItHeads:
    def __init__(self, environment, controller):
        self.env = environment
        self.controller = controller
        self.dims = environment["target_dims"]
        self.history = environment["wrench_hist_size"]
        self.max_distance = [environment["max_distance"][i] for i in self.dims]
        velocities = environment.get("max_velocity", [.5, .5, .5, .785398, .785398, .785398])
        self.max_velocity = [velocities[i] for i in self.dims]
        self.force_limits = controller["max_force_torque"]
        if environment.get("use_dynamic_rewards", False):
            raise ValueError("Dynamic reward weights require the corresponding SliceIt schedule")
        if environment.get("reward_based_on_cl", False):
            raise ValueError("Curriculum reward scaling requires the corresponding SliceIt state")
        self.force_penalty = environment.get("force_penalty", "sliceit")
        if self.force_penalty not in FORCE_PENALTIES:
            raise ValueError("force_penalty must be one of " + ", ".join(FORCE_PENALTIES))
        self.force_penalty_scale = environment.get("force_penalty_scale")
        if self.force_penalty == "linear":
            scale = self.force_penalty_scale
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not scale > 0:
                raise ValueError("The linear force penalty requires a positive force_penalty_scale")
        elif self.force_penalty_scale is not None:
            raise ValueError("force_penalty_scale applies to the linear force penalty only")

    def limits(self, like, limit_scale=None):
        """Force/torque limits, optionally scaled per sample (per-object limits)."""
        limits = like.new_tensor(self.force_limits)
        return limits if limit_scale is None else limits * limit_scale[..., None].to(like)

    def observation(self, pose_error, velocity, jerk, previous_action, wrench_history, limit_scale=None):
        # SliceIt flattens force histories by component, then by time. Forces are normalized
        # by the (per-sample) limits, so the policy observes sensor-limit utilization.
        dims = self.dims
        position = pose_error[..., dims] / pose_error.new_tensor(self.max_distance)
        velocity = velocity[..., dims]
        if self.env["normalize_velocity"]:
            velocity = velocity / velocity.new_tensor(self.max_velocity)
        pieces = [position, velocity]
        if self.env["jerkiness_as_obs"]:
            pieces.append(jerk[..., dims] / 500.)
        if self.env["last_action_as_obs"]:
            pieces.append(previous_action)
        limits = self.limits(wrench_history, limit_scale)
        if limit_scale is not None:
            limits = limits[..., None, :]
        force = wrench_history / limits
        force = force[..., dims].transpose(-1, -2).flatten(-2)
        pieces.append(force)
        return torch.cat(pieces, dim=-1)

    def jerkiness(self, observation):
        """SliceIt's reward jerkiness: norm of the observed jerk (target dims, / 500)."""
        dims = len(self.dims)
        return torch.linalg.vector_norm(observation[..., dims * 2:dims * 3], dim=-1)

    def force_norm(self, observation, limit_scale=None):
        """s = ||mean over the history of the physical target-dim wrench components||."""
        dims = len(self.dims)
        limits = self.limits(observation, limit_scale)[..., self.dims]
        if limit_scale is not None:
            limits = limits[..., None, :]
        # Observations store each component across the history window.
        force = observation[..., -self.history * dims:].reshape(*observation.shape[:-1], dims, self.history)
        force = force.transpose(-1, -2)
        force = (force * limits).mean(dim=-2)
        return torch.linalg.vector_norm(force[..., :3], dim=-1)

    def force_reward(self, observation, limit_scale=None):
        """Unweighted force term of the reward under the configured force penalty."""
        norm_force = self.force_norm(observation, limit_scale)
        if self.force_penalty == "linear":
            return -norm_force / self.force_penalty_scale
        return -torch.sigmoid(norm_force / 2. - 3.)

    def reward_weights(self, observation):
        weights = observation.new_tensor([self.env["w_dist"], self.env["w_force"],
                                          self.env["w_jerkiness"], 1.])
        return weights / weights.abs().sum()

    def weighted_force_reward(self, observation, limit_scale=None):
        """The force term as it enters the reward (with its normalized weight)."""
        return self.reward_weights(observation)[1] * self.force_reward(observation, limit_scale)

    def reward(self, observation, collision, goal_reached, out_of_workspace, limit_scale=None):
        distance = observation[..., 1].abs()
        jerk = self.jerkiness(observation)
        r_distance = -torch.tanh(5. * distance)
        r_force = self.force_reward(observation, limit_scale)
        r_jerk = torch.clamp(-jerk, min=-1., max=0.)
        cost = self.env["cost"]
        weights = self.reward_weights(observation)
        result = weights[0] * r_distance + weights[1] * r_force + weights[2] * r_jerk + weights[3] * cost["step"]
        result = result + collision.to(result.dtype) * cost["collision"]
        result = result + (goal_reached & ~collision).to(result.dtype) * cost["done"]
        return result + out_of_workspace.to(result.dtype) * cost["collision"]

    def termination(self, observation, success_counter, collision):
        error = (observation[..., :len(self.dims)] * observation.new_tensor(self.max_distance)).abs()
        position_reached = (error < error.new_tensor(self.env["goal_threshold"])).all(dim=-1)
        success_counter = success_counter + position_reached.to(success_counter.dtype)
        goal_reached = (position_reached & (success_counter > self.env["successes_threshold"])
                        & ~collision & (not self.env["termination_on_negative_reward"]))
        out_of_workspace = (error > error.new_tensor(self.env["workspace_limit"])).any(dim=-1)
        done = goal_reached | collision | out_of_workspace | self.env["termination_on_negative_reward"]
        return done, goal_reached, out_of_workspace, success_counter
