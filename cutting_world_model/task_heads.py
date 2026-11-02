"""Torch translations of SliceIt's observation, slicing reward and termination formulas."""
# Adapted from ur_python_utilities, copyright 2018-2021 Cristian Beltran.
# MIT license: third_party/sliceit/LICENSE.
import torch


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

    def observation(self, pose_error, velocity, jerk, previous_action, wrench_history):
        # SliceIt flattens force histories by component, then by time.
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
        force = wrench_history / wrench_history.new_tensor(self.force_limits)
        force = force[..., dims].transpose(-1, -2).flatten(-2)
        pieces.append(force)
        return torch.cat(pieces, dim=-1)

    def reward(self, observation, collision, goal_reached, out_of_workspace):
        dims = len(self.dims)
        distance = observation[..., 1].abs()
        jerk = torch.linalg.vector_norm(observation[..., dims * 2:dims * 3], dim=-1)
        limits = observation.new_tensor([self.force_limits[i] for i in self.dims])
        # Preserve cost_utils.slicing's reshape, including its component ordering.
        force = observation[..., -self.history * dims:].reshape(*observation.shape[:-1], -1, dims)
        force = (force * limits).mean(dim=-2)
        norm_force = torch.linalg.vector_norm(force[..., :3], dim=-1)
        r_distance = -torch.tanh(5. * distance)
        r_force = -torch.sigmoid(norm_force / 2. - 3.)
        r_jerk = torch.clamp(-jerk, min=-1., max=0.)
        cost = self.env["cost"]
        weights = observation.new_tensor([self.env["w_dist"], self.env["w_force"],
                                          self.env["w_jerkiness"], 1.])
        weights = weights / weights.abs().sum()
        result = weights[0] * r_distance + weights[1] * r_force + weights[2] * r_jerk + weights[3] * cost["step"]
        result = result + collision.to(result.dtype) * cost["collision"]
        result = result + (goal_reached & ~collision).to(result.dtype) * cost["done"]
        return result + out_of_workspace.to(result.dtype) * cost["collision"]

    def termination(self, observation, success_counter, collision):
        error = (observation[..., :len(self.dims)] * observation.new_tensor(self.max_distance)).abs()
        position_reached = (error < error.new_tensor(self.env["goal_threshold"])).all(dim=-1)
        success_counter = success_counter + position_reached.to(success_counter.dtype)
        goal_reached = position_reached & (success_counter > self.env["successes_threshold"])
        out_of_workspace = (error > error.new_tensor(self.env["workspace_limit"])).any(dim=-1)
        done = goal_reached | collision | out_of_workspace | self.env["termination_on_negative_reward"]
        return done, goal_reached, out_of_workspace, success_counter
