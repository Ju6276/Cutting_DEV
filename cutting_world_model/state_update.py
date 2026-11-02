"""Known state updates from SliceIt and rotation/damage structure from the source models."""
import torch


def compose_rotation(quaternion, rotation_vector):
    """Quaternions use xyzw; the increment is expressed in the body's frame."""
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    vector = .5 * torch.sinc(angle / (2 * torch.pi)) * rotation_vector
    scalar = torch.cos(.5 * angle)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    q_vector, q_scalar = quaternion[..., :3], quaternion[..., 3:]
    result_vector = q_scalar * vector + scalar * q_vector + torch.linalg.cross(q_vector, vector)
    result_scalar = q_scalar * scalar - (q_vector * vector).sum(dim=-1, keepdim=True)
    return torch.cat((result_vector, result_scalar), dim=-1)


def structured_update(state, proposal, action, layout, action_increment):
    parts = {}
    for name, (begin, end) in layout.items():
        previous = state[..., begin:end]
        candidate = proposal[..., begin:end]
        if name == "target_pose":
            candidate = previous
        elif name == "step_count":
            candidate = previous + 1
        elif name == "motion_command":
            delta_x = action[..., 4:5] * action_increment[0]
            delta_yaw = -action[..., 5:6] * action_increment[1]
            zero = torch.zeros_like(delta_x)
            candidate = previous + torch.cat((delta_x, zero, zero, zero, delta_yaw, zero), dim=-1)
        elif name == "observation":
            candidate = torch.cat((candidate[..., :9], action, candidate[..., 15:]), dim=-1)
        elif name == "cut_spring_ke":
            decrement = torch.clamp(previous - candidate, min=0.)
            candidate = torch.clamp(previous - decrement, min=0.)
        elif name == "force":
            candidate = torch.clamp(candidate, min=0.)
        elif name in ("knife_pose", "ee_pose", "previous_ee_pose"):
            rotation = 2 * (candidate[..., 3:6] - previous[..., 3:6])
            quaternion = compose_rotation(previous[..., 3:], rotation)
            candidate = torch.cat((candidate[..., :3], quaternion), dim=-1)
        parts[name] = candidate
    return torch.cat(list(parts.values()), dim=-1)
