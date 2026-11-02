"""Pack recorded fields while retaining their layout and transition alignment."""
import numpy as np

FIELDS = ("particle_q", "particle_qd", "cut_spring_ke", "knife_pose", "knife_velocity", "force")
COUPLED_FIELDS = ("observation", "joint_q", "joint_qd", "ee_pose", "wrench_history",
                  "motion_command", "previous_ee_pose", "previous_velocity",
                  "previous_acceleration", "target_pose", "success_counter", "step_count")


def pack_trajectory(path):
    with np.load(path) as data:
        count = len(data["time"])
        fields = FIELDS + COUPLED_FIELDS if "observation" in data else FIELDS
        arrays = [data[name].reshape(count, -1) for name in fields]
        state = np.concatenate(arrays, axis=1).astype(np.float32)
        actions = data["actions"].astype(np.float32)
        if actions.shape != (count - 1, 6):
            raise ValueError("Each trajectory must have T actions and T+1 states")
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError("Trajectory contains non-finite values")
        layout = {}
        offset = 0
        for name, array in zip(fields, arrays):
            layout[name] = (offset, offset + array.shape[1])
            offset += array.shape[1]
        return state, actions, layout
