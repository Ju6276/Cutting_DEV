"""Import complete recorded episodes into the upstream episode-aware ring buffer."""
import numpy as np
import torch

from .recordings import pack_trajectory
from third_party.supermimic.replay_buffer import ReplayBuffer


def replay_from_episodes(paths, device="cpu"):
    packed = [pack_trajectory(path) for path in paths]
    layout = packed[0][2]
    if any(episode[2] != layout for episode in packed):
        raise ValueError("Recorded state layouts differ")
    buffer = ReplayBuffer(1, 1000, packed[0][0].shape[1], 6, 0, device)
    env_id = torch.tensor([0], device=device)
    for episode_id, (path, (states, actions, _)) in enumerate(zip(paths, packed)):
        with np.load(path) as data:
            rewards = data["rewards"]
            terminated = data["terminated"]
            truncated = data["truncated"]
            clean_actions = data["clean_actions"] if "clean_actions" in data else actions
        if np.any(terminated[:-1]) or np.any(truncated[:-1]):
            raise ValueError("Recorded episode crosses a reset")
        if not (terminated[-1] or truncated[-1]):
            raise ValueError("Expected a completed episode")
        for frame in range(len(states)):
            terminal_row = frame == len(actions)
            executed = np.zeros(6, dtype=np.float32) if terminal_row else actions[frame]
            clean = np.zeros(6, dtype=np.float32) if terminal_row else clean_actions[frame]
            buffer.add(env_id,
                torch.as_tensor(states[frame:frame + 1], device=device),
                torch.as_tensor(clean[None], device=device),
                torch.as_tensor(executed[None], device=device),
                torch.empty((1, 0), device=device),
                torch.tensor([0. if terminal_row else rewards[frame]], dtype=torch.float32, device=device),
                torch.tensor([bool(terminated[frame - 1]) if frame else False], dtype=torch.float32, device=device),
                torch.tensor([frame], device=device), torch.tensor([episode_id], device=device),
                torch.tensor([terminal_row], device=device), torch.tensor([frame == 0], device=device))
        buffer.end_episode(env_id)
    return buffer, layout
