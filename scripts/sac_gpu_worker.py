"""Serve a single unchanged GPUEnvironment to the source SAC process."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cutting_world_model.sac_gpu_bridge import serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-root", type=Path, required=True)
    parser.add_argument("--protocol-fd", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.backend_root))
    import torch
    from cutting_world_model.gpu_env import GPUEnvironment

    torch.set_num_threads(2)
    environment = GPUEnvironment(args.backend_root, worlds=1)

    def pack(frame):
        return {
            "observation": frame["observation"][0].float().cpu().tolist(),
            "sim_time": frame["sim_time"][0, 0].item(),
            "step_count": frame["step_count"][0, 0].item(),
            "wrench_history": frame["wrench_history"][0].cpu().tolist(),
            "joint_q": frame["joint_q"][0].cpu().tolist(),
            "ee_pose": frame["ee_pose"][0].cpu().tolist(),
        }

    def handle(message):
        operation = message["op"]
        if operation == "init":
            return dict(metadata=environment.metadata, **pack(environment.current))
        if operation == "reset":
            return pack(environment.reset())
        if operation == "step":
            action = torch.tensor([message["action"]], device=environment.device, dtype=torch.float32)
            if action.shape != (1, 6) or not torch.isfinite(action).all():
                raise ValueError("Invalid SAC action")
            frame, reward, terminated, truncated, info = environment.step(action)
            return dict(
                **pack(frame), reward=reward[0].item(), terminated=terminated[0].item(),
                truncated=truncated[0].item(),
                info={key: value[0].item() for key, value in info.items()},
            )
        if operation == "close":
            return {}
        raise ValueError("Unknown GPU operation: " + operation)

    serve(args.protocol_fd, handle)


if __name__ == "__main__":
    main()
