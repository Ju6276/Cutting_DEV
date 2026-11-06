"""Execute batched physical transitions for the external manipulation learner."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cutting_world_model.sac_gpu_bridge import serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-root', type=Path, required=True)
    parser.add_argument('--worlds', type=int, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--protocol-fd', type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.backend_root))
    import torch
    import warp as wp
    from cutting_world_model.control_state import ControlStateSpec
    from cutting_world_model.benchmark import physical_contract
    from cutting_world_model.gpu_env import GPUEnvironment

    torch.set_num_threads(2)
    env = GPUEnvironment(args.backend_root, args.worlds, scene=args.scene, capture=args.capture)
    scene = torch.load(args.backend_root / args.scene, map_location='cpu', weights_only=False)
    env.metadata['material_settings'] = dict(scene['settings'])
    env.metadata['controller_parameters'] = asdict(env.physics.config)
    env.metadata['material_particles'] = env.physics.material.particles
    env.metadata['material_springs'] = env.physics.material.springs
    env.metadata['benchmark_contract'] = physical_contract(
        args.backend_root, env, args.scene, args.capture, 'public-apple-ur5e-v1')
    one = {key: value[0].cpu().numpy() for key, value in env.current.items()}
    metadata = ControlStateSpec.metadata_from_frame(env.metadata, one)
    keys = tuple(metadata['state_shapes']) + ('step_count', 'target_pose')
    returns = torch.zeros(args.worlds, device=env.device)

    def pack(frame):
        return {key: frame[key].cpu().tolist() for key in keys}

    def overflow():
        return wp.to_torch(env.physics.robot.data.overflow).cpu().tolist()

    def handle(message):
        operation = message['op']
        if operation == 'init':
            return dict(metadata=env.metadata, model_metadata=metadata,
                        frame=pack(env.current), overflow=overflow())
        if operation == 'reset':
            returns.zero_()
            return dict(frame=pack(env.reset()), overflow=overflow())
        if operation == 'step':
            action = torch.tensor(message['action'], device=env.device, dtype=torch.float32)
            if action.shape != (args.worlds, 6) or not torch.isfinite(action).all():
                raise ValueError('Expected finite batched six-component actions')
            frame, reward, terminated, truncated, info = env.step(action)
            returns.add_(reward)
            done = terminated | truncated
            response = dict(post=pack(frame), reward=reward.cpu().tolist(),
                            terminated=terminated.cpu().tolist(), truncated=truncated.cpu().tolist(),
                            overflow=overflow(), episode_metrics={})
            response['material_force_max'] = float(info['peak_force'].max())
            if done.any():
                response['episode_metrics'] = {
                    'Episode_Metrics/type_0_episodes': int(done.sum()),
                    'Episode_Metrics/type_0_successes': int((info['success'] & done).sum()),
                    'Episode_Metrics/return_mean': float(returns[done].mean()),
                }
                frame = env.reset(done)
                returns[done] = 0
            response['frame'] = pack(frame)
            return response
        if operation == 'close':
            return {}
        raise ValueError(f'Unknown physical operation: {operation}')

    serve(args.protocol_fd, handle)


if __name__ == '__main__':
    main()
