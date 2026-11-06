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
    parser.add_argument('--reset-bank', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--protocol-fd', type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.backend_root))
    import torch
    import warp as wp
    from cutting_world_model.control_state import ControlStateSpec
    from cutting_world_model.benchmark import physical_contract
    from cutting_world_model.gpu_env import GPUEnvironment
    from cutting_world_model.reset_bank import CuttingResetBank, STAGES, restore_snapshot

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
    spec = ControlStateSpec(metadata).to(env.device)
    keys = tuple(metadata['state_shapes']) + ('step_count', 'target_pose')
    returns = torch.zeros(args.worlds, device=env.device)
    bank = (CuttingResetBank(args.reset_bank, env.metadata['benchmark_contract'])
            if args.reset_bank else None)
    generator = torch.Generator().manual_seed(args.seed)
    categories = torch.zeros(args.worlds, dtype=torch.long, device=env.device)
    history = spec.pack_batch(env.current)

    def reset(mask=None):
        nonlocal history
        if mask is None:
            mask = torch.ones(args.worlds, device=env.device, dtype=torch.bool)
        if bank is None:
            frame = env.reset(mask)
            history = spec.reset_batch_history(frame, history, mask)
        else:
            rows, kinds = bank.sample(int(mask.sum()), generator)
            saved_history = restore_snapshot(env, rows, mask)
            history[mask] = saved_history
            history[mask, :spec.base_dim] = spec._base_batch(env.current)[mask]
            categories[mask] = kinds.to(env.device)
        return env.current

    reset()

    def pack(frame):
        return {key: frame[key].cpu().tolist() for key in keys}

    def overflow():
        return wp.to_torch(env.physics.robot.data.overflow).cpu().tolist()

    def handle(message):
        nonlocal history
        operation = message['op']
        if operation == 'init':
            return dict(metadata=env.metadata, model_metadata=metadata,
                        frame=pack(env.current), overflow=overflow(), history=history.cpu().tolist(),
                        reset_bank=bank.metadata if bank else None)
        if operation == 'reset':
            returns.zero_()
            return dict(frame=pack(reset()), overflow=overflow(), history=history.cpu().tolist())
        if operation == 'step':
            action = torch.tensor(message['action'], device=env.device, dtype=torch.float32)
            if action.shape != (args.worlds, 6) or not torch.isfinite(action).all():
                raise ValueError('Expected finite batched six-component actions')
            frame, reward, terminated, truncated, info = env.step(action)
            history = spec.pack_batch_transition(frame, history)
            returns.add_(reward)
            done = terminated | truncated
            response = dict(post=pack(frame), reward=reward.cpu().tolist(),
                            terminated=terminated.cpu().tolist(), truncated=truncated.cpu().tolist(),
                            overflow=overflow(), episode_metrics={})
            response['material_force_max'] = float(info['peak_force'].max())
            if done.any():
                response['episode_metrics']['Episode_Metrics/return_mean'] = float(returns[done].mean())
                for kind in range(len(STAGES) if bank else 1):
                    selected = done & (categories == kind)
                    if not selected.any():
                        continue
                    response['episode_metrics'][f'Episode_Metrics/type_{kind}_episodes'] = int(selected.sum())
                    response['episode_metrics'][f'Episode_Metrics/type_{kind}_successes'] = int((selected & info['success']).sum())
                frame = reset(done)
                returns[done] = 0
            response['frame'] = pack(frame)
            response['history'] = history.cpu().tolist()
            return response
        if operation == 'close':
            return {}
        raise ValueError(f'Unknown physical operation: {operation}')

    serve(args.protocol_fd, handle)


if __name__ == '__main__':
    main()
