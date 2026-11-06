"""Replay each bank stage after restoration and check native transition alignment."""
import argparse
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cutting_world_model.benchmark import physical_contract, write_json
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.manipulation import CuttingMDP
from cutting_world_model.provenance import git_revision
from cutting_world_model.reset_bank import STAGES, CuttingResetBank, capture_snapshot, restore_snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend-root', type=Path, default=ROOT / 'backends/mujoco')
    p.add_argument('--scene', required=True)
    p.add_argument('--capture', required=True)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-stage', type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    sys.path.insert(0, str(args.backend_root))
    from cutting_world_model.gpu_env import GPUEnvironment
    torch.set_num_threads(2)
    begin = time.monotonic()
    n = args.per_stage * len(STAGES)
    env = GPUEnvironment(args.backend_root, n, scene=args.scene, capture=args.capture)
    bank = CuttingResetBank(args.bank, physical_contract(
        args.backend_root, env, args.scene, args.capture, 'public-apple-ur5e-v1'))
    generator = torch.Generator().manual_seed(71)
    rows = [bank.sample(args.per_stage, generator, name)[0] for name in STAGES]
    rows = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
    mask = torch.ones(n, device=env.device, dtype=torch.bool)
    history = restore_snapshot(env, rows, mask)
    one = {k: v[0].cpu().numpy() for k, v in env.current.items()}
    mdp = CuttingMDP(ControlStateSpec.metadata_from_frame(env.metadata, one), env.device)
    history[:, :mdp.spec.base_dim] = mdp.spec._base_batch(env.current)
    saved = capture_snapshot(env, history, mask)
    state = torch.cat((history, env.current['step_count']), -1).float()
    action = torch.zeros(n, 6, device=env.device)
    first, reward, term, trunc, info = env.step(action)
    packed = mdp.spec.pack_batch_transition(first, history).float()
    post = torch.cat((packed, first['step_count'].float()), -1)
    goal = first['target_pose'].float()
    predicted = mdp.integrate(state, mdp.targets(state, post), action, goal)
    reconstructed, done = mdp.reward(state, action, predicted, goal)
    restore_snapshot(env, saved, mask, new_episode=False)
    second, reward2, term2, trunc2, _ = env.step(action)
    metrics = {
        'git': git_revision(ROOT), 'command': sys.argv, 'worlds': n,
        'joint_q_replay_max_error': float((first['joint_q'] - second['joint_q']).abs().max()),
        'reward_replay_max_error': float((reward - reward2).abs().max()),
        'particle_q_replay_max_error': float((first['particle_q'] - second['particle_q']).abs().max()),
        'oracle_observation_max_error': float((mdp.observe(predicted, goal) - first['observation']).abs().max()),
        'oracle_reward_max_error': float((reconstructed - reward).abs().max()),
        'oracle_terminal_mismatches': int((done.bool() != (term | trunc)).sum()),
        'terminal_replay_mismatches': int(((term != term2) | (trunc != trunc2)).sum()),
        'wall_seconds': time.monotonic() - begin,
    }
    write_json(args.output / 'report.json', metrics)
    print(metrics, flush=True)
    torch.testing.assert_close(first['joint_q'], second['joint_q'], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reward, reward2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(reconstructed, reward, atol=1e-4, rtol=1e-4)
    assert metrics['oracle_terminal_mismatches'] == metrics['terminal_replay_mismatches'] == 0


if __name__ == '__main__':
    main()
