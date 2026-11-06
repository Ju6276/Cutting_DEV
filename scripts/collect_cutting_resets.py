"""Collect stage resets from native cutting trajectories and bounded actions."""
import argparse
import json
import random
import signal
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cutting_world_model.benchmark import physical_contract, write_json
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.provenance import git_revision
from cutting_world_model.reset_bank import SCHEMA, STAGES, capture_snapshot, stage_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend-root', type=Path, default=ROOT / 'backends/mujoco')
    parser.add_argument('--scene', required=True)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--worlds', type=int, default=64)
    parser.add_argument('--per-stage', type=int, default=128)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', choices=('train', 'test'), required=True)
    parser.add_argument('--wall-seconds', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.worlds, args.per_stage, args.stride, args.wall_seconds) <= 0:
        parser.error('Collection sizes and budget must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.backend_root.resolve()))
    from cutting_world_model.gpu_env import GPUEnvironment
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    begin = time.monotonic()
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    provenance = {'git': git_revision(ROOT), 'command': sys.argv,
                  'method': 'native prefixes of bounded-action rollouts',
                  'wall_budget_seconds': args.wall_seconds,
                  'endpoint': 'all stage reservoirs filled or wall budget',
                  'proposal': 'uniform first four actions per episode; symmetric alternating fifth action',
                  'stride': args.stride}
    write_json(args.output / 'manifest.json', provenance)
    env = GPUEnvironment(args.backend_root, args.worlds, scene=args.scene, capture=args.capture)
    contract = json.loads(json.dumps(physical_contract(
        args.backend_root, env, args.scene, args.capture, 'public-apple-ur5e-v1')))
    frame = {k: v[0].cpu().numpy() for k, v in env.current.items()}
    spec = ControlStateSpec(ControlStateSpec.metadata_from_frame(env.metadata, frame)).to(env.device)
    history = spec.pack_batch(env.current)
    reservoirs = {name: [] for name in STAGES}
    seen = {name: 0 for name in STAGES}
    episodes = successes = collisions = steps = 0
    closest = [None] * args.worlds
    closest_error = torch.full((args.worlds,), float('inf'), device=env.device,
                               dtype=env.current['ee_pose'].dtype)
    low, high = torch.tensor(env.metadata['action_bounds'], device=env.device)

    def proposals(count):
        value = low + (high - low) * torch.rand(count, 6, device=env.device)
        # Gains consume [0, 1] in the source mapper; its negative half is identical to zero.
        value[:, :4] = torch.rand(count, 4, device=env.device)
        return value

    action = proposals(args.worlds)

    def insert(name, row):
        seen[name] += 1
        slot = len(reservoirs[name])
        if slot >= args.per_stage:
            slot = rng.randrange(seen[name])
        if slot < args.per_stage:
            if slot == len(reservoirs[name]):
                reservoirs[name].append(row)
            else:
                reservoirs[name][slot] = row

    def record(name, mask):
        indices = mask.nonzero().flatten().tolist()
        choices = []
        for i in indices:
            seen[name] += 1
            slot = len(reservoirs[name])
            if slot >= args.per_stage:
                slot = rng.randrange(seen[name])
            if slot < args.per_stage:
                choices.append((i, slot))
                if slot == len(reservoirs[name]):
                    reservoirs[name].append(None)
        if not choices:
            return
        selected = torch.zeros_like(mask)
        selected[[i for i, _ in choices]] = True
        snapshot = capture_snapshot(env, history, selected)
        for row, (_, slot) in enumerate(choices):
            reservoirs[name][slot] = {key: value[row].clone() for key, value in snapshot.items()}

    record('full_task', torch.ones(args.worlds, device=env.device, dtype=torch.bool))
    while not stop and time.monotonic() - begin < args.wall_seconds:
        executed = action.clone()
        # Period is expressed in the source WM window; each proposal stays inside action bounds.
        executed[:, 4] *= 1 if (steps // args.stride) % 2 == 0 else -1
        frame, reward, term, trunc, info = env.step(executed)
        history = spec.pack_batch_transition(frame, history)
        steps += 1
        masks = stage_masks(env)
        record('before_cut', masks['before_cut'])
        if steps % args.stride == 0:
            record('cutting', masks['cutting'])
        error = (frame['ee_pose'][:, 2] - frame['target_pose'][:, 2]).abs()
        improved = masks['cutting'] & (error < closest_error)
        if improved.any():
            snapshots = capture_snapshot(env, history, improved)
            for row, i in enumerate(improved.nonzero().flatten().tolist()):
                closest[i] = {key: value[row].clone() for key, value in snapshots.items()}
            closest_error[improved] = error[improved]
        ended = term | trunc
        episodes += int(ended.sum())
        successes += int(info['success'].sum())
        collisions += int(info['collision'].sum())
        if ended.any():
            for i in ended.nonzero().flatten().tolist():
                if closest[i] is not None:
                    insert('closest_cut', closest[i])
                    closest[i] = None
            closest_error[ended] = float('inf')
            frame = env.reset(ended)
            history = spec.reset_batch_history(frame, history, ended)
            action[ended] = proposals(int(ended.sum()))
            record('full_task', ended)
        progress = {'vector_steps': steps, 'real_transitions': steps * args.worlds,
                    'counts': {k: len(v) for k, v in reservoirs.items()},
                    'episodes': episodes, 'successes': successes, 'collisions': collisions,
                    'max_released_springs': int(info['released_springs'].max()),
                    'minimum_goal_z_error_m': float((frame['ee_pose'][:, 2] - frame['target_pose'][:, 2]).abs().min()),
                    'wall_seconds': time.monotonic() - begin}
        write_json(args.output / 'progress.json', progress)
        if steps % args.stride == 0:
            print(json.dumps(progress), flush=True)
        available_closest = len(reservoirs['closest_cut']) + sum(row is not None for row in closest)
        if (all(len(reservoirs[k]) >= args.per_stage for k in STAGES[:-1])
                and available_closest >= args.per_stage):
            break
    for row in closest:
        if row is not None:
            insert('closest_cut', row)
    progress['counts'] = {k: len(v) for k, v in reservoirs.items()}
    states = {name: {key: torch.stack([row[key] for row in rows]) for key in rows[0]} if rows else {}
              for name, rows in reservoirs.items()}
    complete = all(len(v) >= args.per_stage for v in reservoirs.values())
    artifact = {'schema': SCHEMA, 'contract': contract, 'states': states,
                'split': args.split, 'seed': args.seed, 'provenance': provenance,
                'model_metadata': spec.metadata}
    torch.save(artifact, args.output / ('bank.pt' if complete else 'partial.pt'))
    torch.load(args.output / ('bank.pt' if complete else 'partial.pt'), weights_only=True)
    write_json(args.output / 'report.json', dict(progress, complete=complete,
               peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20))
    if not complete:
        raise RuntimeError('Collection budget ended before every stage was filled; see partial.pt')


if __name__ == '__main__':
    main()
