#!/usr/bin/env python3
"""Run predefined perturbation cases with one frozen controller, or compare two such runs.

    evaluate_perturbed.py run --checkpoint CKPT --cases 256 --seed-base 3000000 --output DIR
    evaluate_perturbed.py run --method sac --checkpoint MODEL_DIR --sac-python ... --output DIR
    evaluate_perturbed.py compare --a DIR_A --b DIR_B --output FILE

The perturbation is added to the executed action: the controller's action is clipped to the
task bounds, the case's perturbation is added, and the result is clipped again. This is the
same for every controller (a policy whose raw output saturates beyond the bounds gets no
protection from the clipping). Runs recorded before this rule added the perturbation to the
raw policy output; `compare` refuses to mix the two.

`--method supermimic-batched` (default) runs SuperMimic-style policies in parallel worlds.
Other methods use the controllers of `evaluate_benchmark.py` in one world, case after case;
their method-specific arguments are passed through. Every case runs to its own end (success,
force limit, workspace or the 500-step time limit). At the wall budget unfinished cases are
listed, not dropped silently.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model.control_state import ControlStateSpec
from cutting_world_model.perturbed_eval import (CaseNoise, compare_cases, make_cases, outcome,
                                                summarize_cases)
from cutting_world_model.policy import SliceItPolicy
from cutting_world_model.provenance import git_revision
from cutting_world_model.task_heads import without_reward_settings

PERTURBATION_POINT = 'executed_action'
LEGACY_POINT = 'raw_policy_output'
SINGLE_WORLD_METHODS = ('supermimic', 'sac', 'wm-mppi', 'constant-zero')


def perturbed(spec, action, noise):
    """Clip the controller's action, add the case perturbation, clip again."""
    return spec.execute(spec.execute(action) + noise)


class CaseRecord:
    def __init__(self, case):
        self.case = case
        self.noise = CaseNoise(case)
        self.steps, self.ret, self.max_force, self.impulse, self.margin, self.seconds = 0, 0., 0., 0., 0., 0.

    def add(self, reward, info, world, elapsed):
        self.steps += 1
        self.ret += float(reward[world])
        self.max_force = max(self.max_force, float(info['peak_force'][world]))
        self.impulse += float(info['material_impulse'][world])
        self.margin = max(self.margin, float(info['safety_margin'][world]))
        # Actual simulated time; a force-limit stop ends inside the control step.
        self.seconds += float(elapsed[world])

    def row(self, info, trunc, world):
        return {'id': self.case['case'], 'outcome': outcome(
            bool(info['success'][world]), bool(info['collision'][world]),
            bool(info['outside'][world]), bool(trunc[world])),
            'steps': self.steps, 'return': self.ret, 'max_material_force_n': self.max_force,
            'simulation_seconds': self.seconds, 'mean_material_force_n': self.impulse / self.seconds,
            'safety_margin': self.margin, 'released_springs': int(info['released_springs'][world]),
            **{k: self.case[k] for k in ('sigma', 'start_step', 'mode')}}


def finish(args, rows, cases, started, stop_reason):
    rows.sort(key=lambda r: r['id'])
    unfinished = sorted(set(range(len(cases))) - {r['id'] for r in rows})
    with (args.output / 'cases.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    summary = {'summary': summarize_cases(rows) if rows else None, 'unfinished_cases': unfinished,
               'stop_reason': stop_reason, 'seconds': time.monotonic() - started}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


def write_manifest(args, cases, extra):
    (args.output / 'manifest.json').write_text(json.dumps({
        'git': git_revision(ROOT), 'command': sys.argv, 'method': args.method,
        'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'perturbation_point': PERTURBATION_POINT, 'cases': cases, **extra}, indent=1, default=str) + '\n')


def run_batched(args):
    from cutting_world_model.gpu_env import GPUEnvironment
    started = time.monotonic()
    env = GPUEnvironment(str((ROOT / 'backends/mujoco').resolve()), args.worlds)
    device = env.device
    frame = env.current
    metadata = ControlStateSpec.metadata_from_frame(env.metadata, {k: v[0].cpu().numpy() for k, v in frame.items()})
    spec = ControlStateSpec(metadata).to(device)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if without_reward_settings(saved['metadata']) != without_reward_settings(metadata):
        raise ValueError('Checkpoint and physical task metadata differ')
    policy = SliceItPolicy().to(device).eval()
    policy.load_state_dict(saved['learner']['policy'])
    cases = make_cases(args.cases, args.seed_base)
    write_manifest(args, cases, {'policy_updates': saved['learner'].get('updates'), 'worlds': args.worlds})
    queue = list(range(len(cases)))
    active = [None] * args.worlds
    rows, records = [], {}

    def start(world):
        if queue:
            index = queue.pop(0)
            active[world] = index
            records[index] = CaseRecord(cases[index])
        else:
            active[world] = None

    for world in range(args.worlds):
        start(world)
    history = frame['observation'].float()[None].repeat(policy.history, 1, 1)
    stop_reason = None
    with torch.no_grad():
        while any(a is not None for a in active):
            if time.monotonic() - started > args.wall_seconds:
                stop_reason = 'wall budget'
                break
            clean = policy(history, frame['target_pose'].float())
            noise = torch.stack([records[a].noise(records[a].steps) if a is not None else torch.zeros(6)
                                 for a in active]).to(device)
            before = frame['sim_time'][:, 0].double().clone()
            frame, reward, term, trunc, info = env.step(perturbed(spec, clean, noise))
            elapsed = (frame['sim_time'][:, 0].double() - before).cpu()
            history = torch.cat((history[1:], frame['observation'].float()[None]), 0)
            ended = (term | trunc).cpu()
            for world, index in enumerate(active):
                if index is None:
                    continue
                records[index].add(reward, info, world, elapsed)
                if bool(ended[world]):
                    rows.append(records[index].row(info, trunc, world))
                    start(world)
            if bool(ended.any()):
                mask = ended.to(device)
                frame = env.reset(mask)
                history[:, mask] = frame['observation'][mask].float()[None]
    finish(args, rows, cases, started, stop_reason)


def benchmark_module():
    spec = importlib.util.spec_from_file_location('evaluate_benchmark', ROOT / 'scripts/evaluate_benchmark.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_single(args, extra):
    """Controllers of evaluate_benchmark.py (SAC, WM-MPPI, ...) in one world, case after case."""
    from cutting_world_model.gpu_env import GPUEnvironment
    started = time.monotonic()
    bench = benchmark_module()
    passthrough = list(extra) + ['--output', str(args.output / 'unused')]
    if args.checkpoint is not None:
        passthrough += ['--checkpoint', str(args.checkpoint)]
    bargs = bench.argument_parser(default_method=args.method).parse_args(passthrough)
    env = GPUEnvironment(str((ROOT / 'backends/mujoco').resolve()), 1)
    frame = env.current
    metadata = ControlStateSpec.metadata_from_frame(env.metadata, {k: v[0].cpu().numpy() for k, v in frame.items()})
    spec = ControlStateSpec(metadata).to(env.device)
    controller_manifest = {}
    controller = bench.make_controller(bargs, env, controller_manifest)
    cases = make_cases(args.cases, args.seed_base)
    write_manifest(args, cases, {'worlds': 1, 'controller': controller_manifest, 'passthrough': list(extra)})
    rows, stop_reason = [], None
    try:
        with torch.no_grad():
            for case in cases:
                if time.monotonic() - started > args.wall_seconds:
                    stop_reason = 'wall budget'
                    break
                frame = env.reset()
                controller.reset(frame)
                record = CaseRecord(case)
                while True:
                    action = controller.act(frame).reshape(1, 6).to(env.device)
                    before = frame['sim_time'][:, 0].double().clone()
                    frame, reward, term, trunc, info = env.step(
                        perturbed(spec, action, record.noise(record.steps)[None].to(env.device)))
                    elapsed = (frame['sim_time'][:, 0].double() - before).cpu()
                    controller.observe(frame)
                    record.add(reward, info, 0, elapsed)
                    if bool((term | trunc)[0]):
                        rows.append(record.row(info, trunc, 0))
                        break
    finally:
        if hasattr(controller, 'close'):
            controller.close()
    finish(args, rows, cases, started, stop_reason)


def load_run(directory):
    """Cases of one completed run; refuses runs with unfinished cases."""
    manifest = json.loads((directory / 'manifest.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    rows = [json.loads(line) for line in (directory / 'cases.jsonl').read_text().splitlines()]
    if summary['unfinished_cases'] or len(rows) != len(manifest['cases']):
        raise ValueError(f'{directory}: not every predefined case finished')
    if sorted(r['id'] for r in rows) != [c['case'] for c in manifest['cases']]:
        raise ValueError(f'{directory}: case indices differ from the manifest')
    return manifest, rows


def compare(args):
    manifest_a, a = load_run(args.a)
    manifest_b, b = load_run(args.b)
    if manifest_a['cases'] != manifest_b['cases']:
        raise ValueError('The two runs used different case configurations or random streams')
    points = [m.get('perturbation_point', LEGACY_POINT) for m in (manifest_a, manifest_b)]
    if points[0] != points[1]:
        raise ValueError('The two runs added the perturbation at different points: %s vs %s' % tuple(points))
    report = {'a': str(args.a), 'b': str(args.b), 'perturbation_point': points[0],
              'summary_a': summarize_cases(a), 'summary_b': summarize_cases(b),
              'a_minus_b': compare_cases(a, b)}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['a_minus_b']['all_cases']['success']))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    r = sub.add_parser('run')
    r.add_argument('--method', choices=('supermimic-batched',) + SINGLE_WORLD_METHODS,
                   default='supermimic-batched')
    r.add_argument('--checkpoint', type=Path)
    r.add_argument('--cases', type=int, default=256)
    r.add_argument('--seed-base', type=int, default=3_000_000)
    r.add_argument('--worlds', type=int, default=32)
    r.add_argument('--wall-seconds', type=float, default=2400.)
    r.add_argument('--output', type=Path, required=True)
    c = sub.add_parser('compare')
    c.add_argument('--a', type=Path, required=True)
    c.add_argument('--b', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    args, extra = parser.parse_known_args()
    if args.mode == 'compare':
        if extra:
            parser.error('unrecognized arguments: ' + ' '.join(extra))
        return compare(args)
    if args.method == 'supermimic-batched':
        if extra:
            parser.error('unrecognized arguments: ' + ' '.join(extra))
        if args.checkpoint is None:
            parser.error('--checkpoint is required')
    args.output.mkdir(parents=True, exist_ok=False)
    run_batched(args) if args.method == 'supermimic-batched' else run_single(args, extra)


if __name__ == '__main__':
    main()
