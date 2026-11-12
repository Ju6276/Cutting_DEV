"""Critic accuracy and imagined-vs-physical returns for a frozen checkpoint.

Inputs: a checkpoint, a frozen evaluate_benchmark.py output (episode_*.npz with
complete physical frames) and optionally a held-out episode split (split.json +
episodes/). No optimizer updates; all networks are frozen.

Recorded rewards use the reward of the recording task (the SliceIt reward for the
benchmark and the held-out data). A checkpoint trained with another force penalty
is compared with returns of its own reward, recomputed from the recorded
observations (value_diagnostics.reward_under).
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model.policy import SliceItCritic, SliceItPolicy  # noqa: E402
from cutting_world_model.provenance import git_revision  # noqa: E402
from cutting_world_model.task_heads import (SliceItHeads, reward_settings,  # noqa: E402
                                            without_reward_settings)
from cutting_world_model.trainer import OnlineLearner, chunks  # noqa: E402
from cutting_world_model.value_diagnostics import complete_episode_returns, reward_under  # noqa: E402
from cutting_world_model.world_model import CuttingWorldModel  # noqa: E402


def stats(v, r):
    err = (v - r).double()
    return dict(samples=len(r), mae=float(err.abs().mean()), rmse=float(err.square().mean().sqrt()),
                bias=float(err.mean()), value_min=float(v.min()), value_max=float(v.max()),
                return_min=float(r.min()), return_max=float(r.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--heldout', type=Path, help='Held-out split directory (split.json, episodes/)')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    started = time.monotonic()
    dev = args.device
    torch.manual_seed(0)
    saved = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    metadata = saved['metadata']
    wm = CuttingWorldModel(metadata).to(dev)
    data = saved['learner']
    variant = data.get('critic_variant', {'target': 'imagined'})
    learner = OnlineLearner(wm, SliceItPolicy().to(dev), SliceItCritic(wm.spec.state_dim).to(dev),
                            microbatch=64, critic_target=variant['target'],
                            actor_horizon=data.get('actor_horizon'),
                            actor_bc_alpha=data.get('actor_bc_alpha'),
                            actor_objective=data.get('actor_objective', 'model'))
    learner.restore(data)
    wm.enable_compile(None)
    for module in (learner.wm, learner.policy, learner.critic):
        module.eval().requires_grad_(False)
    spec, cfg = wm.spec, learner.pcfg
    gamma = cfg.gamma
    # Benchmark and held-out data were recorded with the SliceIt reward.
    task = json.loads((args.evaluation / 'task.json').read_text())
    if reward_settings(task['environment']):
        raise ValueError('Evaluation episodes were recorded with a non-SliceIt reward')
    recorded_heads = SliceItHeads(without_reward_settings(metadata)['environment'], metadata['controller'])

    def own_rewards(states, reward):
        return reward_under(spec.heads, recorded_heads, reward,
                            spec.field(states[1:], 'observation').double().cpu())

    @torch.no_grad()
    def values(states, goals):
        return torch.cat([learner.critic_values(states[p], goals[p]) for p in chunks(len(states), 4096)])

    def load_benchmark():
        episodes = []
        for path in sorted(args.evaluation.glob('episode_*.npz')):
            with np.load(path) as d:
                data = {k: d[k] for k in d.files}
            states = spec.pack_episode(data, device=dev)
            reward = own_rewards(states, torch.as_tensor(data['reward'], dtype=torch.float64))
            term, trunc = data['terminated'].astype(bool), data['truncated'].astype(bool)
            goal = torch.as_tensor(data['target_pose'][0], device=dev, dtype=torch.float32)
            complete = bool(term[-1]) and not bool(trunc[-1])
            ret = complete_episode_returns(reward, term, trunc, gamma) if complete else None
            episodes.append(dict(id=path.name, states=states, goal=goal,
                                 ret=None if ret is None else ret.to(dev).float(), steps=len(reward)))
        return episodes

    def load_heldout():
        split = json.loads((args.heldout / 'split.json').read_text())
        episodes = []
        for item in split['episodes']:
            if item['split'] != 'heldout':
                continue
            with np.load(args.heldout / 'episodes' / item['path']) as d:
                states = torch.as_tensor(np.concatenate((d['state'][:1], d['next_state'])), device=dev)
                reward = own_rewards(states, torch.as_tensor(d['reward'], dtype=torch.float64))
                ret = complete_episode_returns(reward, d['terminated'].astype(bool),
                                               d['truncated'].astype(bool), gamma)
                episodes.append(dict(id=item['path'], states=states,
                                     goal=torch.as_tensor(d['goal'][0], device=dev, dtype=torch.float32),
                                     ret=ret.to(dev).float(), steps=len(reward)))
        return episodes

    def flat(episodes):
        s = torch.cat([e['states'][:-1] for e in episodes])
        g = torch.cat([e['goal'].expand(e['steps'], -1) for e in episodes])
        r = torch.cat([e['ret'] for e in episodes])
        return s, g, r

    @torch.no_grad()
    def imagined_vs_physical(episodes):
        """H-step imagination from every recorded state with the frozen policy and WM."""
        rows, imag_disc, imag_boot, phys = [], [], [], []
        for e in episodes:
            T = e['steps']
            hist = torch.stack([e['states'][torch.clamp(torch.arange(t - cfg.history + 1, t + 1, device=dev),
                                                         min=0)] for t in range(T)], 1)
            goals = e['goal'].expand(T, -1)
            ro = learner.imagine(e['states'][:T], hist, goals)
            disc = gamma ** torch.arange(learner.actor_horizon, device=dev, dtype=torch.float32)
            reward_part = (disc[:, None] * ro.survival[:-1] * ro.rewards).sum(0)
            terminal_value = learner.critic_values(ro.states[-1], goals)
            boot = (gamma ** learner.actor_horizon) * ro.survival[-1] * terminal_value
            collision = (spec.field(ro.states[1:], 'peak_wrench') > spec.force_limits).any(-1)
            done, _, _, _ = spec.heads.termination(
                spec.field(ro.states[1:], 'observation'), spec.field(ro.states[:-1], 'success_counter'),
                collision)
            first = torch.where(done.any(0), done.float().argmax(0) + 1,
                                torch.full_like(done[0], -1, dtype=torch.long))
            for t in range(T):
                rows.append(dict(episode=e['id'], start=t, physical_return=float(e['ret'][t]),
                                 physical_steps_left=T - t, imagined_reward_part=float(reward_part[t]),
                                 imagined_bootstrap=float(boot[t]), imagined_first_terminal=int(first[t]),
                                 imagined_terminal_reward=(float(ro.rewards[first[t] - 1, t])
                                                           if first[t] > 0 else None),
                                 survival_H=float(ro.survival[-1, t]), terminal_value=float(terminal_value[t])))
            imag_disc.append(reward_part)
            imag_boot.append(boot)
            phys.append(e['ret'])
        imag_disc, imag_boot, phys = torch.cat(imag_disc), torch.cat(imag_boot), torch.cat(phys)
        total = imag_disc + imag_boot
        summary = dict(starts=len(phys), physical_mean=float(phys.mean()),
                       imagined_reward_part_mean=float(imag_disc.mean()),
                       imagined_bootstrap_mean=float(imag_boot.mean()),
                       imagined_objective_mean=float(total.mean()),
                       objective_minus_physical=stats(total, phys),
                       reward_part_minus_physical=stats(imag_disc, phys),
                       nonterminal_paths=sum(r['imagined_first_terminal'] < 0 for r in rows),
                       terminal_value_absmax=max(abs(r['terminal_value']) for r in rows),
                       positive_terminal_reward_paths=sum((r['imagined_terminal_reward'] or 0) > 50 for r in rows),
                       negative_terminal_reward_paths=sum((r['imagined_terminal_reward'] or 0) < -50 for r in rows))
        return summary, rows

    args.output.mkdir(parents=True, exist_ok=True)
    bench = load_benchmark()
    complete = [e for e in bench if e['ret'] is not None]
    result = dict(git=git_revision(ROOT), checkpoint=str(args.checkpoint), evaluation=str(args.evaluation),
                  critic_variant=variant, policy_updates=data['updates'],
                  wm_updates=data['training_stage']['wm_updates'],
                  training_reward=reward_settings(metadata['environment']) or 'sliceit',
                  benchmark_episodes=len(bench), complete_episodes=len(complete))
    if complete:
        s, g, r = flat(complete)
        result['benchmark_value'] = stats(values(s, g), r)
        result['benchmark_initial'] = [dict(critic=float(values(e['states'][:1], e['goal'][None])[0]),
                                            ret=float(e['ret'][0])) for e in complete]
        result['benchmark_imagined_vs_physical'], rows = imagined_vs_physical(complete)
        (args.output / 'benchmark_imagined_rows.json').write_text(json.dumps(rows))
    if args.heldout is not None:
        held = load_heldout()
        s, g, r = flat(held)
        result['heldout_value'] = stats(values(s, g), r)
        result['heldout_imagined_vs_physical'], rows = imagined_vs_physical(held)
        (args.output / 'heldout_imagined_rows.json').write_text(json.dumps(rows))
    result['wall_seconds'] = time.monotonic() - started
    (args.output / 'value_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in result if 'value' in k or 'imagined' in k}, indent=1))


if __name__ == '__main__':
    main()
