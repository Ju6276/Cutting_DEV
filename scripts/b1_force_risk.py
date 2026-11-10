#!/usr/bin/env python3
"""Route B1 force-risk screen (PLAN "Route B"): one offline screen, no policy updates.

Subcommands, run in order:
  fit-ensemble  five independently initialized WMs on the declared fit data
  fit-head      0.95 pinball head on frozen features of the existing (mean) WM
  select        development split only: score every family, choose one operating setting
                per family, one B candidate and one ensemble comparator; freeze them
  test          opens the sealed held-out split once and applies the frozen advancement rule

Fit data: the working-baseline training replay plus the 2026-11-05 fixed-data training
split. Development: reliability-supplement ``dev``. Held-out: supplement ``final``.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backends/mujoco')]
from cutting_world_model import force_risk as fr
from cutting_world_model.policy import SliceItCritic, SliceItPolicy
from cutting_world_model.provenance import git_revision
from cutting_world_model.reliability import load_episodes
from cutting_world_model.trainer import OnlineLearner
from cutting_world_model.world_model import CuttingWorldModel

RUNS = ROOT / 'runs'
BASE = RUNS / 'critic_abl_a250_20261106/training/last.pt'
REPLAY = RUNS / 'critic_abl_a250_20261106/training/replay.pt'
FIXED = RUNS / 'wm_frozen_fit_20261105'
SUPPLEMENT = RUNS / 'wm_reliability_supplement_20261108'
DEVICE = 'cuda:0'
BUDGET = 0.10
THRESHOLDS = [round(float(x), 2) for x in np.arange(0.05, 3.0001, 0.01)]
QUANTILES = [0.999, 0.995, 0.99, 0.98, 0.975, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]
B_FAMILIES = ('quantile_head', 'residual_phase', 'residual_global')  # tie order
ENSEMBLE_FAMILIES = ('ensemble_max', 'ensemble_mean2std')
MIN_PHASE_STEPS = 20


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def load_wm(path, state_dict_key=('learner', 'world_model')):
    saved = torch.load(path, map_location=DEVICE, weights_only=False)
    wm = CuttingWorldModel(saved['metadata']).to(DEVICE)
    state = saved
    for key in state_dict_key:
        state = state[key]
    wm.load_state_dict(state)
    return wm.eval(), saved


def fit_data(spec):
    fixed = load_episodes(FIXED, 'train', DEVICE)
    return fr.load_fit_replay(REPLAY, fixed, spec.state_dim, DEVICE), fixed


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def fit_ensemble(args):
    args.output.mkdir(parents=True, exist_ok=True)
    base = torch.load(BASE, map_location='cpu', weights_only=False)
    metadata = base['metadata']
    replay, fixed = fit_data(CuttingWorldModel(metadata).spec)
    write(args.output / 'manifest.json', {
        'git': git_revision(ROOT), 'command': sys.argv, 'base_metadata_from': str(BASE),
        'replay': str(REPLAY), 'fixed_train_episodes': len(fixed), 'seeds': args.seeds,
        'updates': args.updates, 'microbatch': args.microbatch,
        'fit_rows': int(replay.valid.sum()), 'wm_terminal_gradient': 'dynamics',
        'note': 'fresh initialization per seed; uniform window sampling over the fixed fit replay'})
    for member, seed in enumerate(args.seeds):
        out = args.output / f'member_{member}.pt'
        if out.exists():
            continue
        started = time.monotonic()
        torch.manual_seed(seed)
        wm = CuttingWorldModel(metadata).to(DEVICE)
        learner = OnlineLearner(wm, SliceItPolicy().to(DEVICE), SliceItCritic(wm.spec.state_dim).to(DEVICE),
                                microbatch=args.microbatch)
        log = (args.output / f'member_{member}.jsonl').open('w')
        for update in range(1, args.updates + 1):
            metrics = learner.fit_world_model(replay)
            if update % 10 == 0 or update == args.updates:
                log.write(json.dumps({'update': update, 'seconds': time.monotonic() - started, **metrics}) + '\n')
                log.flush()
        torch.save({'metadata': metadata, 'world_model': wm.state_dict(), 'seed': seed,
                    'updates': args.updates}, out)
        print('member', member, 'seed', seed, 'seconds', round(time.monotonic() - started), flush=True)


def fit_head(args):
    args.output.mkdir(parents=True, exist_ok=True)
    wm, _ = load_wm(BASE)
    replay, fixed = fit_data(wm.spec)
    worlds, rows = fr.replay_transitions(replay)
    torch.manual_seed(args.seed)
    head = fr.QuantileHead(wm.dynamics.mlp.net[-1].in_features).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    log = (args.output / 'head.jsonl').open('w')
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        pick = torch.randint(len(worlds), (args.batch,), device=DEVICE)
        e, t = worlds[pick], rows[pick]
        state, target = replay.s[e, t], fr.utilization(wm.spec, replay.s[e, t + 1])
        hidden, predicted = fr.wm_forward(wm, state, replay.a_exec[e, t], replay.xi[e, t])
        loss = fr.pinball(head(hidden, predicted, fr.utilization(wm.spec, state)), target, args.quantile)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            log.write(json.dumps({'step': step, 'pinball': float(loss), 'seconds': time.monotonic() - started}) + '\n')
            log.flush()
    torch.save({'head': head.state_dict(), 'hidden_dim': head.net[0].in_features - 2,
                'quantile': args.quantile, 'base': str(BASE)}, args.output / 'head.pt')
    write(args.output / 'manifest.json', {
        'git': git_revision(ROOT), 'command': sys.argv, 'base': str(BASE), 'base_sha256': sha256(BASE),
        'replay': str(REPLAY), 'fixed_train_episodes': len(fixed), 'transitions': int(len(worlds)),
        'quantile': args.quantile, 'steps': args.steps, 'batch': args.batch, 'lr': args.lr,
        'seed': args.seed, 'seconds': time.monotonic() - started})


@torch.no_grad()
def raw_scores(episodes, ensemble_dir, head_dir):
    """Per family and episode: per-interval predicted utilization (no thresholds yet)."""
    wm, _ = load_wm(BASE)
    members = sorted(ensemble_dir.glob('member_*.pt'))
    if len(members) != 5:
        raise RuntimeError('Expected 5 ensemble members, found %d' % len(members))
    ens = [load_wm(p, ('world_model',))[0] for p in members]
    saved = torch.load(head_dir / 'head.pt', map_location=DEVICE, weights_only=False)
    head = fr.QuantileHead(saved['hidden_dim']).to(DEVICE)
    head.load_state_dict(saved['head'])
    head.eval()
    out = {k: {} for k in ('mean_wm', 'members', 'quantile_head')}
    for ep in episodes:
        state, action, goal = ep['states'][:-1], ep['actions'], ep['goals']
        hidden, predicted = fr.wm_forward(wm, state, action, goal)
        out['mean_wm'][ep['id']] = predicted.cpu().numpy()
        out['members'][ep['id']] = np.stack([fr.wm_forward(m, state, action, goal)[1].cpu().numpy() for m in ens])
        out['quantile_head'][ep['id']] = head(hidden, predicted, fr.utilization(wm.spec, state)).cpu().numpy()
    provenance = {'base': str(BASE), 'base_sha256': sha256(BASE),
                  'members': {str(p): sha256(p) for p in members},
                  'head': str(head_dir / 'head.pt'), 'head_sha256': sha256(head_dir / 'head.pt')}
    return out, provenance


def residual_offsets(records, mean_scores, quantile, by_phase):
    residual = np.concatenate([r['true'] - mean_scores[r['id']] for r in records])
    phases = np.concatenate([r['phase'] for r in records])
    overall = float(np.quantile(residual, quantile, method='higher'))
    if not by_phase:
        return [overall] * len(fr.PHASES)
    return [float(np.quantile(residual[phases == p], quantile, method='higher'))
            if (phases == p).sum() >= MIN_PHASE_STEPS else overall for p in range(len(fr.PHASES))]


def family_upper(family, parameter, records, scores, offsets=None):
    """Per-episode upper score of a family at its frozen setting (residual: R_hat + offset)."""
    if family == 'mean_wm':
        return scores['mean_wm']
    if family == 'quantile_head':
        return scores['quantile_head']
    if family == 'ensemble_max':
        return {i: m.max(0) for i, m in scores['members'].items()}
    if family == 'ensemble_mean2std':
        return {i: m.mean(0) + 2 * m.std(0) for i, m in scores['members'].items()}
    phase = {r['id']: r['phase'] for r in records}
    return {i: s + np.asarray(offsets)[phase[i]] for i, s in scores['mean_wm'].items()}


def family_flags(family, parameter, records, scores, offsets=None):
    upper = family_upper(family, parameter, records, scores, offsets)
    level = 1.0 if family.startswith('residual') else parameter
    return {i: u > level for i, u in upper.items()}, upper


def select(args):
    out = args.output
    if (out / 'frozen.json').exists():
        raise RuntimeError('Frozen selection already exists: ' + str(out / 'frozen.json'))
    out.mkdir(parents=True, exist_ok=True)
    wm, _ = load_wm(BASE)
    dev = load_episodes(SUPPLEMENT, 'dev', DEVICE)
    records = fr.episode_records(wm.spec, dev)
    fr.check_force_limit_labels(records)
    scores, provenance = raw_scores(dev, args.ensemble, args.head)
    report = {}
    for family in ('mean_wm', 'quantile_head') + ENSEMBLE_FAMILIES:
        settings = ((tau, family_flags(family, tau, records, scores)[0]) for tau in THRESHOLDS)
        chosen = fr.select_setting(records, settings, BUDGET)
        if chosen:
            upper = family_upper(family, chosen['parameter'], records, scores)
            chosen['bound'] = fr.bound_quality(records, upper)
        report[family] = chosen
    for family, by_phase in (('residual_phase', True), ('residual_global', False)):
        offsets = {q: residual_offsets(records, scores['mean_wm'], q, by_phase) for q in QUANTILES}
        settings = ((q, family_flags(family, q, records, scores, offsets[q])[0]) for q in QUANTILES)
        chosen = fr.select_setting(records, settings, BUDGET)
        if chosen:
            chosen['offsets'] = offsets[chosen['parameter']]
            upper = family_upper(family, chosen['parameter'], records, scores, chosen['offsets'])
            chosen['bound'] = fr.bound_quality(records, upper)
        report[family] = chosen

    def pick(families):
        eligible = [(f, report[f]) for f in families if report[f]]
        if not eligible:
            return None
        rank = lambda item: (item[1]['dev']['detected'], -item[1]['dev']['false_alarm_episodes'],
                             -families.index(item[0]))
        return max(eligible, key=rank)[0]

    b, ensemble = pick(B_FAMILIES), pick(ENSEMBLE_FAMILIES)
    index = [json.loads(l) for l in (SUPPLEMENT / 'episodes.jsonl').read_text().splitlines() if l.strip()]
    frozen = {
        'git': git_revision(ROOT), 'command': sys.argv, 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'phase_edges': list(fr.PHASE_EDGES), 'phases': list(fr.PHASES), 'false_alarm_budget': BUDGET,
        'provenance': provenance, 'ensemble_dir': str(args.ensemble), 'head_dir': str(args.head),
        'dev_episode_ids': [r['id'] for r in records],
        'test_episode_ids': [x['path'] for x in index if x['split'] == 'final'],
        'b_candidate': None if b is None else {'family': b, **{k: report[b][k] for k in report[b] if k != 'dev'}},
        'ensemble_comparator': None if ensemble is None else {
            'family': ensemble, **{k: report[ensemble][k] for k in report[ensemble] if k != 'dev'}},
        'references': {f: None if report[f] is None else {k: v for k, v in report[f].items() if k != 'dev'}
                       for f in report},
        'status': 'frozen' if b and ensemble else 'inconclusive: no eligible setting on development',
    }
    write(out / 'dev_report.json', {'records': len(records), 'families': report,
                                    'b_candidate': b, 'ensemble_comparator': ensemble})
    write(out / 'frozen.json', frozen)
    print(json.dumps({'b_candidate': b, 'ensemble': ensemble, 'status': frozen['status'],
                      **{f: None if report[f] is None else [report[f]['parameter'], report[f]['dev']['detected'],
                                                            report[f]['dev']['force_limit_episodes'],
                                                            report[f]['dev']['false_alarm_episodes'],
                                                            report[f]['dev']['other_episodes']]
                         for f in report}}, indent=1))


def test(args):
    frozen = json.loads(args.frozen.read_text())
    out = args.frozen.parent / 'test_report.json'
    if out.exists():
        raise RuntimeError('The held-out split was already opened: ' + str(out))
    if frozen['status'] != 'frozen':
        raise RuntimeError('No frozen setting; the held-out split stays sealed')
    scores_dir = Path(frozen['ensemble_dir']), Path(frozen['head_dir'])
    episodes = load_episodes(SUPPLEMENT, 'final', DEVICE)
    if sorted(e['id'] for e in episodes) != sorted(frozen['test_episode_ids']):
        raise RuntimeError('Held-out episodes differ from the frozen list')
    wm, _ = load_wm(BASE)
    records = fr.episode_records(wm.spec, episodes)
    fr.check_force_limit_labels(records)
    scores, provenance = raw_scores(episodes, *scores_dir)
    if provenance != frozen['provenance']:
        raise RuntimeError('Checkpoints differ from the frozen selection')
    rows = {}
    for name, item in [('b_candidate', frozen['b_candidate']), ('ensemble_comparator', frozen['ensemble_comparator'])] + \
            [(f, None if v is None else {'family': f, **v}) for f, v in frozen['references'].items()]:
        if item is None:
            rows[name] = None
            continue
        flags, upper = family_flags(item['family'], item['parameter'], records, scores, item.get('offsets'))
        rows[name] = {'family': item['family'], 'parameter': item['parameter'],
                      **fr.summarize_setting(records, flags), 'bound': fr.bound_quality(records, upper)}
    b, c = rows['b_candidate'], rows['ensemble_comparator']
    only_b = sorted(set(b['detected_ids']) - set(c['detected_ids']))
    only_c = sorted(set(c['detected_ids']) - set(b['detected_ids']))
    gate = {'net_additional_detections': len(only_b) - len(only_c), 'only_b': only_b, 'only_ensemble': only_c,
            'b_false_alarm_episodes': b['false_alarm_episodes'],
            'ensemble_false_alarm_episodes': c['false_alarm_episodes']}
    gate['advance'] = gate['net_additional_detections'] >= 4 and b['false_alarm_episodes'] <= c['false_alarm_episodes']
    write(out, {'git': git_revision(ROOT), 'command': sys.argv, 'frozen': str(args.frozen),
                'episodes': len(records), 'gate': gate, 'rows': rows})
    print(json.dumps(gate, indent=1))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('fit-ensemble')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs=5, default=[1, 2, 3, 4, 5])
    p.add_argument('--updates', type=int, default=890)
    p.add_argument('--microbatch', type=int, default=512)
    p = sub.add_parser('fit-head')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--quantile', type=float, default=0.95)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=0)
    p = sub.add_parser('select')
    p.add_argument('--ensemble', type=Path, required=True)
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('test')
    p.add_argument('--frozen', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    {'fit-ensemble': fit_ensemble, 'fit-head': fit_head, 'select': select, 'test': test}[args.command](args)


if __name__ == '__main__':
    main()
