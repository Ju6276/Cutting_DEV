"""Content validation of recorded short-run evidence; no simulation or training."""
import json, math, sys
from pathlib import Path

def finite_tree(value):
    import torch
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all()) if value.is_floating_point() or value.is_complex() else True
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def finite_learner(learner):
    # Exact saved names, not a substring filter that silently omits world_model.
    names = ('world_model', 'policy', 'critic')
    return all(isinstance(learner.get(k), dict) and bool(learner[k]) for k in names) and finite_tree(learner)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Validate explicit entry-check stages; missing evidence fails.')
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('label')
    parser.add_argument('--backends', nargs='+', choices=('disect','blademaster'), default=['disect','blademaster'])
    parser.add_argument('--checks', nargs='+', choices=('pose_check','train_wm','train_sac','eval_wm_policy','eval_sac','eval_wm_mppi','eval_analytic_mpc','diagnose','resume_compare'),
                        default=['pose_check','train_wm','train_sac','eval_wm_policy','eval_sac','eval_wm_mppi','eval_analytic_mpc','diagnose','resume_compare'])
    args = parser.parse_args()
    R = args.run_dir; out = {}
    def finite(v): return not isinstance(v, float) or math.isfinite(v)
    def put(name, ok, info): out[name] = dict(ok=bool(ok), **info)
    for B in args.backends:
        p = R / f'pose_check_{B}.json'
        if p.exists():
            r = json.load(open(p))
            put(p.stem, r.get('status') == 'passed' and r.get('wm_reward_error_max', 1) < 1e-3 and r.get('masked_reset_unselected_exact') is True,
                dict(status=r.get('status'), wm_reward_error_max=r.get('wm_reward_error_max'), masked_reset_exact=r.get('masked_reset_unselected_exact')))
        w = R / f'wm_{B}'
        if (w / "updates.jsonl").exists() and (w / "last.pt").exists():  # copied checkpoint dirs lack logs
            import torch
            pr = json.load(open(w / 'progress.json'))
            ck = torch.load(w / 'last.pt', map_location='cpu', weights_only=False)
            rp = torch.load(w / 'replay.pt', map_location='cpu', weights_only=False)
            opt = pr.get('optimizer_steps', {})
            bad = []
            for line in open(w / 'updates.jsonl'):
                for k, v in json.loads(line).items():
                    if any(s in k for s in ('loss', 'grad_norm', 'objective', 'critic_value')) and not finite(v): bad.append(k)
            ver = ck['metadata'].get('free_cut_task', {}).get('version')
            put(f'train_wm_{B}', pr.get('state') == 'completed' and pr.get('transitions', 0) >= 720 and pr.get('wm_updates', 0) > 0
                and pr.get('policy_updates', 0) > 0 and all(opt.get(k, 0) > 0 for k in ('world_model', 'policy', 'critic'))
                and rp.get('checkpoint') == 'last.pt' and rp.get('iteration') == ck['schedule']['iteration'] and not bad and ver == 'pose_depth_v3' and finite_learner(ck['learner']),
                dict(state=pr.get('state'), transitions=pr.get('transitions'), wm_updates=pr.get('wm_updates'), policy_updates=pr.get('policy_updates'),
                     optimizer_steps=opt, replay_paired=rp.get('checkpoint') == 'last.pt' and rp.get('iteration') == ck['schedule']['iteration'],
                     nonfinite_update_metrics=sorted(set(bad)), task_version=ver, learner_finite=finite_learner(ck['learner'])))
        s = R / f'sac_{B}'
        if (s / 'progress.json').exists():
            pr = json.load(open(s / 'progress.json'))
            put(f'train_sac_{B}', pr.get('state') == 'completed' and pr.get('transitions') == 240 and (s / 'model.zip').exists(),
                dict(state=pr.get('state'), transitions=pr.get('transitions'), model=(s / 'model.zip').exists()))
        for m in ('wm_policy', 'sac', 'wm_mppi', 'analytic_mpc'):
            e = R / f'eval_{m}_{B}'
            if (e / 'summary.json').exists():
                sm = json.load(open(e / 'summary.json')); pr = json.load(open(e / 'progress.json'))
                ov = sm.get('overall', {})
                nums = [v for v in ov.values() if isinstance(v, float)]
                put(e.name, sm.get('state') == 'completed' and pr.get('cases_done') == pr.get('expected_cases') == 3 and all(map(finite, nums)),
                    dict(state=sm.get('state'), cases_done=pr.get('cases_done'), expected=pr.get('expected_cases'),
                         overall={k: v for k, v in ov.items() if not isinstance(v, (dict, list))}))
        d = R / f'diagnose_{B}'
        if d.exists():
            files = sorted(x.name for x in d.iterdir())
            summary = json.load(open(d / 'summary.json')) if 'summary.json' in files else {}
            required = {'summary.json', 'records.pt', 'transitions.npz', 'steps.npz'}
            worlds = summary.get('worlds', 0)
            valid = (required.issubset(files) and worlds > 0 and summary.get('task_version') == 'pose_depth_v3'
                     and summary.get('episodes', {}).get('overall', {}).get('episodes') == worlds
                     and summary.get('overall', {}).get('transitions', 0) > 0 and finite_tree(summary))
            put(d.name, valid, dict(files=files, worlds=worlds, episodes=summary.get('episodes', {}).get('overall'),
                                   transitions=summary.get('overall', {}).get('transitions')))
        c = R / f'resume_compare_{B}.json'
        if c.exists():
            r = json.load(open(c))
            put(c.stem, r.get('passed') is True, dict(checks=r.get('checks'), pending_rounds_at_save=r.get('pending_rounds_at_save'),
                                                      pre_transitions=r.get('pre_transitions'), post_transitions=r.get('post_transitions')))
    required = [f'{stage}_{backend}' for backend in args.backends for stage in args.checks]
    for name in required:
        if name not in out:
            put(name, False, {'error': 'Required evidence missing'})
    out = {name: out[name] for name in required}
    json.dump(out, open(R / f'verify_{Path(args.label).name}.json', 'w'), indent=1, default=str)
    for k, v in out.items():
        print('OK  ' if v['ok'] else 'BAD ', k, {a: b for a, b in v.items() if a != 'ok'})
    
    sys.exit(0 if out and all(v['ok'] for v in out.values()) else 1)


if __name__ == "__main__":
    main()
