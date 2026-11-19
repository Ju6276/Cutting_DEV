"""Validate a non-wrapped short-run checkpoint/resume pair without changing artifacts."""
import json, math, sys
from pathlib import Path
import torch
from scripts.verify_entry import finite_learner


def main():
    pre, post, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ld = lambda p: torch.load(p, map_location=dev, weights_only=False)
    a, b = ld(pre / 'last.pt'), ld(post / 'last.pt')
    ra, rb = ld(pre / 'replay.pt'), ld(post / 'replay.pt')
    pa, pb = json.load(open(pre / 'progress.json')), json.load(open(post / 'progress.json'))
    sa, sb = a['schedule'], b['schedule']
    qa = a.get('update_queue') or {}
    r = {'pre_state': pa.get('state'), 'post_state': pb.get('state'),
         'pre_transitions': sa['transitions'], 'post_transitions': sb['transitions'],
         'warmup_start_pre': sa.get('warmup_start_transitions'), 'warmup_start_post': sb.get('warmup_start_transitions'),
         'resume_provenance_post': sb.get('resume_provenance'),
         'pending_rounds_at_save': len(qa.get('pending', [])) + (qa.get('active') is not None),
         'pre_updates': pa.get('updates'), 'post_updates': pb.get('updates'),
         'pre_wm_updates': pa.get('wm_updates'), 'post_wm_updates': pb.get('wm_updates'),
         'pre_policy_updates': pa.get('policy_updates'), 'post_policy_updates': pb.get('policy_updates'),
         'pre_optimizer_steps': pa.get('optimizer_steps'), 'post_optimizer_steps': pb.get('optimizer_steps')}
    # Replay pairing of the final saves.
    r['pre_replay_pairs_last'] = ra.get('checkpoint') == 'last.pt' and ra.get('iteration') == a['schedule']['iteration']
    r['post_replay_pairs_last'] = rb.get('checkpoint') == 'last.pt' and rb.get('iteration') == b['schedule']['iteration']
    # Pre-resume rows are restored exactly; the first post-resume row of every world starts a new episode.
    ta, tb = ra['buffer']['tensors'], rb['buffer']['tensors']
    worlds = ta['ptr'].shape[0]
    # This prefix comparison only proves non-wrapped short-run replay restoration.
    # Refuse a wrapped/full ring rather than reporting a misleading prefix identity.
    if any(int(rp['buffer']['rows']) >= int(rp['buffer']['capacity']) for rp in (ra, rb)):
        raise ValueError('Prefix check requires pre-resume replay below capacity')
    exact, boundary_new_episode, boundary_flag = True, [], []
    for w in range(worlds):
        k = int(ta['ptr'][w])
        if k <= 0:
            raise ValueError('Every checked world must have pre-resume transitions')
        for name in ('s', 'a', 'a_exec', 'xi', 'r', 'done', 'frame', 'clip', 'ep_id', 'terminal', 'ep_start', 'valid'):
            if not torch.equal(ta[name][w, :k], tb[name][w, :k]):
                exact = False
        if int(tb['ptr'][w]) > k:
            boundary_new_episode.append(bool(tb['ep_id'][w, k] != ta['ep_id'][w, k - 1]))
            boundary_flag.append(bool(tb['ep_start'][w, k]))
    r['pre_rows_restored_exactly'] = exact
    r['boundary_new_episode_all_worlds'] = all(boundary_new_episode) and len(boundary_new_episode) == worlds
    r['boundary_ep_start_all_worlds'] = all(boundary_flag) and len(boundary_flag) == worlds
    # Finite parameters and logged learning metrics after resume.
    r['post_learner_params_finite'] = finite_learner(b['learner'])
    bad = []
    for line in open(post / 'updates.jsonl'):
        row = json.loads(line)
        for k, v in row.items():
            if any(s in k for s in ('loss', 'grad_norm', 'reward', 'return', 'objective')) and isinstance(v, float) and not math.isfinite(v):
                bad.append(k)
    r['post_update_metrics_nonfinite'] = sorted(set(bad))
    r['post_update_rows'] = sum(1 for _ in open(post / 'updates.jsonl'))
    checks = {
      'post_completed': r['post_state'] == 'completed',
      'transitions_continue': r['post_transitions'] > r['pre_transitions'],
      'no_rewarmup': r['warmup_start_pre'] == r['warmup_start_post'] and r['resume_provenance_post'] != 'empty_replay_refill',
      'updates_continue': (r['post_wm_updates'] or 0) > (r['pre_wm_updates'] or 0) and (r['post_policy_updates'] or 0) > (r['pre_policy_updates'] or 0),
      'replay_restored_exactly': r['pre_rows_restored_exactly'],
      'new_episode_at_resume': r['boundary_new_episode_all_worlds'] and r['boundary_ep_start_all_worlds'],
      'resaved_paired': r['pre_replay_pairs_last'] and r['post_replay_pairs_last'],
      'finite': r['post_learner_params_finite'] and not r['post_update_metrics_nonfinite'],
    }
    r['checks'] = checks
    r['pending_round_path_exercised'] = r['pending_rounds_at_save'] > 0
    r['passed'] = all(checks.values())
    json.dump(r, open(out, 'w'), indent=1, default=str)
    print(json.dumps({'passed': r['passed'], 'checks': checks, 'pending_rounds_at_save': r['pending_rounds_at_save']}))
    sys.exit(0 if r['passed'] else 1)


if __name__ == "__main__":
    main()
