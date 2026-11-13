#!/usr/bin/env python3
"""Pair the two frozen-policy runs without presenting a task change as learning."""
import argparse,json
from pathlib import Path
FIELDS=('youngs_modulus','cut_spring_stiffness','density','target_speed_mps','start_lift_m')
def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--historical',type=Path)
 a=p.parse_args()
 def read(name):
  rows=[json.loads(x) for x in (a.root/name/'episodes.jsonl').read_text().splitlines()]
  return {r['world']:r for r in rows}
 old,new=read('legacy64'),read('board64')
 if set(old)!=set(new):raise ValueError('Different case sets')
 pairs=[]
 for i in sorted(old):
  x,y=old[i],new[i]
  if any(abs(x[k]-y[k])>1e-6 for k in FIELDS):raise ValueError('Different sampled conditions')
  pairs.append(dict(world=i,old_ending=x['ending'],new_ending=y['ending'],
                    old_min_height_mm=1000*x['min_knife_height_m'],new_min_height_mm=1000*y['min_knife_height_m'],
                    old_release=x['released_fraction'],new_release=y['released_fraction'],
                    new_path_fraction=y['path_fraction'],old_steps=x['steps'],new_steps=y['steps']))
 summary={'cases':len(pairs),'conditions_match':True,
          'old_success':sum(x['success'] for x in old.values()),
          'new_path_success':sum(x['success'] for x in new.values()),
          'old_success_with_below_board':sum(x['success'] and x['min_knife_height_m'] < -2e-6 for x in old.values()),
          'old_below_board':sum(x['min_knife_height_m'] < -2e-6 for x in old.values()),
          'new_below_board':sum(x['min_knife_height_m'] < -2e-6 for x in new.values()),
          'old_timeouts':sum(x['ending']=='timeout' for x in old.values()),
          'old_timeout_to_path_success':sum(x['old_ending']=='timeout' and x['new_ending']=='success' for x in pairs),
          'old_success_to_new_incomplete':sum(x['old_ending']=='success' and x['new_ending']!='success' for x in pairs),
          'new_path_success_not_separated':sum(x['success'] and x['released_fraction']<1 for x in new.values()),
          'new_material_separated':sum(x['released_fraction']==1 for x in new.values()),
          'minimum_new_gap_m':min(x['min_knife_height_m'] for x in new.values()),
          'interpretation':'Different tasks and trajectories after old early endings. Not a policy improvement or single-cause proof.'}
 if a.historical:
  hist={r['world']:r for r in json.loads(a.historical.read_text())['per_world']}
  for i,x in old.items():
   if any(abs(x[k]-hist[i][k])>1e-6 for k in FIELDS):raise ValueError('Historical conditions differ')
  summary['historical_success']=sum(x['success'] for x in hist.values())
  summary['legacy_replay_outcome_differences']=sum(old[i]['success']!=hist[i]['success'] for i in old)
  summary['historical_timeout_to_path_success']=sum(hist[i]['timeout'] and new[i]['success'] for i in new)
 (a.root/'comparison.json').write_text(json.dumps({'summary':summary,'pairs':pairs},indent=2))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
