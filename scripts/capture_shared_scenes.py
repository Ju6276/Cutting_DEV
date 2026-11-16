"""Capture settled shared-tool scenes for both backends, without a policy."""
import argparse, json, signal, time
from pathlib import Path
import torch
from cutting_world_model.training_snapshot import capture_training_environment
from cutting_world_model.provenance import git_revision

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--shared-contract',type=Path,required=True)
 p.add_argument('--prepared',type=Path,required=True)
 p.add_argument('--seconds',type=int,default=900)
 p.add_argument('--backend',choices=['both','disect','blademaster'],default='both')
 a=p.parse_args()
 if a.seconds<=0: raise ValueError('Positive wall budget required')
 def expire(*_): raise TimeoutError('Scene capture budget')
 signal.signal(signal.SIGALRM,expire);signal.alarm(a.seconds)
 a.output.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(2)
 from cutting_world_model.shared_scene import load_configs
 dc,bc=load_configs(a.shared_contract,a.prepared)
 for backend,cfg in [('disect',dc),('blademaster',bc)]:
  if a.backend not in ('both',backend):continue
  out=a.output/backend;out.mkdir()
  (out/'config.json').write_text(json.dumps(cfg,indent=2))
  env=None;started=time.monotonic()
  try:
   torch.manual_seed(0)
   if backend=='disect':
    from cutting_world_model.multi_env import MultiConditionEnvironment
    env=MultiConditionEnvironment.from_config('backends/mujoco',cfg)
   else:
    from cutting_world_model.blademaster_backend import BladeMasterEnvironment
    env=BladeMasterEnvironment('backends/mujoco',cfg)
   status=dict(iteration=0,training_stage='settled_scene_preview',policy_updates=None,
               seed=cfg['episode_conditions']['seed'],ee_pose_base=env.current['ee_pose'].detach().cpu().tolist())
   report=capture_training_environment(env,'backends/mujoco',out/'geometry/0000',status)
   report['scope']='Settled initial scenes; no policy loaded, no control steps, no training'
   (out/'geometry/0000/report.json').write_text(json.dumps(report,indent=2))
   audit=dict(status='complete',git=git_revision(Path.cwd()),config_source=str(a.shared_contract),prepared=str(a.prepared),
              conditions=env.conditions(range(env.worlds)),
              config=cfg,scope=report['scope'],metadata=env.metadata,
              wall_seconds=time.monotonic()-started)
   (out/'manifest.json').write_text(json.dumps(audit,indent=2))
   print(backend,'captured',audit['wall_seconds'],flush=True)
  finally:
   if env is not None:env.close()
 signal.alarm(0)
if __name__=='__main__': main()
