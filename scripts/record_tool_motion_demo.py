"""Record one physical scripted rollout and replay both tool-position rules.

The final-only task drives the actual simulation; the path-mode panel is stopped
at its first terminal transition. This is a rule demonstration, not two learned
policies or a claim about physical braking after an RL episode terminates.
"""
import argparse, copy, json, signal, time
from pathlib import Path
import torch
from cutting_world_model.shared_scene import load_configs
from cutting_world_model.multi_env import MultiConditionEnvironment
from cutting_world_model.free_cut import FreeCutTask
from cutting_world_model.training_snapshot import capture_training_environment
from cutting_world_model.provenance import git_revision
from scripts.evaluate_pose_policy import evaluation_config

def excursion(t, amplitude=.075, ramp_seconds=1.5):
    if t <= ramp_seconds:
        return amplitude * t / ramp_seconds
    return amplitude * max(0., 2.-t/ramp_seconds)

def ending(ev):
    for key in ('collision','outside','tool_motion_failure','success','timeout'):
        if bool(ev.get(key,torch.tensor([False]))[0]): return key
    return 'running'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shared-contract',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=900)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    def expire(*_):raise TimeoutError('Position demo wall budget')
    signal.signal(signal.SIGALRM,expire);signal.alarm(a.seconds)
    torch.set_num_threads(2);torch.manual_seed(0)
    cfg,_=load_configs(a.shared_contract,a.prepared)
    cfg=evaluation_config(cfg,1,0)
    cfg['groups']=cfg['groups'][:1]
    cfg['parallel']=False
    cfg['task']['tool_motion']['enabled']=False
    report=dict(status='running',git=git_revision(Path.cwd()),config=cfg,
        scope='One real scripted reference rollout; alternative task-rule replay, not policy training',
        reference=dict(x_amplitude_m=.075,ramp_seconds=1.5,return_to_start=True),
        rows=[],modes={})
    (a.output/'config.json').write_text(json.dumps(cfg,indent=2))
    env=None;start=time.monotonic()
    try:
        env=MultiConditionEnvironment.from_config('backends/mujoco',cfg)
        inner=env.members[0].env
        original=inner.physics.reference_target
        def scripted(t):
            target=original(t)
            target[:,0]+=excursion(float(t[0]))
            return target
        inner.physics.reference_target=scripted
        frame=env.current
        rules={}
        counters={}
        totals={'final_only':0.,'path_and_final':0.}
        stops={}
        for name,enabled in [('final_only',False),('path_and_final',True)]:
            task=copy.deepcopy(cfg['task']);task['tool_motion']['enabled']=enabled
            rules[name]=FreeCutTask(task,env.heads)
            counters[name]=inner.success_counter.clone()
        def capture(step):
            capture_training_environment(env,Path('backends/mujoco'),
                a.output/'geometry'/('%04d'%step),
                dict(iteration=step,training_stage='scripted_rule_demo',policy_updates=None,
                     seed=0,ee_pose_base=frame['ee_pose'].cpu().tolist()))
        report['conditions']=env.conditions([0])
        capture(0)
        for step in range(1,cfg['task']['horizon_steps']+1):
            previous=frame['cut_progress'][:,0].clone()
            frame,reward,done,trunc,info=env.step(torch.zeros(1,6,device=env.device),reference=True)
            raw=inner.current
            capture(step)
            row=dict(step=step,time_s=step*.05,
                     ee_pose=frame['ee_pose'][0].cpu().tolist(),
                     target_pose=frame['target_pose'][0].cpu().tolist(),
                     distance_m=float(frame['completion_xy_error'][0,0]),
                     depth_progress=float(frame['cut_progress'][0,0]),
                     peak_ratio=float(info['safety_margin'][0]),
                     physical_reward=float(reward[0]),modes={})
            for name,task in rules.items():
                if name in stops:
                    row['modes'][name]=dict(active=False,return_total=totals[name],ending=stops[name]['ending'])
                    continue
                ev=task.events(frame['observation'],frame['peak_wrench'],None,
                    frame['step_count'][:,0],info['collision'],
                    pose=frame['ee_pose'],target_pose=frame['target_pose'],board=raw,
                    success_counter=counters[name])
                r=task.reward(previous,frame['cut_progress'][:,0],frame['observation'],
                    frame['peak_wrench'],frame['target_speed'][:,0],ev)
                counters[name]=ev['counter']
                if name=='final_only':
                    torch.testing.assert_close(r.to(reward),reward,atol=2e-6,rtol=1e-7)
                    assert torch.equal(ev['terminated'],done)
                totals[name]+=float(r[0])
                from cutting_world_model.motion_limits import penalty
                cost=float(penalty(ev['completion_xy_error_m'],task.config['tool_motion'],task.horizon)[0]) if name=='path_and_final' else 0.
                row['modes'][name]=dict(active=True,reward=float(r[0]),return_total=totals[name],
                                       position_cost=cost,ending=ending(ev))
                if ev['terminated'][0]:stops[name]=dict(step=step,ending=ending(ev),return_total=totals[name])
            report['rows'].append(row)
            print(step,round(row['distance_m']*1000,2),{k:v['ending'] for k,v in row['modes'].items()},flush=True)
            if bool(done[0] or trunc[0]):break
        report.update(status='complete',modes=stops,steps=step)
        (a.output/'episodes.jsonl').write_text(json.dumps(dict(case=0,steps=step,scope=report['scope']))+'\n')
        if 'path_and_final' not in stops or stops['path_and_final']['ending']!='tool_motion_failure':
            raise RuntimeError('Script did not demonstrate a tool-motion terminal event; inspect captured result')
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        if env is not None:env.close()
        report['wall_seconds']=time.monotonic()-start
        (a.output/'comparison.json').write_text(json.dumps(report,indent=2))
        signal.alarm(0)
if __name__=='__main__':main()
