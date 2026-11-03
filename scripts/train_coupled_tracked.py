"""Run the source connected trainer and mirror its JSON measurements to W&B."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cutting_world_model.sidecar_tracking import SidecarTracker
from cutting_world_model.lifecycle import finalize_training

spec = importlib.util.spec_from_file_location('source_train_coupled', ROOT / 'scripts/train_coupled.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
write_json = source.write_json
append_json = source.append_json
tracker = None
output = None
completed_episodes = 0
successful_episodes = 0
last_status = {}
phase_codes = {'physics': 0, 'world_model': 1, 'policy': 2, 'critic': 3}


def tracked_write(path, data):
    global tracker, output, last_status
    write_json(path, data)
    if path.name == 'manifest.json':
        output = path.parent
        tracker = SidecarTracker(output.parent, dict(data, run_number=2,
            algorithm='SuperMimic-Reproduced',
            model_state=data.get('arguments', {}).get('model_state'),
            execution_budget_hours=data.get('arguments', {}).get('max_hours'),
            iteration_endpoint=data.get('schedule', {}).get('total_iterations')), job_type='training')
        print(json.dumps({'wandb': tracker.info}), flush=True)
    elif path.name == 'progress.json' and tracker is not None:
        last_status = dict(data)
        values = {'iteration': data['iteration'], 'runtime/wall_seconds': data['wall_seconds'],
                  'runtime/updates': data['updates']}
        if 'phase' in data:
            values['runtime/phase_code'] = phase_codes[data['phase']]
        for name in ('minibatch', 'minibatches', 'samples', 'batch_size', 'epoch', 'epochs'):
            if name in data:
                values['runtime/' + name] = data[name]
        values.update(learning_metrics(data))
        tracker.log(values)


def learning_metrics(data):
    values = {}
    for key in ('wm_updates', 'policy_updates', 'wm_only_updates', 'training_stage'):
        if key in data:
            values['learning/' + key] = data[key]
    for key, value in data.get('optimizer_steps', {}).items():
        values['optimizer/' + key + '_steps'] = value
    return values


def tracked_append(path, data):
    global completed_episodes, successful_episodes
    append_json(path, data)
    if tracker is None:
        raise RuntimeError('Training metrics arrived before tracker initialization')
    values = {'iteration': data['iteration']}
    if 'update_iteration' in data:
        values['learning/update_iteration'] = data['update_iteration']
    if path.name == 'steps.jsonl':
        values.update({'reward': data['last_reward'], 'reward/step': data['last_reward'],
                       'physics/reward_mean': data['last_reward'],
                       'reward/episode_running': data['episode_return'],
                       'physics/episode_step': data['episode_step'],
                       'physics/transitions': data['transitions']})
    elif path.name == 'episodes.jsonl':
        completed_episodes += 1
        successful_episodes += int(data['success'])
        values.update({'reward/episode': data['return'], 'reward/episode_return': data['return'],
                       'success_rate': successful_episodes / completed_episodes,
                       'physics/episode_success': int(data['success']),
                       'physics/episode_collision': int(data['collision']),
                       'physics/episode_truncated': int(data['truncated']),
                       'physics/episode_length': data['steps'],
                       'physics/episodes': completed_episodes,
                       'physics/released_springs': data['released_springs']})
    elif path.name == 'updates.jsonl':
        values.update({'learning/update': data['update'], 'learning/lr': data['lr'],
                       'runtime/peak_gpu_mib': data['peak_gpu_mib']})
        for group in ('world_model', 'policy', 'timing'):
            values.update({group + '/' + key: value for key, value in data[group].items()})
    values.update(learning_metrics(data))
    tracker.log(values)


source.write_json = tracked_write
source.append_json = tracked_append
error = None
try:
    source.main()
except BaseException as exc:
    error = exc
    raise
finally:
    status = dict(last_status)
    if error is not None:
        primary = status.get('error')
        if primary is None and error.__context__ is not None:
            primary = repr(error.__context__)
        if primary is not None:
            status['primary_error'] = primary
        status.update(state='failed', exit_error=repr(error))
    if output is not None:
        status['checkpoint_path'] = str(output / 'last.pt')
    status['completed_episodes'] = completed_episodes
    status['successful_episodes'] = successful_episodes

    def record_final_status():
        if output is not None:
            write_json(output / 'progress.json', status)

    errors = finalize_training(
        save=lambda: None,
        close=lambda: None,
        record=record_final_status,
        finish=lambda: tracker.finish(status=status) if tracker is not None else None,
        state=status)
    if errors and error is None:
        raise RuntimeError('Training finalization failed: ' + repr(errors))
