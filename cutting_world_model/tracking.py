"""Experiment metrics and checkpoints in the user's W&B workspace."""
import json
from pathlib import Path


class ExperimentTracker:
    def __init__(self, output, config, project='sliceit-supermimic', entity=None, mode='online', job_type='training'):
        import wandb
        self.wandb = wandb
        self.run = wandb.init(project=project, entity=entity, mode=mode,
                              name=Path(output).name, dir=str(output), job_type=job_type,
                              config=config, save_code=False, settings=wandb.Settings(disable_git=True))
        self.run.define_metric('iteration')
        self.run.define_metric('*', step_metric='iteration')
        info = {key: getattr(self.run, key) for key in ('id', 'url', 'entity', 'project')}
        (Path(output) / 'wandb.json').write_text(json.dumps(info, indent=2) + '\n')

    def log(self, values):
        self.run.log(values)

    def finish(self, checkpoint=None, status=None):
        if status is not None:
            self.run.summary.update(status)
        if checkpoint is not None and Path(checkpoint).exists():
            artifact = self.wandb.Artifact('checkpoint-' + self.run.id, type='model')
            artifact.add_file(str(checkpoint), name=Path(checkpoint).name)
            self.run.log_artifact(artifact)
        self.run.finish(exit_code=1 if status and status.get('state') == 'failed' else 0)
