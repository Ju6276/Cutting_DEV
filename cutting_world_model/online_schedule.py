"""Environment-step update cadence from SuperMimic-Reproduced Trainer.run."""
from dataclasses import asdict
import math
import warnings

from third_party.supermimic.schedule_config import ScheduleCfg


class OnlineSchedule:
    replay_capacity = 1000  # Source rows per environment.

    def __init__(self, config=None, start_iteration=0, worlds=4096,
                 start_transitions=None):
        if not isinstance(worlds, int) or worlds < 1:
            raise ValueError('worlds must be a positive integer')
        self.config = config or ScheduleCfg()
        self.worlds = worlds
        self.iteration = start_iteration
        self.transitions = (start_iteration * worlds if start_transitions is None
                            else start_transitions)
        if self.transitions < 0 or start_iteration < 0:
            raise ValueError('Schedule counters must be nonnegative')
        self.start_iteration = start_iteration
        self.accumulator = 0.
        self.resume_provenance = "fresh"
        self.checkpoint_due = False
        self.update_iterations = []

    def state_dict(self):
        return {'unit': 'vector_steps_v1', 'state_version': 2, 'worlds': self.worlds,
                'iteration': self.iteration, 'transitions': self.transitions,
                'replay_rows_per_world': self.replay_capacity,
                'config': asdict(self.config), 'warmup_start_iteration': self.start_iteration,
                'accumulator': self.accumulator, 'resume_provenance': self.resume_provenance}

    @classmethod
    def resume(cls, saved, worlds, *, replay_restored=True):
        if saved.get('unit') != 'vector_steps_v1':
            raise ValueError('Checkpoint lacks the vector-step schedule')
        if saved['worlds'] != worlds:
            raise ValueError('Checkpoint environment count differs')
        version = saved.get('state_version', 1)
        if version not in (1, 2):
            raise ValueError('Unsupported schedule state version')
        config = ScheduleCfg(**saved['config']) if version == 2 else ScheduleCfg()
        result = cls(config=config, start_iteration=saved['iteration'], worlds=worlds,
                     start_transitions=saved['transitions'])
        if version == 2:
            anchor = saved['warmup_start_iteration']
            accumulator = saved['accumulator']
            if not isinstance(anchor, int) or not 0 <= anchor <= result.iteration:
                raise ValueError('Invalid warmup start iteration')
            if not math.isfinite(accumulator) or not 0 <= accumulator < 1:
                raise ValueError('Invalid schedule accumulator')
            result.start_iteration = anchor
            result.accumulator = float(accumulator)
            result.resume_provenance = saved.get('resume_provenance', 'saved_v2')
        elif replay_restored:
            # Old files lost the prior warmup anchor and fractional update phase.
            # Do not invent an exact phase, especially after earlier resumes.
            warnings.warn('Legacy schedule lacks warmup/accumulator/config state: using '
                          'source defaults, original warmup origin 0 and fractional phase 0. '
                          'This is a documented migration, not exact cadence recovery.',
                          RuntimeWarning, stacklevel=2)
            result.start_iteration = 0
            result.resume_provenance = 'legacy_defaults_origin0_phase0'
        if not replay_restored:
            # A deliberately empty replay needs data before any sampled updates.
            result.start_iteration = result.iteration
            result.accumulator = 0.
            result.resume_provenance = 'empty_replay_refill'
        return result

    def after_environment_step(self):
        self.iteration += 1
        self.transitions += self.worlds
        updates = 0
        self.checkpoint_due = self.iteration % self.config.checkpoint_interval == 0
        self.update_iterations = []
        if self.iteration - self.start_iteration >= self.config.warmup_iterations:
            self.accumulator += self.config.updates_per_env_step
            while self.accumulator >= 1.:
                self.accumulator -= 1.
                updates += 1
                self.update_iterations.append(self.iteration)
        return updates
