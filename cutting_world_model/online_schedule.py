"""Environment-step update cadence from SuperMimic-Reproduced Trainer.run."""
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
        self.checkpoint_due = False
        self.update_iterations = []

    def state_dict(self):
        return {'unit': 'vector_steps_v1', 'worlds': self.worlds,
                'iteration': self.iteration, 'transitions': self.transitions,
                'replay_rows_per_world': self.replay_capacity}

    @classmethod
    def resume(cls, saved, worlds):
        if saved.get('unit') != 'vector_steps_v1':
            raise ValueError('Checkpoint lacks the vector-step schedule')
        if saved['worlds'] != worlds:
            raise ValueError('Checkpoint environment count differs')
        return cls(start_iteration=saved['iteration'], worlds=worlds,
                   start_transitions=saved['transitions'])

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
