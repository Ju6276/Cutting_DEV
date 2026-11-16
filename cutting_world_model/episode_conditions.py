"""Independent per-world episode streams shared by both physical backends."""
import copy
import math
import random
import numpy as np

REVISION = "paired_episode_conditions_v1"
KEYS = ("youngs_modulus", "density", "cut_strength", "target_speed")

def validate(config):
    if not isinstance(config, dict) or set(config) != {"revision", "seed", *KEYS}:
        raise ValueError("Explicit episode condition revision, seed and ranges required")
    if config["revision"] != REVISION or type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("Unsupported episode sampling revision or seed")
    for key in KEYS:
        bounds = config[key]
        if (not isinstance(bounds, (tuple, list)) or len(bounds) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) for x in bounds)
                or not 0 < bounds[0] <= bounds[1]):
            raise ValueError("Positive finite ordered episode ranges required")
    return copy.deepcopy(config)

class EpisodeConditions:
    def __init__(self, config, groups):
        self.config = validate(config)
        self.ids = [f"{g['name']}/{i}" for g in groups for i in range(g["worlds"])]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Unique object names required")
        self.next_episode = np.zeros(len(self.ids), dtype=np.int64)
        self.values = {k: np.zeros(len(self.ids), dtype=np.float64) for k in KEYS}

    def sample(self, mask):
        selected = np.asarray(mask)
        if selected.shape != self.next_episode.shape or not np.isin(selected, [0, 1]).all():
            raise ValueError("One reset flag per world required")
        for i in np.flatnonzero(selected):
            seed = f"{self.config['seed']}:{self.ids[i]}:{self.next_episode[i]}:{REVISION}"
            rng = random.Random(seed)
            for key in KEYS:
                low, high = self.config[key]
                u = rng.random()
                self.values[key][i] = (low + u * (high-low) if key == "target_speed"
                                      else math.exp(math.log(low)+u*(math.log(high)-math.log(low))))
            self.next_episode[i] += 1
        return {k:v.copy() for k,v in self.values.items()}

    def state_dict(self):
        return dict(config=copy.deepcopy(self.config), ids=list(self.ids),
                    next_episode=self.next_episode.tolist(),
                    values={k:v.tolist() for k,v in self.values.items()})

    def load_state_dict(self, state):
        if state["config"] != self.config or state["ids"] != self.ids:
            raise ValueError("Episode sampler identity differs from checkpoint")
        counts = np.asarray(state["next_episode"])
        if counts.shape != self.next_episode.shape or counts.dtype.kind not in "iu" or (counts<0).any():
            raise ValueError("Invalid episode counters")
        values={k:np.asarray(state["values"][k],dtype=np.float64) for k in KEYS}
        for key,value in values.items():
            low,high=self.config[key]
            if (value.shape != counts.shape or not np.isfinite(value).all()
                    or (((value < low-1e-12)|(value > high+1e-12)) & (counts>0)).any()):
                raise ValueError("Invalid saved episode conditions")
        self.next_episode=counts.astype(np.int64)
        self.values=values

def metadata(config):
    return {k:v for k,v in validate(config).items() if k != "seed"}

def resume_environment(env, state):
    """Resume condition streams, starting fresh physical episodes as the trainer does."""
    sampler = getattr(env, "condition_sampler", None)
    if sampler is None:
        return env.current
    if state is None:
        raise ValueError("Randomized resume requires saved episode sampling state")
    sampler.load_state_dict(state)
    return env.reset()
