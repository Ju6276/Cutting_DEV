"""Stable-Baselines3 VecEnv bridge for the actual batched cutting environment.

All physical worlds contribute transitions. Terminal observations are captured
BEFORE masked reset; finite-horizon task timeouts are terminals, not truncations.
No material properties or particle state enter the actor/critic observation.
"""
import time
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import SAC


def observation(history, goal):
    # Same ten observable frames and commanded goal available to WM+BPTT actor.
    return torch.cat((history.permute(1,0,2).flatten(1),goal),-1).cpu().numpy().astype(np.float32)


class CuttingVecEnv(VecEnv):
    def __init__(self, env, history=10):
        if type(history) is not int or history < 1:
            raise ValueError('Positive history length required')
        self.env = env
        self.history_length = history
        self.render_mode = None
        self.closed = False
        self.pending = None
        self.steps = np.zeros(env.worlds,dtype=np.int64)
        self.returns = np.zeros(env.worlds,dtype=np.float64)
        self.history = env.current['observation'].float()[None].repeat(history,1,1)
        dim = observation(self.history,env.goal(env.current)).shape[-1]
        super().__init__(env.worlds, spaces.Box(-np.inf,np.inf,(dim,),np.float32),
                         spaces.Box(-1.,1.,(6,),np.float32))
        self._first_reset = True

    def reset(self):
        # Construction already performs randomized initial settling.
        if not self._first_reset:
            self.env.reset(torch.ones(self.num_envs,dtype=torch.bool,device=self.env.device))
        self._first_reset = False
        self.history = self.env.current['observation'].float()[None].repeat(self.history_length,1,1)
        self.steps[:] = 0
        self.returns[:] = 0
        self._reset_seeds()
        self._reset_options()
        return observation(self.history,self.env.goal(self.env.current))

    def step_async(self, actions):
        if self.pending is not None:
            raise RuntimeError('Pending vector step has not been consumed')
        actions = np.asarray(actions,dtype=np.float32)
        if actions.shape != (self.num_envs,6) or not np.isfinite(actions).all():
            raise ValueError('Finite six-axis action per world required')
        self.pending = torch.as_tensor(actions,device=self.env.device).clamp(-1,1)

    def step_wait(self):
        if self.pending is None:
            raise RuntimeError('step_async must precede step_wait')
        action,self.pending=self.pending,None
        frame,reward,terminated,truncated,info=self.env.step(action)
        if bool(truncated.any()):
            raise ValueError('Current finite-horizon cutting task must terminate, not truncate')
        if not bool(torch.isfinite(reward).all() & torch.isfinite(frame['observation']).all()):
            raise FloatingPointError('Nonfinite vector transition')
        self.history=torch.cat((self.history[1:],frame['observation'].float()[None]),0)
        terminal_obs=observation(self.history,self.env.goal(frame))
        done=terminated.cpu().numpy().astype(bool)
        rewards=reward.cpu().numpy().astype(np.float32)
        self.steps+=1
        self.returns+=rewards
        infos=[]
        for w in range(self.num_envs):
            row={k:v[w].detach().cpu().tolist() for k,v in info.items()
                 if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==self.num_envs}
            row['TimeLimit.truncated']=False
            if done[w]:
                row['terminal_observation']=terminal_obs[w].copy()
                row['episode']={'r':float(self.returns[w]),'l':int(self.steps[w])}
            infos.append(row)
        if bool(terminated.any()):
            frame=self.env.reset(terminated)
            self.history[:,terminated]=frame['observation'][terminated].float()[None]
            self.steps[done]=0
            self.returns[done]=0
        return observation(self.history,self.env.goal(frame)),rewards,done,infos

    def close(self):
        if not self.closed:
            self.closed=True
            self.env.close()

    def get_attr(self,attr_name,indices=None):
        return [getattr(self,attr_name) for _ in self._get_indices(indices)]

    def set_attr(self,attr_name,value,indices=None):
        raise NotImplementedError('Configure physics before construction; per-world mutation is unsupported')

    def env_method(self,method_name,*args,indices=None,**kwargs):
        raise NotImplementedError('Use batched cutting API directly')

    def env_is_wrapped(self,wrapper_class,indices=None):
        return [False for _ in self._get_indices(indices)]

    def get_images(self):
        raise NotImplementedError('Record shared physical geometry; no alternate renderer')


class SACWallBudgetReached(RuntimeError):
    """Stop between complete rollout/update rounds, never inside replay insertion."""


class TransitionSAC(SAC):
    """Explicit optimizer cadence per newly collected physical transition.

    SB3 standard SAC equations are untouched. Credits below learning_starts are
    not converted into a burst of retroactive updates. Residual credit is saved
    in the regular SB3 model archive.
    """
    def __init__(self,*args,transitions_per_gradient=1,**kwargs):
        if type(transitions_per_gradient) is not int or transitions_per_gradient < 1:
            raise ValueError('Positive transition/update interval required')
        self.transitions_per_gradient=transitions_per_gradient
        self.transition_credit=0
        self.wall_deadline=None
        super().__init__(*args,**kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['wall_deadline']

    def _wall_budget_reached(self):
        return self.wall_deadline is not None and time.monotonic() >= self.wall_deadline

    def collect_rollouts(self,*args,**kwargs):
        if self._wall_budget_reached():
            raise SACWallBudgetReached()
        before=self.num_timesteps
        result=super().collect_rollouts(*args,**kwargs)
        eligible=max(0,self.num_timesteps-max(before,self.learning_starts))
        self.transition_credit+=eligible
        self.gradient_steps,self.transition_credit=divmod(
            self.transition_credit,self.transitions_per_gradient)
        return result
