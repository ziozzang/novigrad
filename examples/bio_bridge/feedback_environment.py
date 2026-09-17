#!/usr/bin/env python3
"""Deterministic hidden-goal action-feedback environment and host Bayes baseline.

The belief baseline is explicit host-side inference, not biological learning.
Environment observations never contain the hidden goal or latent state.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np

ACTIONS=4

@dataclass(frozen=True)
class FeedbackConfig:
 actions:int=ACTIONS
 max_steps:int=4
 action_cost:float=.05
 reward_flip_probability:float=0.0
 delay:int=0
 def validate(self):
  if type(self.actions) is not int or self.actions<2:raise ValueError('actions must be an integer >=2')
  if type(self.max_steps) is not int or self.max_steps<=0:raise ValueError('max_steps must be a positive integer')
  if isinstance(self.action_cost,(bool,np.bool_)) or not isinstance(self.action_cost,(int,float,np.integer,np.floating)) or not np.isfinite(self.action_cost) or self.action_cost<0:raise ValueError('action_cost must be finite and nonnegative')
  if isinstance(self.reward_flip_probability,(bool,np.bool_)) or not isinstance(self.reward_flip_probability,(int,float,np.integer,np.floating)) or not np.isfinite(self.reward_flip_probability) or not 0<=self.reward_flip_probability<=1:raise ValueError('reward_flip_probability must be in [0,1]')
  if type(self.delay) is not int or self.delay not in (0,1):raise ValueError('only feedback delays 0 or 1 are supported')
  return self

class HiddenGoalEnvironment:
 def __init__(self,goals,seed=2901,config=FeedbackConfig(),action_permutation=None):
  self.config=config.validate();g=np.asarray(goals)
  if g.ndim!=1 or len(g)==0 or g.dtype.kind not in 'iu' or np.any((g<0)|(g>=config.actions)):raise ValueError('goals must be a nonempty 1D integer sequence')
  if type(seed) is not int:raise ValueError('seed must be an integer')
  permutation=np.arange(config.actions) if action_permutation is None else np.asarray(action_permutation)
  if permutation.ndim!=1 or permutation.dtype.kind not in 'iu' or sorted(permutation.tolist())!=list(range(config.actions)):raise ValueError('action_permutation must be an integer bijection')
  self._goals=g.astype(np.int64);self._permutation=permutation.astype(np.int64);self._episode=-1;self._active=False;self._step=0;self._queue=[]
  self._flips=np.random.default_rng(seed).random((len(g),config.max_steps))<config.reward_flip_probability
  self.noise_schedule_sha256=hashlib.sha256(self._flips.tobytes()).hexdigest()
 @property
 def episode_count(self):return len(self._goals)
 def map_action(self,intended_action):
  self._check_action(intended_action);return int(self._permutation[intended_action])
 def reset(self):
  if self._active:raise RuntimeError('cannot reset an unfinished episode')
  self._episode+=1
  if self._episode>=len(self._goals):raise StopIteration('no episodes remain')
  self._active=True;self._step=0;self._queue=[]
  return self._observation(None,None,None,False)
 def step(self,executed_action):
  if not self._active:raise RuntimeError('reset required before step')
  self._check_action(executed_action);executed_action=int(executed_action);actual_success=executed_action==int(self._goals[self._episode]);flip=bool(self._flips[self._episode,self._step]);observed_success=actual_success!=flip;feedback=(executed_action,(1.0 if observed_success else 0.0)-self.config.action_cost)
  self._queue.append(feedback);self._step+=1;done=actual_success or self._step>=self.config.max_steps
  emitted=self._queue.pop(0) if len(self._queue)>self.config.delay else None
  if done:self._active=False
  return self._observation(executed_action,None if emitted is None else emitted[0],None if emitted is None else emitted[1],done)
 def execute(self,intended_action):return self.step(self.map_action(intended_action))
 def _check_action(self,action):
  if not isinstance(action,(int,np.integer)) or isinstance(action,(bool,np.bool_)) or not 0<=int(action)<self.config.actions:raise ValueError('action must be an in-range integer')
 def _observation(self,executed,feedback_action,reward,done):
  # Keep this allowlist narrow: hidden goal, success state, noise bit, and permutation never appear.
  return {'executed_action':executed,'feedback_action':feedback_action,'reward':reward,'done':bool(done),'step':self._step}

def _history_item(item,action_cost):
 if not isinstance(item,dict) or set(item)-{'executed_action','feedback_action','reward','done','step'}:raise ValueError('history items must be environment observations')
 action=item.get('feedback_action');reward=item.get('reward')
 if action is None and reward is None:return None
 if not isinstance(action,(int,np.integer)) or isinstance(action,(bool,np.bool_)) or not 0<=int(action)<ACTIONS or isinstance(reward,(bool,np.bool_)) or not isinstance(reward,(int,float,np.integer,np.floating)) or not np.isfinite(reward):raise ValueError('invalid feedback action/reward')
 positive=np.isclose(float(reward),1.0-action_cost,rtol=0,atol=1e-12);negative=np.isclose(float(reward),-action_cost,rtol=0,atol=1e-12)
 if not positive and not negative:raise ValueError('reward does not match configured binary feedback and action cost')
 return int(action),bool(positive)

def posterior(prior,history,reliability=.9,action_cost=.05):
 """Host Bayes update using only emitted action-linked feedback."""
 p=np.asarray(prior,dtype=np.float64)
 if p.shape!=(ACTIONS,) or not np.isfinite(p).all() or np.any(p<0) or p.sum()<=0:raise ValueError('prior must be four finite nonnegative masses')
 if not np.isfinite(reliability) or not 0<=reliability<=1 or not np.isfinite(action_cost) or action_cost<0:raise ValueError('invalid reliability or action cost')
 p=p/p.sum()
 for raw in history:
  item=_history_item(raw,action_cost)
  if item is None:continue
  action,positive=item;same=np.arange(ACTIONS)==action;likelihood=np.where(same,reliability,1-reliability) if positive else np.where(same,1-reliability,reliability);p=p*likelihood;total=p.sum()
  if total==0:raise ValueError('feedback has zero probability under prior/model')
  p=p/total
 return p

def bayes_action(prior,history,reliability=.9,action_cost=.05):return int(np.argmax(posterior(prior,history,reliability,action_cost)))

def feedback_events(observations,action_cost=.05):
 """Yield only emitted (credited action, reward, delivery timestep) triples."""
 for item in observations:
  parsed=_history_item(item,action_cost)
  if parsed is not None:yield (int(item['feedback_action']),float(item['reward']),int(item['step']))
