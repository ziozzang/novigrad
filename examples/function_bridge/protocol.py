"""A small strict FunctionGemma call grammar and allowlisted novi tools."""
import math
import re

PROPERTIES = {
 'set_goal': {'goal': {'type':'string','enum':['water','food','warmth','rest']}},
 'observe_heading': {'degrees': {'type':'number','minimum':-180,'maximum':180}},
 'choose_action': {}, 'get_status': {},
}
DESCRIPTIONS = {
 'set_goal':'Set the current need: water for thirst, food for hunger, warmth for cold, rest for tiredness.',
 'observe_heading':'Record an observed heading angle in degrees from -180 to 180.',
 'choose_action':'Ask the novi neural policy to choose an action for the current goal.',
 'get_status':'Read the current goal, heading, and pending action without changing state.',
}

def tools():
 return [{'type':'function','function':{'name':name,'description':DESCRIPTIONS[name],
          'parameters':{'type':'object','properties':props,'required':list(props),'additionalProperties':False}}}
         for name,props in PROPERTIES.items()]


def validate(name, args):
 if name not in PROPERTIES or not isinstance(args,dict): raise ValueError('unknown function or non-object arguments')
 if set(args)!=set(PROPERTIES[name]): raise ValueError('wrong argument names')
 if name=='set_goal' and args['goal'] not in ('water','food','warmth','rest'): raise ValueError('unknown goal')
 if name=='observe_heading':
  value=args['degrees']
  if type(value) not in (int,float) or not math.isfinite(value) or not -180<=value<=180: raise ValueError('invalid heading')
 return name,args


def parse(raw):
 """Accept one fully delimited scalar-only call. No eval, repair, or partial execution."""
 match=re.fullmatch(r'\s*<start_function_call>call:([a-z_]+)\{(.*?)\}<end_function_call>\s*(?:<start_function_response>|<end_of_turn>|<eos>)?\s*',raw,re.S)
 if not match: raise ValueError('expected exactly one complete function call')
 name,body=match.groups(); args={}
 # Current schemas have at most one scalar; reject duplicate keys and trailing content.
 if body:
  item=re.fullmatch(r'([a-z_]+):(?:<escape>([^<>]*)<escape>|(-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?))',body,re.S)
  if not item: raise ValueError('malformed arguments')
  key,string,number=item.groups();args[key]=string if string is not None else float(number)
 return validate(name,args)


class Session:
 def __init__(self, decide):
  self.decide=decide;self.goal=None;self.heading=None;self.pending=None;self.sequence=0
 def call(self,name,args):
  validate(name,args)
  if name=='set_goal':
   if self.pending is not None: raise ValueError('resolve pending action before changing goal')
   self.goal=args['goal']
  elif name=='observe_heading':self.heading=args['degrees']
  elif name=='choose_action':
   if self.goal is None: raise ValueError('set a goal first')
   if self.pending is not None:raise ValueError('pending action already exists')
   result=self.decide(self.goal)  # No mutation if policy raises.
   self.sequence+=1; self.pending={'id':self.sequence,'goal':self.goal,**result}
   return dict(self.pending)
  return self.status()
 def status(self):return {'goal':self.goal,'heading':self.heading,'heading_source':'user_reported_unverified' if self.heading is not None else None,'pending':None if self.pending is None else dict(self.pending)}
 def environment_feedback(self, action_id, reward, learn):
  """Trusted environment hook; never declared as a model-callable function."""
  if self.pending is None or type(action_id) is not int or action_id!=self.pending['id']:raise ValueError('invalid or stale action id')
  if type(reward) not in (int,float) or not math.isfinite(reward) or not -1<=reward<=1:raise ValueError('invalid reward')
  learn(dict(self.pending),reward)
  self.pending=None
