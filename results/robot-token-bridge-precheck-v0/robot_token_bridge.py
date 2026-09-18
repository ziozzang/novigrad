#!/usr/bin/env python3
"""Typed host-token/robot loop demonstration; neither an LM nor a fly reconstruction."""
from __future__ import annotations
import argparse,hashlib,json,math
from dataclasses import asdict,dataclass
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/robot-token-bridge'
DT=.01;STEPS=300;SWITCH_TICK=150;SEEDS=tuple(range(2001,2009));INITIALS=16
SOURCES=('true_feedback','action_copy_feedback');FRAMES=('body','world')

def finite(x):return isinstance(x,(int,float,np.integer,np.floating)) and not isinstance(x,(bool,np.bool_)) and math.isfinite(float(x))
def wrap_angle(x):return (float(x)+math.pi)%(2*math.pi)-math.pi

@dataclass(frozen=True)
class ObservationPacket:
 heading_sin:float;heading_cos:float;goal_sin:float;goal_cos:float
 actual_body_turn:float;action_copy:float;timestamp_s:float;source:str;units:str='radian';valid:bool=True
 def __post_init__(self):
  values=(self.heading_sin,self.heading_cos,self.goal_sin,self.goal_cos,self.actual_body_turn,self.action_copy,self.timestamp_s)
  if not all(finite(x) for x in values):raise ValueError('observation fields must be finite')
  if self.timestamp_s<0:raise ValueError('timestamp must be nonnegative')
  if self.source not in SOURCES:raise ValueError('unknown observation source')
  if self.units!='radian':raise ValueError('observation units must be radian')
  if type(self.valid) is not bool:raise ValueError('valid must be boolean')
  for a,b,name in ((self.heading_sin,self.heading_cos,'heading'),(self.goal_sin,self.goal_cos,'goal')):
   if not math.isclose(a*a+b*b,1.,rel_tol=0,abs_tol=1e-5):raise ValueError(name+' ring must have unit norm')
 def angles(self):
  if not self.valid:raise ValueError('invalid observation')
  return math.atan2(self.heading_sin,self.heading_cos),math.atan2(self.goal_sin,self.goal_cos)

@dataclass(frozen=True)
class MotorCommand:
 frame:str;turn_rate_rad_s:float;duration_s:float;issued_s:float;expires_s:float;sequence_id:int
 def __post_init__(self):
  if self.frame not in FRAMES:raise ValueError('unknown command frame')
  if not all(finite(x) for x in (self.turn_rate_rad_s,self.duration_s,self.issued_s,self.expires_s)):raise ValueError('command fields must be finite')
  if self.duration_s<=0 or self.issued_s<0 or self.expires_s<self.issued_s or self.expires_s>self.issued_s+self.duration_s+1e-12:raise ValueError('invalid command timing')
  if type(self.sequence_id) is not int or self.sequence_id<0:raise ValueError('sequence_id must be a nonnegative integer')

class CommandReceiver:
 def __init__(self):self.last_sequence=-1;self.command=None
 def accept(self,command,now):
  if command.frame!='body':raise ValueError('command frame must be body')
  if not finite(now) or now<0:raise ValueError('invalid receiver time')
  if command.sequence_id<=self.last_sequence:raise ValueError('duplicate or out-of-order sequence')
  if command.issued_s>now+1e-12 or command.expires_s<now-1e-12:raise ValueError('stale or future command')
  self.last_sequence=command.sequence_id;self.command=command
 def rate(self,now):
  if not finite(now):raise ValueError('invalid receiver time')
  return 0. if self.command is None or now>self.command.expires_s+1e-12 else self.command.turn_rate_rad_s

class HeadingTokenCodec:
 """Host-owned discrete IDs, explicitly unrelated to any LM vocabulary."""
 def __init__(self,bins):
  if type(bins) is not int or bins<4:raise ValueError('bins must be integer >=4')
  self.bins=bins
 def encode_angle(self,angle,kind):
  if kind not in ('heading','goal') or not finite(angle):raise ValueError('invalid angle token input')
  index=int(math.floor(((wrap_angle(angle)+math.pi)/(2*math.pi))*self.bins))%self.bins
  return (1000 if kind=='heading' else 2000)+index
 def decode_angle(self,token,kind):
  base=1000 if kind=='heading' else 2000
  if kind not in ('heading','goal') or not isinstance(token,(int,np.integer)) or isinstance(token,(bool,np.bool_)) or not base<=int(token)<base+self.bins:raise ValueError('token is outside host codec inventory')
  return wrap_angle(-math.pi+(int(token)-base+.5)*2*math.pi/self.bins)
 def roundtrip(self,heading,goal):return (self.decode_angle(self.encode_angle(heading,'heading'),'heading'),self.decode_angle(self.encode_angle(goal,'goal'),'goal'))

def pfl3_inspired_rate(heading,goal):
 """A clipped sine steering rule inspired by, but not implementing, PFL3."""
 return float(np.clip(2.*math.sin(wrap_angle(goal-heading)),-2.5,2.5))

CONDITIONS=(
 {'name':'continuous_true_disturbance','bins':None,'feedback':'true_feedback','schedule':'100hz','disturbance':'applied'},
 *({'name':f'bins{b}_true_disturbance','bins':b,'feedback':'true_feedback','schedule':'100hz','disturbance':'applied'} for b in (16,64,256)),
 {'name':'bins64_actioncopy_disturbance','bins':64,'feedback':'action_copy_feedback','schedule':'100hz','disturbance':'applied'},
 {'name':'bins64_wrongframe_disturbance','bins':64,'feedback':'true_feedback','schedule':'wrong_frame','disturbance':'applied'},
 {'name':'bins64_oneshot_disturbance','bins':64,'feedback':'true_feedback','schedule':'one_shot','disturbance':'applied'},
 {'name':'bins64_slow2hz_disturbance','bins':64,'feedback':'true_feedback','schedule':'2hz_hold','disturbance':'applied'},
 {'name':'bins64_true_disturbance_noeffect','bins':64,'feedback':'true_feedback','schedule':'100hz','disturbance':'logged_not_applied'})
CONFIG={'dt_s':DT,'rate_hz':100,'steps':STEPS,'duration_s':3.,'goal_switch_s':1.5,'seeds':list(SEEDS),'initial_conditions_per_seed':INITIALS,'controller_gain':2.,'max_turn_rate_rad_s':2.5,'success_error_rad':.15,'conditions':list(CONDITIONS),'anti_goal_boundary':'Initial and switched goals within .02 rad of exact pi error are deterministically shifted by .03 rad; exact anti-goal is an unstable zero of the sine controller.'}

def packet(heading,goal,actual_turn,action_copy,timestamp,source):return ObservationPacket(math.sin(heading),math.cos(heading),math.sin(goal),math.cos(goal),actual_turn,action_copy,timestamp,source)

def _case_parameters(seed,index):
 rng=np.random.default_rng(seed*1000+index);heading=float(rng.uniform(-math.pi,math.pi));goals=[float(rng.uniform(-math.pi,math.pi)) for _ in range(2)]
 for i in range(2):
  if abs(abs(wrap_angle(goals[i]-heading))-math.pi)<.02:goals[i]=wrap_angle(goals[i]+.03)
 disturbance=float(rng.uniform(-.6,.6));return heading,goals,disturbance

def simulate_case(condition,seed,index):
 heading,goals,disturbance=_case_parameters(seed,index);copy_heading=heading;receiver=CommandReceiver();codec=None if condition['bins'] is None else HeadingTokenCodec(condition['bins']);sequence=0;actual_turn=action_copy=0.;trace=[];errors=[]
 for tick in range(STEPS+1):
  now=tick*DT;goal=goals[0 if tick<SWITCH_TICK else 1];observed=heading if condition['feedback']=='true_feedback' else copy_heading
  obs=packet(observed,goal,actual_turn,action_copy,now,condition['feedback']);oh,og=obs.angles()
  tokens=None
  if codec is not None:
   tokens=[codec.encode_angle(oh,'heading'),codec.encode_angle(og,'goal')];oh,og=codec.roundtrip(oh,og)
  schedule=condition['schedule'];update=(schedule in ('100hz','wrong_frame')) or (schedule=='one_shot' and tick==0) or (schedule=='2hz_hold' and tick%50==0)
  command_error=None
  if update:
   duration={'100hz':.02,'wrong_frame':.02,'one_shot':.25,'2hz_hold':.5}[schedule];cmd=MotorCommand('world' if schedule=='wrong_frame' else 'body',pfl3_inspired_rate(oh,og),duration,now,now+duration,sequence);sequence+=1
   try:receiver.accept(cmd,now)
   except ValueError as exc:command_error=str(exc)
  rate=receiver.rate(now);error=abs(wrap_angle(goal-heading));errors.append(error)
  trace.append([tick,heading,goal,observed,rate,disturbance,actual_turn,action_copy,error,tokens,command_error])
  if tick==STEPS:break
  applied_disturbance=disturbance if condition['disturbance']=='applied' else 0.;delta=(rate+applied_disturbance)*DT;heading=wrap_angle(heading+delta);action_copy=rate*DT;actual_turn=delta;copy_heading=wrap_angle(copy_heading+action_copy)
 final_error=errors[-1];return {'seed':seed,'initial_index':index,'initial_heading':trace[0][1],'goals':goals,'disturbance_rad_s':disturbance,'success':final_error<CONFIG['success_error_rad'],'final_error_rad':final_error,'mean_abs_error_last_half_second_rad':float(np.mean(errors[-50:])),'command_errors':sum(row[-1] is not None for row in trace),'trace_columns':['tick','true_heading','goal','observed_heading','commanded_turn_rate','external_disturbance','actual_body_turn','action_copy','absolute_error','host_tokens','command_error'],'trace':trace}

def sha(path):
 h=hashlib.sha256();h.update(Path(path).read_bytes());return h.hexdigest()
def build_report():
 results={}
 for condition in CONDITIONS:
  rows=[simulate_case(condition,seed,index) for seed in SEEDS for index in range(INITIALS)]
  results[condition['name']]={'metrics':{'cases':len(rows),'successes':sum(x['success'] for x in rows),'success_rate':float(np.mean([x['success'] for x in rows])),'mean_final_error_rad':float(np.mean([x['final_error_rad'] for x in rows])),'mean_last_half_second_error_rad':float(np.mean([x['mean_abs_error_last_half_second_rad'] for x in rows])),'command_errors':sum(x['command_errors'] for x in rows)},'cases':rows}
 return {'design':CONFIG,'claims_boundary':['Host token IDs are not tokens from an embedding model or general LM vocabulary','The sine steering rule is only PFL3-inspired and is not an anatomical or learned fly-brain implementation','The explicit goal is a supervised requested heading, not a hidden label or inferred intention','Action-copy feedback omits external plant rotation by construction'],'results':results}
def run():
 if OUT.exists():raise FileExistsError('refusing to overwrite robot-token result directory')
 OUT.mkdir(parents=True);report=build_report();source=Path(__file__);test=source.with_name('test_robot_token_bridge.py');report['provenance']={'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (source,test)},'config_sha256':hashlib.sha256(json.dumps(CONFIG,sort_keys=True,separators=(',',':')).encode()).hexdigest()};(OUT/'results.json').write_text(json.dumps(report,separators=(',',':'),allow_nan=False)+'\n');(OUT/'protocol-lock.json').write_text(json.dumps({'config':CONFIG,'source_sha256':report['provenance']['source_sha256'],'results_sha256':sha(OUT/'results.json')},indent=2)+'\n')
def verify():
 lock=json.loads((OUT/'protocol-lock.json').read_text());
 for name,digest in lock['source_sha256'].items():
  if sha(ROOT/name)!=digest:raise ValueError('source changed: '+name)
 if sha(OUT/'results.json')!=lock['results_sha256']:raise ValueError('result hash changed')
 saved=json.loads((OUT/'results.json').read_text());replayed=build_report();replayed['provenance']=saved['provenance']
 if replayed!=saved:raise AssertionError('exact deterministic replay differs')
 (OUT/'verification.json').write_text(json.dumps({'exact_full_trace_replay':True,'cases':len(CONDITIONS)*len(SEEDS)*INITIALS,'results_sha256':lock['results_sha256']},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('run','verify'));globals()[p.parse_args().stage]()
