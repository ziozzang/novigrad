"""Original BF16 FunctionGemma; deterministic unconstrained generation audit."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from protocol import tools,parse
from cases import DEVELOPMENT,HOLDOUT,ROBUSTNESS

BASE='You are a model that can do function calling with the following functions'
FEWSHOT=[
 ('Set the goal to getting a drink.','<start_function_call>call:set_goal{goal:<escape>water<escape>}<end_function_call>'),
 ('Record a heading of 30 degrees.','<start_function_call>call:observe_heading{degrees:30}<end_function_call>'),
 ('Choose my next action.','<start_function_call>call:choose_action{}<end_function_call>'),
 ('Show current state.','<start_function_call>call:get_status{}<end_function_call>'),
]

def main():
 p=argparse.ArgumentParser();p.add_argument('--adapter',type=Path);p.add_argument('--model',default='/Users/a405394/models/google_functiongemma-270m-it');p.add_argument('--variant',choices=['base','descriptive','fewshot'],default='base');p.add_argument('--split',choices=['development','holdout','robustness','fresh'],default='development');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16).to('mps').eval()
 if a.adapter:
  from peft import PeftModel
  model=PeftModel.from_pretrained(model,str(a.adapter)).eval()
 schema=tools()
 if a.variant!='base':
  descriptions={'set_goal':'Set a new goal or need, including thirst/hydration=water, hunger/meal=food, cold/shivering=warmth, tired/sleep=rest. 목표 설정: 물 water, 음식 food, 보온 warmth, 휴식 rest.', 'observe_heading':'Record the measured compass bearing/heading, in degrees. 방향 각도를 기록합니다. Do not invent an angle.', 'choose_action':'Select the next action with the novi neural policy. 다음 행동을 선택합니다. No parameters.', 'get_status':'Read stored goal, heading and action state. 현재 상태 조회. No parameters.'}
  for s in schema:s['function']['description']=descriptions[s['function']['name']]
 from train_data import FRESH
 cases={'fresh':FRESH,'development':DEVELOPMENT,'holdout':HOLDOUT,'robustness':ROBUSTNESS}[a.split]
 report={'variant':a.variant,'adapter':str(a.adapter) if a.adapter else None,'split':a.split,'dtype':'bfloat16','device':'mps','do_sample':False,'max_new_tokens':80,'cases':[]}
 for case in cases:
  messages=[{'role':'developer','content':BASE}]
  if a.variant=='fewshot':
   for question,answer in FEWSHOT:messages.extend([{'role':'user','content':question},{'role':'assistant','content':answer}])
  messages.append({'role':'user','content':case[0]})
  inputs=tok.apply_chat_template(messages,tools=schema,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('mps')
  start=time.perf_counter()
  with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=80,do_sample=False,pad_token_id=tok.pad_token_id)
  torch.mps.synchronize();elapsed=time.perf_counter()-start
  ids=generated[0,inputs['input_ids'].shape[1]:];raw=tok.decode(ids,skip_special_tokens=False)
  result=None;error=None
  try:result=parse(raw)
  except ValueError as e:error=str(e)
  expected=None if a.split=='robustness' else (case[1],case[2])
  correct=(result is None) if expected is None else result==expected
  report['cases'].append({'prompt':case[0],'expected':expected,'raw':raw,'parsed':result,'error':error,'correct':correct,'seconds':elapsed,'generated_tokens':len(ids)})
  print(a.variant,a.split,correct,raw,flush=True)
 report['accuracy']=sum(x['correct'] for x in report['cases'])/len(cases)
 report['model_sha256']=hashlib.file_digest(open(Path(a.model)/'model.safetensors','rb'),'sha256').hexdigest()
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(report['accuracy'])
if __name__=='__main__':main()
