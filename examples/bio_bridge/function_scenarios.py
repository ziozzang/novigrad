"""Biological-state-inspired intent scenarios, with frozen FunctionGemma models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'function_bridge'))
from protocol import tools, parse
from evaluate import BASE

ROOT=Path(__file__).resolve().parents[2]
POLICY=(' Only execute the latest user request, using previous conversation only as context. '
        'A past need, a negated need, or a quoted example is not a current goal. '
        'Use the goal explicitly prioritized now. Call at most one supported function. '
        'If multiple needs have no priority, the request is unsupported, or the user says not to execute, '
        'do not call any function; ask a short clarification instead. Never invent a measurement.')


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def metrics(rows):
    valid=[r for r in rows if r['expected'] is not None]
    negative=[r for r in rows if r['expected'] is None]
    v=float(np.mean([r['correct'] for r in valid])) if valid else None
    n=float(np.mean([r['correct'] for r in negative])) if negative else None
    return {'count':len(rows),'valid_correct':sum(r['correct'] for r in valid),'valid_count':len(valid),'valid_accuracy':v,'rejected':sum(r['correct'] for r in negative),'negative_count':len(negative),'rejection_rate':n,'balanced_accuracy':(v+n)/2 if v is not None and n is not None else None,'unsupported_state_calls':sum(r['parsed'] is not None and r['parsed'][0]!='get_status' for r in negative)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,default=Path('/Users/a405394/models/google_functiongemma-270m-it'))
    p.add_argument('--adapter',type=Path)
    p.add_argument('--intent-policy',action='store_true')
    p.add_argument('--history-ablation',action='store_true',help='Evaluate only16multi-turn cases using last user message alone')
    p.add_argument('--out',type=Path,default=ROOT/'results/bio-bridge/function-base.json')
    a=p.parse_args();casespath=Path(__file__).with_name('cases.json')
    cases=json.loads(casespath.read_text())
    if a.history_ablation:cases=[c for c in cases if len(c['messages'])>1]
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16).to('mps').eval()
    if a.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,str(a.adapter)).eval()
    rows=[]
    for c in cases:
        messages=[{'role':'developer','content':BASE+(POLICY if a.intent_policy else '')}]+(c['messages'][-1:] if a.history_ablation else c['messages'])
        inp=tok.apply_chat_template(messages,tools=tools(),add_generation_prompt=True,return_dict=True,return_tensors='pt').to('mps')
        torch.mps.synchronize();start=time.perf_counter()
        with torch.inference_mode():out=model.generate(**inp,max_new_tokens=80,do_sample=False,pad_token_id=tok.pad_token_id)
        torch.mps.synchronize();elapsed=time.perf_counter()-start
        raw=tok.decode(out[0,inp['input_ids'].shape[1]:],skip_special_tokens=False)
        result,error=None,None
        try:result=list(parse(raw))
        except ValueError as e:error=str(e)
        rows.append({**c,'parsed':result,'raw':raw,'error':error,'correct':result==c['expected'],'seconds':elapsed})
        print(c['id'],rows[-1]['correct'],flush=True)
    files=[casespath,Path(__file__),a.model/'model.safetensors',a.model/'tokenizer_config.json',a.model/'chat_template.jinja']
    if a.adapter:files.extend([a.adapter/'adapter_model.safetensors',a.adapter/'adapter_config.json'])
    report={'history_ablation':a.history_ablation,'model':str(a.model),'adapter':str(a.adapter) if a.adapter else None,'intent_policy':POLICY if a.intent_policy else None,'sha256':{str(f):digest(f) for f in files},'overall':metrics(rows),'by_scenario':{s:metrics([r for r in rows if r['scenario']==s]) for s in sorted({r['scenario'] for r in rows})},'by_language':{s:metrics([r for r in rows if r['language']==s]) for s in ['en','ko']},'latency_seconds':{'p50':float(np.median([r['seconds'] for r in rows])),'p95':float(np.percentile([r['seconds'] for r in rows],95))},'cases':rows,'limits':['hand-authored protocol-specific cases, not deployment benchmark','multi-turn histories contain supplied assistant calls, not self-generated rollouts','parser nonexecution is not necessarily semantic refusal','no adapter retraining or prompt search on these cases']}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['overall']),flush=True)
if __name__=='__main__':main()
