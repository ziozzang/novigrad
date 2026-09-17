"""Frozen, independently authored cases; no prompt adaptation or output repair."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from protocol import tools, parse
from evaluate import BASE


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=Path('/Users/a405394/models/google_functiongemma-270m-it'))
    p.add_argument('--cases', type=Path, default=Path(__file__).with_name('deep_holdout.json'))
    p.add_argument('--adapter', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    cases = json.loads(a.cases.read_text())
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True, dtype=torch.bfloat16).to('mps').eval()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(a.adapter)).eval()
    rows = []
    for case in cases:
        inputs = tok.apply_chat_template([{'role':'developer','content':BASE}, {'role':'user','content':case['prompt']}], tools=tools(), add_generation_prompt=True, return_dict=True, return_tensors='pt').to('mps')
        torch.mps.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
        torch.mps.synchronize()
        elapsed = time.perf_counter()-start
        raw = tok.decode(output[0,inputs['input_ids'].shape[1]:], skip_special_tokens=False)
        parsed, error = None, None
        try:
            parsed = list(parse(raw))
        except ValueError as e:
            error = str(e)
        rows.append({**case, 'raw':raw, 'parsed':parsed, 'error':error, 'correct':parsed == case['expected'], 'seconds':elapsed})
        print(case['id'], rows[-1]['correct'], flush=True)
    groups = {}
    for name, selected in [('valid', [r for r in rows if r['expected'] is not None]), ('unsupported', [r for r in rows if r['expected'] is None])] + [(lang,[r for r in rows if r['language']==lang and r['expected'] is not None]) for lang in sorted({r['language'] for r in rows})]:
        groups[name] = {'correct':sum(r['correct'] for r in selected), 'count':len(selected), 'accuracy':float(np.mean([r['correct'] for r in selected]))}
    files = [a.cases, Path(__file__), Path(__file__).with_name('protocol.py')] + list(a.model.glob('*.json')) + [a.model/'model.safetensors',a.model/'chat_template.jinja']
    if a.adapter:
        files += [a.adapter/'adapter_model.safetensors',a.adapter/'adapter_config.json']
    report = {'model':str(a.model),'adapter':str(a.adapter) if a.adapter else None,'sha256':{str(f):digest(f) for f in files},'groups':groups,'latency_seconds':{'p50':float(np.median([r['seconds'] for r in rows])),'p95':float(np.percentile([r['seconds'] for r in rows],95))},'cases':rows,'scope':'Single greedy generation per prompt, no retries. Unsupported rejection means parser refusal, not semantic model refusal. Independent author had schema access only; not population-representative.'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(groups),flush=True)

if __name__=='__main__':
    main()
