"""Fixed-budget LoRA on domain calls; original BF16 checkpoint remains unchanged."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig,get_peft_model
from protocol import tools
from evaluate import BASE
from train_data import training_records,FRESH,call

p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=80);p.add_argument('--out',type=Path,default=Path('results/function-bridge/lora'));a=p.parse_args()
torch.manual_seed(731);rng=np.random.default_rng(731)
model_path='/Users/a405394/models/google_functiongemma-270m-it'
tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
base=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16)
model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','v_proj'],lora_dropout=0.,bias='none',task_type='CAUSAL_LM')).to('mps')
model.train(); records=training_records();a.out.mkdir(parents=True,exist_ok=True)
(a.out.parent/'training-cases.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
(a.out.parent/'fresh-cases.json').write_text(json.dumps(FRESH,ensure_ascii=False,indent=2)+'\n')
assert not {x[0] for x in records}&{x[0] for x in FRESH}
examples=[]
for text,name,args in records:
 prefix=tok.apply_chat_template([{'role':'developer','content':BASE},{'role':'user','content':text}],tools=tools(),add_generation_prompt=True,tokenize=False)
 prompt=tok.encode(prefix,add_special_tokens=False); response=tok.encode(call(name,args),add_special_tokens=False)
 examples.append((prompt+response,len(prompt)))
optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=2e-4)
losses=[];start=time.perf_counter()
for step in range(a.steps):
 selected=[examples[i] for i in rng.integers(len(examples),size=2)]
 length=max(len(ids) for ids,_ in selected)-1
 inputs=[];labels=[];masks=[];keep=max(len(ids)-n for ids,n in selected)
 for ids,n in selected:
  pad=length-(len(ids)-1)
  inputs.append([tok.pad_token_id]*pad+ids[:-1]);masks.append([0]*pad+[1]*(len(ids)-1))
  lab=[-100]*pad+[-100]*(n-1)+ids[n:];assert len(lab)==length;labels.append(lab[-keep:])
 optimizer.zero_grad(set_to_none=True)
 result=model(input_ids=torch.tensor(inputs,device='mps'),attention_mask=torch.tensor(masks,device='mps'),use_cache=False,logits_to_keep=keep)
 loss=torch.nn.functional.cross_entropy(result.logits.float().reshape(-1,result.logits.shape[-1]),torch.tensor(labels,device='mps').reshape(-1),ignore_index=-100)
 loss.backward();torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.);optimizer.step()
 losses.append(float(loss.detach().cpu()))
 if step%10==0: print(step,losses[-1],time.perf_counter()-start,flush=True)
model.save_pretrained(a.out,safe_serialization=True)
report={'seed':731,'steps':a.steps,'batch_size':2,'learning_rate':2e-4,'rank':8,'alpha':16,'modules':['q_proj','v_proj'],'train_examples':len(examples),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'base_dtype':'bfloat16','base_weights_modified':False,'losses':losses,'seconds':time.perf_counter()-start}
(a.out.parent/'finetune.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
