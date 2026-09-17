"""Capacity control on a direct embedding adapter, outside the connectome path.

This experiment does not fine-tune EmbeddingGemma. It fits a train-only PCA bridge,
then compares rank-constrained adapter deltas with an unconstrained matrix delta.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from precise_bridge import dataset

INPUT_DIM=768
OUTPUT_DIM=128
CLASSES=4
RANKS=(4,16,64)
SEEDS=(11,23,47)
EPOCHS=300
LEARNING_RATE=0.03


def _pca(x):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=INPUT_DIM or len(x)<2 or not np.isfinite(x).all():
        raise ValueError(f'finite [N,{INPUT_DIM}] training embeddings required')
    mean=x.mean(0,dtype=np.float64).astype(np.float32)
    _,singular,v=np.linalg.svd(x-mean,full_matrices=False)
    w=np.zeros((INPUT_DIM,OUTPUT_DIM),np.float32)
    numerical=int(np.sum(singular>singular[0]*1e-6)) if singular[0]>0 else 0
    n=min(OUTPUT_DIM,len(x)-1,numerical);w[:,:n]=v[:n].T
    for j in range(n):
        i=np.argmax(np.abs(w[:,j]))
        if w[i,j]<0:w[:,j]*=-1
    return mean,w


def _device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def load_state(path,kind,seed):
    """Load a checkpoint only after its metadata and complete tensor schema agree."""
    path=Path(path)
    with safe_open(str(path),framework='pt',device='cpu') as handle:
        metadata=handle.metadata() or {};names=set(handle.keys())
    expected_metadata={'format':'novigrad.architecture_adapter','version':'1','kind':kind,'seed':str(seed),'scope':'direct_embedding_adapter_not_connectome'}
    if metadata!=expected_metadata:raise ValueError(f'invalid checkpoint metadata: {path}')
    base={'mean':(INPUT_DIM,),'w0':(INPUT_DIM,OUTPUT_DIM),'head_weight':(CLASSES,OUTPUT_DIM),'head_bias':(CLASSES,)}
    if kind=='frozen':schema=base
    elif kind=='full':schema={**base,'delta':(INPUT_DIM,OUTPUT_DIM)}
    elif kind.startswith('rank') and kind[4:].isdigit():
        rank=int(kind[4:]);schema={**base,'adapter_a':(INPUT_DIM,rank),'adapter_b':(rank,OUTPUT_DIM)}
    else:raise ValueError(f'unknown checkpoint kind: {kind}')
    if names!=set(schema):raise ValueError(f'invalid checkpoint tensor names: {path}')
    state=load_file(str(path),device='cpu')
    for name,shape in schema.items():
        tensor=state[name]
        if tensor.dtype!=torch.float32 or tuple(tensor.shape)!=shape or not torch.isfinite(tensor).all():
            raise ValueError(f'invalid checkpoint tensor {name}: {path}')
    return state

def _logits(x,state):
    delta=state['delta'] if 'delta' in state else (state['adapter_a']@state['adapter_b'] if 'adapter_a' in state else torch.zeros_like(state['w0']))
    return ((x-state['mean'])@(state['w0']+delta))@state['head_weight'].T+state['head_bias']


def _train_one(train_x,train_y,val_x,val_y,mean,w0,kind,seed):
    device=_device();torch.manual_seed(seed)
    x=torch.tensor(train_x,dtype=torch.float32,device=device);y=torch.tensor(train_y,dtype=torch.long,device=device)
    vx=torch.tensor(val_x,dtype=torch.float32,device=device)
    tm=torch.tensor(mean,device=device);tw=torch.tensor(w0,device=device)
    # Identical head initialization for every capacity at a given seed.
    g=torch.Generator(device='cpu').manual_seed(seed^0x5A17)
    hw=(torch.randn(CLASSES,OUTPUT_DIM,generator=g)*0.02).to(device).requires_grad_()
    hb=torch.zeros(CLASSES,device=device,requires_grad=True)
    if kind=='frozen':
        params=[hw,hb]
    elif kind=='full':
        delta=torch.zeros(INPUT_DIM,OUTPUT_DIM,device=device,requires_grad=True);params=[delta,hw,hb]
    else:
        rank=int(kind[4:]);ga=torch.Generator(device='cpu').manual_seed(seed^0xA341)
        aa=(torch.randn(INPUT_DIM,rank,generator=ga)/np.sqrt(INPUT_DIM)).to(device).requires_grad_()
        ab=torch.zeros(rank,OUTPUT_DIM,device=device,requires_grad=True);params=[aa,ab,hw,hb]
    opt=torch.optim.Adam(params,lr=LEARNING_RATE)
    if device.type=='mps':torch.mps.synchronize()
    start=time.perf_counter()
    for _ in range(EPOCHS):
        opt.zero_grad(set_to_none=True)
        d=torch.zeros_like(tw) if kind=='frozen' else (delta if kind=='full' else aa@ab)
        loss=torch.nn.functional.cross_entropy(((x-tm)@(tw+d))@hw.T+hb,y)
        loss.backward();opt.step()
    if device.type=='mps':torch.mps.synchronize()
    elapsed=time.perf_counter()-start
    d=torch.zeros_like(tw) if kind=='frozen' else (delta if kind=='full' else aa@ab)
    with torch.no_grad():
        train_loss=float(torch.nn.functional.cross_entropy(((x-tm)@(tw+d))@hw.T+hb,y).cpu())
        pred=(((vx-tm)@(tw+d))@hw.T+hb).argmax(1).cpu().numpy()
    state={'mean':tm.detach().cpu(),'w0':tw.detach().cpu(),'head_weight':hw.detach().cpu(),'head_bias':hb.detach().cpu()}
    if kind=='full':state['delta']=delta.detach().cpu()
    elif kind!='frozen':state.update(adapter_a=aa.detach().cpu(),adapter_b=ab.detach().cpu())
    return state,train_loss,float(np.mean(pred==val_y)),elapsed


def _validate_data(x,y,name):
    raw_y=np.asarray(y)
    if raw_y.ndim!=1 or not np.issubdtype(raw_y.dtype,np.integer):raise ValueError(f'{name} labels must be integers')
    x=np.asarray(x,dtype=np.float32);y=raw_y.astype(np.int64,copy=False)
    if x.ndim!=2 or x.shape[1]!=INPUT_DIM or len(x)==0 or y.shape!=(len(x),) or not np.isfinite(x).all():raise ValueError(f'invalid {name} data')
    if np.any((y<0)|(y>=CLASSES)):raise ValueError(f'invalid {name} labels')
    return x,y


def train_all(out):
    """Train all fixed capacities and seeds, save weights, and return the report."""
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'refusing to overwrite nonempty output: {out}')
    out.mkdir(parents=True,exist_ok=True)
    data=dataset();train_x,train_y,_=data['train'];val_x,val_y,_=data['validation']
    train_x,train_y=_validate_data(train_x,train_y,'train');val_x,val_y=_validate_data(val_x,val_y,'validation')
    original=train_x.copy();mean,w0=_pca(train_x);results={}
    for kind in ('frozen',)+tuple(f'rank{r}' for r in RANKS)+('full',):
        rows=[]
        for seed in SEEDS:
            state,_,_,elapsed=_train_one(train_x,train_y,val_x,val_y,mean,w0,kind,seed)
            path=out/f'{kind}-seed{seed}.safetensors'
            metadata={'format':'novigrad.architecture_adapter','version':'1','kind':kind,'seed':str(seed),'scope':'direct_embedding_adapter_not_connectome'}
            save_file(state,str(path),metadata=metadata)
            restored=load_state(path,kind,seed);cpu_x=torch.from_numpy(train_x);cpu_vx=torch.from_numpy(val_x)
            if any(not torch.isfinite(t).all() for t in restored.values()):raise FloatingPointError('non-finite saved weight')
            before=_logits(cpu_x,state);after=_logits(cpu_x,restored)
            if not torch.equal(before,after):raise AssertionError('CPU safetensors inference roundtrip changed logits')
            validation_logits=_logits(cpu_vx,restored)
            if not torch.isfinite(after).all() or not torch.isfinite(validation_logits).all():raise FloatingPointError('non-finite logits')
            train_probability=torch.softmax(after,1);validation_probability=torch.softmax(validation_logits,1)
            train_prediction=after.argmax(1).numpy();validation_prediction=validation_logits.argmax(1).numpy()
            cpu_train_loss=float(torch.nn.functional.cross_entropy(after,torch.from_numpy(train_y)))
            cpu_validation_loss=float(torch.nn.functional.cross_entropy(validation_logits,torch.from_numpy(val_y)))
            params=sum(t.numel() for n,t in state.items() if n not in ('mean','w0'))
            combined=restored['w0']+(restored['delta'] if 'delta' in restored else (restored['adapter_a']@restored['adapter_b'] if 'adapter_a' in restored else 0))
            effective=combined@restored['head_weight'].T
            rows.append({'seed':seed,'parameters':params,'train_loss':cpu_train_loss,'train_accuracy':float(np.mean(train_prediction==train_y)),'train_correct_probability':float(train_probability[range(len(train_y)),train_y].mean()),'validation_loss':cpu_validation_loss,'validation_accuracy':float(np.mean(validation_prediction==val_y)),'validation_correct_probability':float(validation_probability[range(len(val_y)),val_y].mean()),'effective_linear_rank':int(torch.linalg.matrix_rank(effective).item()),'seconds':elapsed,'weights':path.name,'cpu_roundtrip_exact':True})
        results[kind]=rows
    if not np.array_equal(train_x,original):raise AssertionError('source embeddings mutated')
    report={'scope':'Direct 768-to-128 embedding adapter capacity control; no connectome path and no EmbeddingGemma fine-tuning. PCA is fit on train only. The complete model is linear with four outputs, so its effective input-to-logit rank is at most 4; rank-64 adds parameter capacity but cannot add output rank.', 'fixed_hyperparameters':{'epochs':EPOCHS,'learning_rate':LEARNING_RATE,'optimizer':'Adam full-batch','seeds':list(SEEDS)},'models':results}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def evaluate_all(out,embeddings,labels):
    """Load every saved model without retraining and evaluate supplied labeled rows."""
    out=Path(out);x,y=_validate_data(embeddings,labels,'evaluation');tx=torch.from_numpy(x);result={}
    for kind in ('frozen',)+tuple(f'rank{r}' for r in RANKS)+('full',):
        rows=[]
        for seed in SEEDS:
            state=load_state(out/f'{kind}-seed{seed}.safetensors',kind,seed)
            if any(not torch.isfinite(t).all() for t in state.values()):raise FloatingPointError('non-finite saved weight')
            logits=_logits(tx,state)
            if not torch.isfinite(logits).all():raise FloatingPointError('non-finite logits')
            probability=torch.softmax(logits,1);prediction=logits.argmax(1).numpy()
            rows.append({'seed':seed,'loss':float(torch.nn.functional.cross_entropy(logits,torch.from_numpy(y))),'accuracy':float((prediction==y).mean()),'correct_probability':float(probability[range(len(y)),y].mean()),'predictions':prediction.tolist()})
        result[kind]=rows
    return result

if __name__=='__main__':
    train_all(Path(__file__).resolve().parents[2]/'results'/'architecture-adapter')
