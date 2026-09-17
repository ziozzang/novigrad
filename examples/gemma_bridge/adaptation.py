"""Follow-up after reversal failure: exploration and entropy regularization.

Original experiment remains intact. This is an exploratory mechanism test,
not an untouched confirmatory evaluation on the reused holdout.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from safetensors.numpy import load_file
from bridge import SemanticPorts,NoviPolicy,LinearPolicy
from experiment import score,ROOT


def learn(policy,x,y,seed,steps=512,beta=.1):
    rng=np.random.default_rng(seed); start=time.perf_counter()
    for _ in range(steps):
        ix=rng.integers(len(x),size=8); batch=x[ix]; p=policy.predict(batch)
        p=p/p.sum(1,keepdims=True); q=.8*p+.2/4
        actions=np.array([rng.choice(4,p=row) for row in q]); reward=np.where(actions==y[ix],1.,-1.)
        # Importance correction keeps the reward-gradient objective on p.
        r=reward*p[np.arange(8),actions]/q[np.arange(8),actions]
        # Exact -beta*sum(p log p) gradient via the existing score-function API.
        # Five rows per observation; compensate the engine's mean update by 5.
        xx=np.concatenate([batch,np.repeat(batch,4,axis=0)])
        aa=np.concatenate([actions,np.tile(np.arange(4),8)])
        rr=np.concatenate([5*r,(-5*beta*p*(np.log(np.maximum(p,1e-30))+1)).ravel()])
        policy.learn(xx,aa,rr)
    return time.perf_counter()-start


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--beta',type=float,default=.1); parser.add_argument('--gain',type=float,default=6.); parser.add_argument('--steps',type=int,default=512); parser.add_argument('--novi-lr',type=float); parser.add_argument('--filename',default='adaptation.json'); args=parser.parse_args()
    out=ROOT/'results/gemma-bridge'; rows=json.loads((out/'dataset.json').read_text())
    e=load_file(str(out/'embeddings.safetensors'))['embeddings']; rates=SemanticPorts().encode(e)
    masks={s:np.array([r['split']==s for r in rows]) for s in ['train','test','korean']}
    labels=np.array([r['label'] for r in rows]); x={s:rates[m] for s,m in masks.items()}; y={s:labels[m] for s,m in masks.items()}
    base=json.loads((out/'experiment.json').read_text()); report={'exploratory_reused_holdout':True,'epsilon':.2,'entropy_beta':args.beta,'steps':args.steps,'logit_gain':args.gain,'runs':[]}
    for seed in [101,202,303]:
        mapping=np.random.default_rng(seed).permutation(4); reverse=(mapping+1)%4
        for kind in ['novi','linear']:
            lr=base['selection'][kind]['chosen_lr']
            if kind=='novi' and args.novi_lr is not None: lr=args.novi_lr
            policy=NoviPolicy(ROOT,lr,args.gain) if kind=='novi' else LinearPolicy(319,lr)
            seconds=learn(policy,x['train'],mapping[y['train']],seed,steps=args.steps,beta=args.beta)
            after={s:score(policy,x[s],mapping[y[s]]) for s in ['test','korean']}
            seconds+=learn(policy,x['train'],reverse[y['train']],seed+1,steps=args.steps,beta=args.beta)
            rev={s:score(policy,x[s],reverse[y[s]]) for s in ['test','korean']}
            report['runs'].append({'seed':seed,'kind':kind,'learning_rate':lr,'after':after,'reversal_after':rev,'seconds':seconds})
            (out/args.filename).write_text(json.dumps(report,indent=2)+'\n')
            print(seed,kind,'initial',after['test']['accuracy'],'reversed',rev['test']['accuracy'],flush=True)

if __name__=='__main__':main()
