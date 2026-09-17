"""Reward-only, train/validation/test separated semantic policy experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from safetensors.numpy import load_file
from bridge import SemanticPorts, NoviPolicy, LinearPolicy
from novigrad import Engine
ROOT = Path(__file__).resolve().parents[2]


def score(policy, x, labels):
    p=policy.predict(x); pred=p.argmax(1)
    matrix=np.zeros((4,4),int)
    for a,b in zip(labels,pred): matrix[a,b]+=1
    return {'accuracy':float(np.mean(pred==labels)), 'confusion':matrix.tolist(),
            'predictions':pred.tolist(),'mean_correct_probability':float(p[np.arange(len(labels)),labels].mean())}


def train(policy, x, labels, seed, steps, random_reward=False):
    rng=np.random.default_rng(seed); rewards_seen=[]
    start=time.perf_counter()
    for _ in range(steps):
        ix=rng.integers(len(x),size=8); batch=x[ix]
        probs=policy.predict(batch)
        actions=np.array([rng.choice(4,p=p/p.sum()) for p in probs])
        rewards=np.where(actions==labels[ix],1.,-1.)
        if random_reward: rewards=rng.permutation(rewards)
        policy.learn(batch,actions,rewards)
        rewards_seen.extend(rewards.tolist())
    return {'seconds':time.perf_counter()-start,'interactions':steps*8,'mean_training_reward':float(np.mean(rewards_seen))}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,default=ROOT/'results/gemma-bridge')
    parser.add_argument('--steps',type=int,default=512); args=parser.parse_args()
    out=args.out; rows=json.loads((out/'dataset.json').read_text()); e=load_file(str(out/'embeddings.safetensors'))['embeddings']
    ports=SemanticPorts(); rates=ports.encode(e)
    masks={s:np.array([r['split']==s for r in rows]) for s in ['train','validation','test','korean']}
    labels=np.array([r['label'] for r in rows]); x={s:rates[m] for s,m in masks.items()}; y={s:labels[m] for s,m in masks.items()}
    factories={'novi':lambda lr:NoviPolicy(ROOT,lr),'linear':lambda lr:LinearPolicy(rates.shape[1],lr)}
    report={'protocol':{'steps':args.steps,'batch':8,'reward':'1 correct, -1 incorrect','random_reward':'within-batch permutation preserves reward histogram','seeds':[101,202,303],'dimensions':128,'validation_seed':11,'encoder_frozen':True},'selection':{},'runs':[]}
    for kind,lrs in [('novi',[.03,.1,.3]),('linear',[.3,1.,3.])]:
        candidates=[]
        for lr in lrs:
            policy=factories[kind](lr); timing=train(policy,x['train'],y['train'],11,args.steps)
            val=score(policy,x['validation'],y['validation'])
            candidates.append({'lr':lr,'validation':val,'training':timing})
            print(kind,lr,val['accuracy'],flush=True)
        best=max(candidates,key=lambda c:(c['validation']['accuracy'],c['validation']['mean_correct_probability']))
        report['selection'][kind]={'candidates':candidates,'chosen_lr':best['lr']}
    for seed in [101,202,303]:
        # Arbitrary independently sampled actuator wiring, not inherent meaning in output index.
        mapping=np.random.default_rng(seed).permutation(4); reverse=(mapping+1)%4
        for kind in ['novi','linear']:
            for condition in ['reward','shuffled_reward']:
                policy=factories[kind](report['selection'][kind]['chosen_lr'])
                run={'seed':seed,'kind':kind,'condition':condition,'mapping':mapping.tolist(),
                     'initial':{s:score(policy,x[s],mapping[y[s]]) for s in ['test','korean']}}
                run['training']=train(policy,x['train'],mapping[y['train']],seed,args.steps,condition!='reward')
                run['after']={s:score(policy,x[s],mapping[y[s]]) for s in ['train','validation','test','korean']}
                if condition=='reward':
                    run['reversal_before']={s:score(policy,x[s],reverse[y[s]]) for s in ['test','korean']}
                    if kind=='novi':
                        dest=out/f'novi-{seed}.safetensors'; policy.engine.save(dest,overwrite=True)
                        restored=Engine.load(dest)
                        run['roundtrip_exact']=policy.engine.infer_batch(rates.tolist())==restored.infer_batch(rates.tolist())
                    run['reversal_training']=train(policy,x['train'],reverse[y['train']],seed+1,args.steps)
                    run['reversal_after']={s:score(policy,x[s],reverse[y[s]]) for s in ['test','korean']}
                report['runs'].append(run)
                (out/'experiment.json').write_text(json.dumps(report,indent=2)+'\n')
                print(seed,kind,condition,{s:v['accuracy'] for s,v in run['after'].items()},flush=True)
    # Frozen encoder nearest-training-example baseline: uses labeled exemplars, unlike reward policy.
    emb=e/np.linalg.norm(e,axis=1,keepdims=True)
    report['nearest_neighbor']={}
    for s in ['test','korean']:
        similarities=emb[masks[s]]@emb[masks['train']].T
        pred=y['train'][similarities.argmax(1)]
        report['nearest_neighbor'][s]={'accuracy':float(np.mean(pred==y[s])), 'max_train_cosine':similarities.max(1).tolist()}
    report['data_sha256']=hashlib.sha256((out/'dataset.json').read_bytes()).hexdigest()
    report['feature_sha256']=hashlib.sha256((out/'embeddings.safetensors').read_bytes()).hexdigest()
    (out/'experiment.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
