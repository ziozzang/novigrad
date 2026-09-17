"""Mechanism-inspired channel adaptation before an unchanged native policy.

These are engineered temporal filters, not fitted fly receptor dynamics.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from novigrad import Engine
from signal_recall import load_cases, checkpoint, sha, metrics, ROOT, SEEDS
from delayed_credit import load_data

OUT = ROOT/'results/mechanism-bridge'
MODES = ['direct','divisive_ema','subtractive_ema','fixed_divisive','shuffled_subtractive','oracle_background']


def adapt(rows, mode, training_mean, background=None, tau=8., floor=.05):
    rows=np.asarray(rows,np.float32); training_mean=np.asarray(training_mean,np.float32)
    if rows.ndim!=2 or not np.isfinite(rows).all() or (rows<0).any():
        raise ValueError('finite nonnegative time x ports required')
    if mode not in MODES or not np.isfinite(tau) or tau<=0 or not np.isfinite(floor) or floor<=0:
        raise ValueError('invalid mechanism or time/gain constant')
    if training_mean.shape!=(rows.shape[1],) or not np.isfinite(training_mean).all() or (training_mean<0).any():
        raise ValueError('invalid fixed channel means')
    if mode=='oracle_background' and (background is None or np.asarray(background).shape!=rows.shape):
        raise ValueError('oracle needs matched background rows')
    state=np.zeros(rows.shape[1],np.float32); alpha=np.float32(np.exp(-1/tau))
    # Shuffle only populated embedding ports, retaining unused zero-port padding.
    permutation=np.arange(rows.shape[1]); permutation[:min(256,len(permutation))]=np.random.default_rng(431).permutation(min(256,len(permutation)))
    outputs=[]
    for t,row in enumerate(rows):
        if mode=='direct': y=row.copy()
        elif mode=='divisive_ema': y=row/(floor+state)
        elif mode=='subtractive_ema': y=np.maximum(row-state,0)
        elif mode=='fixed_divisive': y=row/(floor+training_mean)
        elif mode=='shuffled_subtractive': y=np.maximum(row-state[permutation],0)
        else: y=np.maximum(row-background[t],0)
        outputs.append(y)
        # State update follows emission: no current/future sample in background estimate.
        state=alpha*state+(1-alpha)*row
    return np.asarray(outputs)


def build_episodes(x, labels, strength, schedule, drift):
    mask=np.zeros(48,bool)
    if schedule=='sustained8': mask[12:20]=True;mask[32:40]=True
    elif schedule=='pulse1':mask[[12,32]]=True
    else:raise ValueError('unknown schedule')
    episodes=[]; backgrounds=[]
    for i,label in enumerate(labels):
        a=np.flatnonzero(labels==(label+1)%4);b=np.flatnonzero(labels==(label+2)%4)
        background=np.repeat((strength*x[a[i%len(a)]])[None,:],48,axis=0)
        if drift:background[24:]=strength*x[b[i%len(b)]]
        episode=background.copy();episode[mask]+=x[i]
        episodes.append(episode);backgrounds.append(background)
    return np.stack(episodes),np.stack(backgrounds),mask


def score(engine, episodes, backgrounds, mask, labels, mode, training_mean):
    z=np.stack([adapt(e,mode,training_mean,b) for e,b in zip(episodes,backgrounds)])
    p=np.asarray(engine.infer_batch(z.reshape(-1,z.shape[-1]).tolist())).reshape(len(z),48,4)
    target=np.repeat(labels[:,None],48,axis=1)
    norms=np.linalg.norm(z,axis=-1)
    return {'foreground':metrics(p[:,mask],target[:,mask]),
            'onset':metrics(p[:,[12,32]],target[:,[12,32]]),
            'accuracy_by_tick':(p.argmax(-1)==target).mean(0).tolist(),
            'correct_probability_by_tick':np.take_along_axis(p,target[...,None],-1)[...,0].mean(0).tolist(),
            'mean_output_norm_by_tick':norms.mean(0).tolist(),
            'background_only_mean_norm':float(norms[:,~mask].mean()),
            'foreground_mean_norm':float(norms[:,mask].mean()),
            'state_boundary':'No detection gate: a small nonzero background residual can still drive normalized Engine output'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',action='store_true');args=parser.parse_args()
    start=time.perf_counter();x,labels,provenance=load_cases();train=load_data()['train'][0]
    training_mean=train.mean(0)
    report={'design':{'mechanism':'causal per-channel EMA adaptation at host input; not physiological ORN fit',
        'tau_logical_ticks':8,'divisive_floor':.05,'modes':MODES,'background_strengths':[.25,1.,4.],
        'schedules':['sustained8','pulse1'],'drifting_background':[False,True],
        'primary':'foreground greedy accuracy, static background1 sustained8, subtractive EMA vs direct',
        'evaluation':'reused32confirmationtexts, frozen128D2% checkpoints; no new generalization claim',
        'foreground':'same target in two encounters; background is class+1 then optional class+2 after tick24',
        'oracle':'exact background vector known only to oracle comparator, no causal estimator has access',
        'no_learning':True,'settings':'fixed before measurements, no result-based threshold/time-constant selection',
        'source_sha256':sha(Path(__file__)), 'provenance':provenance,
        'training_feature_sha256':sha(ROOT/'results/gemma-bridge/embeddings.safetensors'),
        'checkpoints':{str(checkpoint(s).relative_to(ROOT)):sha(checkpoint(s)) for s in SEEDS}},'runs':[]}
    settings=[(1.,'sustained8',False)] if args.baseline else [(g,s,d) for g in [.25,1.,4.] for s in ['sustained8','pulse1'] for d in [False,True]]
    for seed in SEEDS:
        engine=Engine.load(checkpoint(seed));weights=engine.weights
        for strength,schedule,drift in settings:
            episodes,backgrounds,mask=build_episodes(x,labels,strength,schedule,drift)
            for mode in ['direct'] if args.baseline else MODES:
                result=score(engine,episodes,backgrounds,mask,labels,mode,training_mean)
                report['runs'].append(dict(seed=seed,mode=mode,background_strength=strength,schedule=schedule,drift=drift,**result))
        assert weights==engine.weights
    report['primary']={m:float(np.mean([r['foreground']['accuracy'] for r in report['runs'] if (r['mode'],r['background_strength'],r['schedule'],r['drift'])==(m,1.,'sustained8',False)])) for m in (['direct'] if args.baseline else MODES)}
    report['runtime_seconds']=time.perf_counter()-start
    OUT.mkdir(parents=True,exist_ok=True);name='adaptation-baseline' if args.baseline else 'adaptation'
    (OUT/f'{name}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'primary':report['primary'],'seconds':report['runtime_seconds']}))


if __name__=='__main__':main()
