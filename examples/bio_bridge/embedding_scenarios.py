"""Frozen EmbeddingGemma geometry, abstention calibration, and KC sparsity audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from safetensors.numpy import load_file,save_file
from novigrad import Engine
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gemma_bridge'))
from bridge import SemanticPorts
ROOT=Path(__file__).resolve().parents[2]
GOALS=['water','food','warmth','rest']
UNKNOWN_CALIBRATION=[
 'Please open the calculator.','What is the current time?',
 'Food and water are equally urgent; I cannot pick one.',
 'Warmth or sleep would help equally; no priority is set.',
 '계산기를 열어 주세요.','현재 시각은 무엇인가요?',
 '먹는 것과 마시는 것이 똑같이 급하고 우선순위가 없습니다.',
 '보온과 잠이 똑같이 필요해서 하나를 고르지 못했습니다.']


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def unit(x,dimensions):
    z=np.asarray(x[:,:dimensions],np.float32).copy()
    return z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)


def prototype_scores(train,labels,test,dimensions):
    tr=unit(train,dimensions);te=unit(test,dimensions)
    prototypes=np.stack([tr[labels==k].mean(axis=0) for k in range(4)])
    prototypes/=np.maximum(np.linalg.norm(prototypes,axis=1,keepdims=True),1e-12)
    return te@prototypes.T


def margin_predictions(scores,threshold):
    ordered=np.sort(scores,axis=1)
    pred=scores.argmax(axis=1).astype(int)
    pred[ordered[:,-1]-ordered[:,-2]<threshold]=-1
    return pred


def recognition_metrics(pred,labels):
    known=labels>=0;unknown=~known
    valid=float(np.mean(pred[known]==labels[known])) if known.any() else None
    reject=float(np.mean(pred[unknown]==-1)) if unknown.any() else None
    return {'known_accuracy':valid,'unknown_rejection':reject,'balanced_accuracy':(valid+reject)/2 if valid is not None and reject is not None else None,'overall_accuracy':float(np.mean(pred==labels)),'predictions':pred.tolist()}


def train_policy(train,labels,dimensions,active_fraction,seed,steps=512):
    engine=Engine.from_edges(ROOT/'data/pn_kc.tsv',ROOT/'data/kc_mbon.tsv',actions=4,learning_rate=.3,logit_gain=1.,active_fraction=active_fraction,homeostasis=False,readout='opponent')
    x=SemanticPorts(319,dimensions).encode(train)
    rng=np.random.default_rng(seed)
    for _ in range(steps):
        ix=rng.integers(len(x),size=8);batch=x[ix]
        p=np.asarray(engine.infer_batch(batch.tolist()))
        u=rng.random(8);actions=(u[:,None]>np.cumsum(p/p.sum(axis=1,keepdims=True),axis=1)).sum(axis=1).clip(0,3)
        rewards=np.where(actions==labels[ix],1.,-1.)
        engine.learn(batch.tolist(),actions.tolist(),rewards.tolist())
    return engine


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,default=Path('/Users/a405394/models/google_embeddinggemma-300m'))
    p.add_argument('--out',type=Path,default=ROOT/'results/bio-bridge')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    casespath=Path(__file__).with_name('embedding_cases.json');functions=Path(__file__).with_name('cases.json')
    cases=json.loads(casespath.read_text());frows=json.loads(functions.read_text())
    old=ROOT/'results/gemma-bridge';oldrows=json.loads((old/'dataset.json').read_text());e=load_file(str(old/'embeddings.safetensors'))['embeddings']
    trainmask=np.array([r['split']=='train' for r in oldrows]);valmask=np.array([r['split']=='validation' for r in oldrows]);oldlabels=np.array([r['label'] for r in oldrows])
    assert not {r['text'] for r in oldrows if r['split']=='train'}&{r['text'] for r in cases},'exact training overlap'
    encoder=SentenceTransformer(str(a.model),device='mps',local_files_only=True)
    texts=[r['text'] for r in cases]+UNKNOWN_CALIBRATION+[r['messages'][-1]['content'] for r in frows]
    start=time.perf_counter();vectors=encoder.encode(texts,batch_size=16,prompt_name='Classification',normalize_embeddings=True)
    torch.mps.synchronize();seconds=time.perf_counter()-start
    dtype=sorted({str(v.dtype) for v in encoder.parameters()})
    del encoder;torch.mps.empty_cache()
    test=vectors[:len(cases)];unknown=vectors[len(cases):len(cases)+len(UNKNOWN_CALIBRATION)];fvectors=vectors[len(cases)+len(UNKNOWN_CALIBRATION):]
    save_file({'test':test.astype(np.float32),'calibration_unknown':unknown.astype(np.float32),'function_latest_message':fvectors.astype(np.float32)},str(a.out/'embedding-features.safetensors'))
    (a.out/'calibration-unknown.json').write_text(json.dumps(UNKNOWN_CALIBRATION,ensure_ascii=False,indent=2)+'\n')
    train=e[trainmask];y=oldlabels[trainmask];labels=np.array([GOALS.index(r['class']) if r['class']!='none' else -1 for r in cases])
    calibration=np.concatenate([e[valmask],unknown]);clabels=np.concatenate([oldlabels[valmask],np.full(len(unknown),-1)])
    report={'design':{'goals':GOALS,'encoder_frozen':True,'encoder_dtypes':dtype,'encode_seconds':seconds,'train_source':'previous32 labeled training utterances, reward policies receive only actual-action feedback','calibration':'previous12 validation utterances plus8 separately declared unknown/ambiguous examples','test':'60 new independently authored bilingual texts; none includes ambiguity as well as unsupported requests','seeds':[601,602,603],'updates':512,'batch':8,'interactions_per_model':4096,'active_fractions':[.02,.1,.2,.5],'dimensions':[64,128],'novi_unknown_class':'not trained; four-action policy evaluated on48 known texts only','sparsity_scope':'global top-k rate-code proxy; not APL inhibitory dynamics','dimension_scope':'128/768 are documented EmbeddingGemma dimensions;64 is an experimental truncation below documented MRL sizes, not an advertised supported mode'},'sha256':{'test_cases':sha(casespath),'function_cases':sha(functions),'old_dataset':sha(old/'dataset.json'),'old_features':sha(old/'embeddings.safetensors'),'features':sha(a.out/'embedding-features.safetensors'),'calibration_unknown':sha(a.out/'calibration-unknown.json')},'model_sha256':{str(f.relative_to(a.model)):sha(f) for f in a.model.rglob('*.safetensors')},'prototype':{},'novi_runs':[]}
    for d in [64,128,768]:
        cs=prototype_scores(train,y,calibration,d)
        candidates=[{'threshold':float(t),'balanced_accuracy':recognition_metrics(margin_predictions(cs,t),clabels)['balanced_accuracy']} for t in np.linspace(0,.3,31)]
        chosen=max(candidates,key=lambda r:(r['balanced_accuracy'],-r['threshold']))['threshold']
        scores=prototype_scores(train,y,test,d);pred=margin_predictions(scores,chosen)
        result={'threshold':chosen,'calibration_grid':candidates,'forced_known':recognition_metrics(scores.argmax(axis=1),labels),'calibrated':recognition_metrics(pred,labels),'by_kind':{},'by_language':{}}
        for field,dest in [('kind','by_kind'),('language','by_language')]:
            for kind in sorted({r[field] for r in cases}):
                ix=np.array([r[field]==kind for r in cases]);result[dest][kind]=recognition_metrics(pred[ix],labels[ix])
        report['prototype'][str(d)]=result
        if d==768:
            fscores=prototype_scores(train,y,fvectors,d);fpred=margin_predictions(fscores,chosen)
            report['function_semantic_predictions']=[{'id':r['id'],'goal':GOALS[k] if k>=0 else None,'margin':float(np.sort(row)[-1]-np.sort(row)[-2])} for r,k,row in zip(frows,fpred,fscores)]
    known=labels>=0
    for d in [64,128]:
        rates=SemanticPorts(319,d).encode(test)
        for fraction in [.02,.1,.2,.5]:
            for seed in [601,602,603]:
                start=time.perf_counter();engine=train_policy(train,y,d,fraction,seed)
                probs=np.asarray(engine.infer_batch(rates.tolist()));pred=probs.argmax(axis=1)
                checkpoint=a.out/f'sparse-d{d}-f{fraction}-s{seed}.safetensors';engine.save(checkpoint,overwrite=True)
                report['novi_runs'].append({'dimensions':d,'active_fraction':fraction,'seed':seed,'known_accuracy':float(np.mean(pred[known]==labels[known])),'by_kind':{k:float(np.mean(pred[ix]==labels[ix])) for k in sorted({r['kind'] for r in cases}) if (ix:=np.array([r['kind']==k for r in cases])&known).any()},'by_language':{k:float(np.mean(pred[ix]==labels[ix])) for k in ['en','ko'] if (ix:=np.array([r['language']==k for r in cases])&known).any()},'predictions':pred.tolist(),'seconds':time.perf_counter()-start,'checkpoint':checkpoint.name,'checkpoint_sha256':sha(checkpoint),'config':engine.config})
                (a.out/'embedding.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                print(d,fraction,seed,report['novi_runs'][-1]['known_accuracy'],flush=True)
    print('prototype',{k:v['calibrated']['balanced_accuracy'] for k,v in report['prototype'].items()},flush=True)
if __name__=='__main__':main()
