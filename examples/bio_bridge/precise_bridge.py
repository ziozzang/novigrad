"""Train-only embedding calibration through fixed fly-derived PN-KC connectivity."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file, load_file
from thought_embedding import unit, fit_alignment, decode, candidate_data, evaluation, GOALS
from inhibition_mechanism import ShadowEngine, checkpoint, sha256
from delayed_credit import ROOT

OUT=ROOT/'results/precise-bridge'
MODES=('fixed','centered','pca')
RIDGES=(.01,.1,1.)

class PortBridge:
    def __init__(self,mean,projection,mode):
        self.mean=np.ascontiguousarray(mean,dtype=np.float32)
        self.projection=np.ascontiguousarray(projection,dtype=np.float32)
        self.mode=mode
    @classmethod
    def fit(cls,train,mode):
        x=np.asarray(train,dtype=np.float32)
        if x.ndim!=2 or len(x)<2 or x.shape[1]!=768 or not np.isfinite(x).all():
            raise ValueError('finite 768-dimensional training rows required')
        if mode not in MODES:raise ValueError('unknown mode')
        mean=np.zeros(768,np.float32) if mode=='fixed' else x.mean(0)
        projection=np.eye(768,128,dtype=np.float32)
        if mode=='pca':
            _,s,v=np.linalg.svd(x-mean,full_matrices=False)
            rank=min(128,int(np.sum(s>s[0]*1e-6)),len(x)-1)
            projection=np.zeros((768,128),np.float32)
            projection[:,:rank]=v[:rank].T
            # Resolve arbitrary SVD signs to make port assignment reproducible.
            for j in range(rank):
                if projection[np.argmax(np.abs(projection[:,j])),j]<0:projection[:,j]*=-1
        return cls(mean,projection,mode)
    def encode(self,embeddings):
        x=np.asarray(embeddings,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=768 or not np.isfinite(x).all():raise ValueError('finite 768-dimensional embeddings required')
        z=(x-self.mean)@self.projection
        z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
        out=np.zeros((len(x),319),np.float32)
        out[:,:128]=np.maximum(z,0);out[:,128:256]=np.maximum(-z,0)
        return out
    def save(self,path):
        save_file({'mean':self.mean,'projection':self.projection},str(path),metadata={'format':'novigrad.embedding_port_bridge','mode':self.mode,'version':'1'})
    @classmethod
    def load(cls,path,mode):
        a=load_file(str(path));return cls(a['mean'],a['projection'],mode)

def dataset():
    rows=json.loads((ROOT/'results/gemma-bridge/dataset.json').read_text())
    x=load_file(str(ROOT/'results/gemma-bridge/embeddings.safetensors'))['embeddings']
    return {s:(x[[r['split']==s for r in rows]],np.array([r['label'] for r in rows if r['split']==s]),[r['text'] for r in rows if r['split']==s]) for s in ('train','validation')}

def hidden(shadow,ports):return shadow.inhibit(shadow.raw_hidden(ports),'native_topk_2',{})

def score(pred,x,y,text):
    descriptions,vectors=candidate_data()
    return evaluation(pred,unit(x[:,:128]),y,text,descriptions,vectors)

def development(modes):
    OUT.mkdir(exist_ok=True);data=dataset();x,y,text=data['train'];vx,vy,vt=data['validation']
    shadow=ShadowEngine(checkpoint(601));report={}
    for mode in modes:
        bridge=PortBridge.fit(x,mode);ports=bridge.encode(x);h=hidden(shadow,ports);vh=hidden(shadow,bridge.encode(vx))
        trials=[]
        for ridge in RIDGES:
            b,c=fit_alignment(h,x[:,:128],ridge)
            r=score(decode(b,c,vh),vx,vy,vt)
            trials.append({'ridge':ridge,'accuracy':r['semantic_category_accuracy'],'cosine':r['mean_cosine_to_input_embedding']})
        best=max(trials,key=lambda r:(r['accuracy'],r['cosine'],-r['ridge']))
        b,c=fit_alignment(h,x[:,:128],best['ridge'])
        bridge.save(OUT/f'ports-{mode}.safetensors')
        restored=PortBridge.load(OUT/f'ports-{mode}.safetensors',mode)
        assert np.array_equal(ports,restored.encode(x))
        save_file({'basis':np.ascontiguousarray(b),'coefficients':np.ascontiguousarray(c)},str(OUT/f'decoder-{mode}.safetensors'))
        a=load_file(str(OUT/f'decoder-{mode}.safetensors'));assert np.array_equal(decode(b,c,vh),decode(a['basis'],a['coefficients'],vh))
        report[mode]={'selected':best,'trials':trials,'train':score(decode(b,c,h),x,y,text),'validation':score(decode(b,c,vh),vx,vy,vt),'projection_rank':int(np.linalg.matrix_rank(bridge.projection)), 'train_active_kcs':float(np.count_nonzero(h,axis=1).mean()),'roundtrip_exact':True}
    selection={'protocol':'All transforms fit train32 only; ridge selected validation12 accuracy then cosine then larger regularization. No final test read here.', 'modes':report,'source_sha256':sha256(__file__),'input_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'results/gemma-bridge/dataset.json',ROOT/'results/gemma-bridge/embeddings.safetensors',checkpoint(601)]}}
    (OUT/('baseline.json' if modes==('fixed',) else 'development.json')).write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps({k:v['selected'] for k,v in report.items()}))

def final():
    dev=json.loads((OUT/'development.json').read_text())
    rows=json.loads((ROOT/'examples/bio_bridge/precise_holdout.json').read_text())
    if isinstance(rows,dict):rows=rows['cases']
    x=load_file(str(OUT/'holdout-embeddings.safetensors'))['embeddings'];y=np.array([GOALS.index(r['class']) for r in rows]);text=[r['text'] for r in rows]
    shadow=ShadowEngine(checkpoint(601));results={}
    for mode in MODES:
        bridge=PortBridge.load(OUT/f'ports-{mode}.safetensors',mode);a=load_file(str(OUT/f'decoder-{mode}.safetensors'))
        start=time.perf_counter();h=hidden(shadow,bridge.encode(x));pred=decode(a['basis'],a['coefficients'],h);elapsed=time.perf_counter()-start
        results[mode]={'all':score(pred,x,y,text),'languages':{lang:score(pred[ix],x[ix],y[ix],[t for t,i in zip(text,ix) if i]) for lang in ('en','ko') for ix in [np.array([r['language']==lang for r in rows])]},'batch_bridge_seconds':elapsed}
    results['raw_embedding']={'all':score(unit(x[:,:128]),x,y,text)}
    report={'boundary':'Authored bilingual scenario test, not real neural recordings or external benchmark. Translation pairs are dependent. One final evaluation; no retuning.', 'development_sha256':sha256(OUT/'development.json'),'source_sha256':sha256(__file__),'results':results}
    (OUT/'final.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({m:r['all']['semantic_category_accuracy'] for m,r in results.items()}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',action='store_true');parser.add_argument('--final',action='store_true');args=parser.parse_args()
    if args.final:final()
    else:development(('fixed',) if args.baseline else MODES)
