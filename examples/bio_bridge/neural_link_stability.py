#!/usr/bin/env python3
"""Paired-anchor affine calibration of simulated frozen KC recording drift."""
import argparse, hashlib, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file
from delayed_credit import ROOT, semantic_ports
from inhibition_mechanism import ShadowEngine, checkpoint, fit_dual_ridge, l2_rows, sha256

OUT=ROOT/'results/neural-link-bridge'; CONDITIONS=('clean','gain_offset','dropout25','dropout50'); ANCHORS=(8,16,32); RIDGE=.1; SEED=4401

@dataclass(frozen=True)
class RecordingBatch:
    sample_ids: np.ndarray; time: np.ndarray; neural_features: np.ndarray; channel_ids: np.ndarray; targets: np.ndarray|None=None
    def validate(self,feature_count=None,channel_ids=None):
        ids=np.asarray(self.sample_ids); t=np.asarray(self.time); x=np.asarray(self.neural_features); channels=np.asarray(self.channel_ids)
        if ids.ndim!=1 or ids.dtype.kind not in 'iu': raise ValueError('sample_ids must be a 1D integer array')
        if t.ndim!=1 or t.dtype.kind not in 'iuf' or not np.isfinite(t).all(): raise ValueError('time must be a finite 1D numeric array in seconds')
        if x.ndim!=2 or x.dtype.kind not in 'iuf' or x.shape[1]==0 or not np.isfinite(x).all(): raise ValueError('neural_features must be a finite nonempty 2D numeric array')
        if channels.ndim!=1 or channels.dtype.kind not in 'iu' or len(channels)!=x.shape[1] or len(np.unique(channels))!=len(channels): raise ValueError('channel_ids must be unique 1D integers, one per feature')
        n=len(ids)
        if n==0 or len(t)!=n or len(x)!=n: raise ValueError('recording arrays must share a nonzero row count')
        if len(np.unique(ids))!=n: raise ValueError('sample_ids must be unique')
        if feature_count is not None and x.shape[1]!=feature_count: raise ValueError('feature count mismatch')
        if channel_ids is not None and not np.array_equal(channels,np.asarray(channel_ids)): raise ValueError('channel identity/order mismatch')
        if self.targets is not None:
            y=np.asarray(self.targets)
            if y.ndim!=1 or y.dtype.kind not in 'iu' or len(y)!=n: raise ValueError('targets must be 1D integer class IDs with matching rows')
        return self

def save_recording(path,batch):
    batch.validate();features=np.asarray(batch.neural_features,np.float32)
    if not np.isfinite(features).all():raise ValueError('neural_features exceed the serialized float32 range')
    tensors={'sample_ids':np.asarray(batch.sample_ids,np.int64),'time':np.asarray(batch.time,np.float64),'neural_features':features,'channel_ids':np.asarray(batch.channel_ids,np.int64)}
    if batch.targets is not None:tensors['targets']=np.asarray(batch.targets,np.int64)
    path=Path(path)
    if path.suffix=='.npz':np.savez(path,**tensors)
    elif path.suffix=='.safetensors':save_file(tensors,str(path),metadata={'format':'novi.recording','version':'2','time_unit':'seconds','targets_semantics':'class_ids' if batch.targets is not None else 'absent'})
    else:raise ValueError('recording must be .npz or .safetensors')

def load_recording(path,feature_count=None,channel_ids=None):
    path=Path(path)
    if path.suffix=='.npz': a=dict(np.load(path,allow_pickle=False))
    elif path.suffix=='.safetensors':
        with safe_open(str(path),framework='numpy') as f: meta=f.metadata() or {}
        if meta.get('format')!='novi.recording' or meta.get('version')!='2' or meta.get('time_unit')!='seconds' or meta.get('targets_semantics') not in ('class_ids','absent'): raise ValueError('invalid recording metadata')
        a=load_file(str(path))
        if (meta['targets_semantics']=='class_ids')!=('targets' in a):raise ValueError('targets metadata disagrees with tensors')
    else:a=None
    if a is None:raise ValueError('recording must be .npz or .safetensors')
    required={'sample_ids','time','neural_features','channel_ids'}
    if not required<=a.keys() or set(a)-required-{'targets'}:raise ValueError('invalid recording schema')
    return RecordingBatch(a['sample_ids'],a['time'],a['neural_features'],a['channel_ids'],a.get('targets')).validate(feature_count,channel_ids)

class AffineCalibrator:
    def __init__(self,anchors,dual,ridge=RIDGE,channel_ids=None):
        self.anchors=np.ascontiguousarray(anchors,np.float64);self.dual=np.ascontiguousarray(dual,np.float64);self.ridge=float(ridge)
        if self.anchors.ndim!=2 or self.dual.ndim!=2 or len(self.anchors)==0 or self.anchors.shape[0]!=self.dual.shape[0] or self.anchors.shape[1]<2 or not np.isfinite(self.anchors).all() or not np.isfinite(self.dual).all() or not np.isfinite(self.ridge) or self.ridge<=0: raise ValueError('invalid calibrator tensors or ridge')
        self.channel_ids=np.arange(self.input_features,dtype=np.int64) if channel_ids is None else np.asarray(channel_ids)
        if self.channel_ids.ndim!=1 or self.channel_ids.dtype.kind not in 'iu' or len(self.channel_ids)!=self.input_features or len(np.unique(self.channel_ids))!=len(self.channel_ids): raise ValueError('invalid calibrator channel_ids')
    @property
    def input_features(self):return self.anchors.shape[1]-1
    @property
    def output_features(self):return self.dual.shape[1]
    @classmethod
    def fit(cls,observed,reference,ridge=RIDGE,channel_ids=None):
        obs=np.asarray(observed); y=np.asarray(reference)
        if obs.ndim!=2 or y.ndim!=2 or len(obs)==0 or obs.shape!=y.shape or obs.dtype.kind not in 'iuf' or y.dtype.kind not in 'iuf' or not np.isfinite(obs).all() or not np.isfinite(y).all() or not np.isfinite(ridge) or ridge<=0:raise ValueError('invalid paired anchors')
        x=np.c_[np.asarray(obs,np.float64),np.ones(len(obs))]
        return cls(x,np.linalg.solve(x@x.T+ridge*np.eye(len(x)),np.asarray(y,np.float64)),ridge,channel_ids)
    def infer(self,observed):
        obs=np.asarray(observed)
        if obs.ndim!=2 or obs.shape[1]!=self.input_features or obs.dtype.kind not in 'iuf' or not np.isfinite(obs).all():raise ValueError('invalid observed feature matrix')
        return np.c_[np.asarray(obs,np.float64),np.ones(len(obs))]@self.anchors.T@self.dual
    def save(self,path):save_file({'anchors':self.anchors,'dual':self.dual,'channel_ids':np.asarray(self.channel_ids,np.int64)},str(path),metadata={'format':'novi.paired-affine-ridge','version':'2','ridge':repr(self.ridge),'input_features':str(self.input_features),'output_features':str(self.output_features)})
    @classmethod
    def load(cls,path):
        with safe_open(str(path),framework='numpy') as f:meta=f.metadata() or {}
        if meta.get('format')!='novi.paired-affine-ridge' or meta.get('version')!='2':raise ValueError('invalid calibrator metadata')
        try: ridge=float(meta['ridge']); ni=int(meta['input_features']); no=int(meta['output_features'])
        except (KeyError,ValueError):raise ValueError('invalid calibrator metadata')
        a=load_file(str(path))
        if set(a)!={'anchors','dual','channel_ids'}:raise ValueError('invalid calibrator tensor schema')
        c=cls(a['anchors'],a['dual'],ridge,a['channel_ids'])
        if (c.input_features,c.output_features)!=(ni,no) or ni!=no:raise ValueError('calibrator dimensions disagree with same-channel metadata contract')
        return c

def align_paired(observed,reference,time_tolerance=0.0):
    observed.validate();reference.validate()
    if not np.isfinite(time_tolerance) or time_tolerance<0:raise ValueError('time tolerance must be finite and nonnegative')
    if not np.array_equal(observed.channel_ids,reference.channel_ids):raise ValueError('paired recordings require identical channel identity and order')
    oi=np.asarray(observed.sample_ids);ri=np.asarray(reference.sample_ids)
    if set(oi.tolist())!=set(ri.tolist()):raise ValueError('paired recordings have missing or extra sample IDs')
    lookup={int(v):i for i,v in enumerate(ri)}; order=np.array([lookup[int(v)] for v in oi])
    if np.any(np.abs(np.asarray(observed.time,dtype=float)-np.asarray(reference.time,dtype=float)[order])>time_tolerance):raise ValueError('paired sample times exceed tolerance')
    return np.asarray(observed.neural_features),np.asarray(reference.neural_features)[order]

def _file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def _manifest_path(output):return Path(str(output)+'.manifest.json')
def _write_manifest(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n')

def fit_recordings(observed_path,reference_path,calibration_path,ridge=RIDGE,time_tolerance=0.0,manifest_path=None):
    observed=load_recording(observed_path);reference=load_recording(reference_path)
    x,y=align_paired(observed,reference,time_tolerance)
    calibration_path=Path(calibration_path);calibration_path.parent.mkdir(parents=True,exist_ok=True)
    cal=AffineCalibrator.fit(x,y,ridge,observed.channel_ids);cal.save(calibration_path)
    restored=AffineCalibrator.load(calibration_path)
    if not np.array_equal(restored.infer(x),cal.infer(x)):raise RuntimeError('calibration roundtrip failed')
    manifest={'operation':'fit','schema_version':1,'inputs':{'observed':{'path':str(observed_path),'sha256':_file_sha(observed_path)},'reference':{'path':str(reference_path),'sha256':_file_sha(reference_path)}},'calibration':{'path':str(calibration_path),'sha256':_file_sha(calibration_path)},'paired_rows':len(x),'features':x.shape[1],'ridge':float(ridge),'time_tolerance_seconds':float(time_tolerance),'alignment':'sample_id with identical channel_ids/order'}
    _write_manifest(manifest_path or _manifest_path(calibration_path),manifest);return manifest

def apply_recording(calibration_path,input_path,output_path,manifest_path=None):
    cal=AffineCalibrator.load(calibration_path);batch=load_recording(input_path,cal.input_features,cal.channel_ids)
    corrected=cal.infer(batch.neural_features);out=RecordingBatch(batch.sample_ids,batch.time,corrected,cal.channel_ids,batch.targets)
    output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True);save_recording(output_path,out)
    loaded=load_recording(output_path,cal.output_features,cal.channel_ids)
    # Recording files standardize features to float32, so verify the serialized representation.
    if not np.array_equal(loaded.neural_features,np.asarray(corrected,np.float32)):raise RuntimeError('corrected recording roundtrip failed')
    manifest={'operation':'apply','schema_version':1,'inputs':{'calibration':{'path':str(calibration_path),'sha256':_file_sha(calibration_path)},'recording':{'path':str(input_path),'sha256':_file_sha(input_path)}},'output':{'path':str(output_path),'sha256':_file_sha(output_path)},'rows':len(batch.sample_ids),'input_features':cal.input_features,'output_features':cal.output_features,'channel_policy':'exact identity and order required'}
    _write_manifest(manifest_path or _manifest_path(output_path),manifest);return manifest

def verify_recording(calibration_path,input_path,output_path,manifest_path=None):
    cal=AffineCalibrator.load(calibration_path);source=load_recording(input_path,cal.input_features,cal.channel_ids);actual=load_recording(output_path,cal.output_features,cal.channel_ids)
    if not np.array_equal(source.sample_ids,actual.sample_ids) or not np.array_equal(source.time,actual.time) or not np.array_equal(np.asarray(cal.infer(source.neural_features),np.float32),actual.neural_features):raise ValueError('output does not match calibration applied to input')
    if (source.targets is None)!=(actual.targets is None) or source.targets is not None and not np.array_equal(source.targets,actual.targets):raise ValueError('output class IDs do not match input')
    if manifest_path:
        m=json.loads(Path(manifest_path).read_text())
        if m.get('operation')!='apply' or m.get('inputs',{}).get('calibration',{}).get('sha256')!=_file_sha(calibration_path) or m.get('inputs',{}).get('recording',{}).get('sha256')!=_file_sha(input_path) or m.get('output',{}).get('sha256')!=_file_sha(output_path):raise ValueError('manifest provenance mismatch')
    return {'verified':True,'rows':len(source.sample_ids),'features':cal.output_features,'max_error':0.0}

def drift_parameters(d):
    rng=np.random.default_rng(SEED); gain=np.exp(rng.normal(0,.35,d));offset=rng.normal(0,.08,d);order=rng.permutation(d)
    return gain.astype(np.float32),offset.astype(np.float32),order
def apply_drift(x,condition,params):
    gain,offset,order=params
    if condition=='clean':return x.copy()
    y=x*gain+offset if condition=='gain_offset' else x.copy()
    if condition.startswith('dropout'):
        fraction=.25 if condition=='dropout25' else .5;y=y.copy();y[:,order[:int(len(order)*fraction)]]=0
    return y
def anchor_indices(labels,n):
    return np.concatenate([np.flatnonzero(labels==c)[:n//4] for c in range(4)])
def metrics(pred,clean,scores,labels):
    cosine=np.sum(pred*clean,1)/(np.maximum(np.linalg.norm(pred,axis=1)*np.linalg.norm(clean,axis=1),1e-12))
    return {'cosine':float(cosine.mean()),'rmse':float(np.sqrt(np.mean((pred-clean)**2))), 'accuracy':float(np.mean(scores.argmax(1)==labels))}

def run(output=OUT/'stability.json'):
    output=Path(output);output=output if output.is_absolute() else ROOT/output
    rows=json.loads((ROOT/'results/gemma-bridge/dataset.json').read_text()); emb=load_file(str(ROOT/'results/gemma-bridge/embeddings.safetensors'))['embeddings'];labels=np.array([r['label'] for r in rows])
    shadow=ShadowEngine(checkpoint(601)); rates=semantic_ports(emb); hidden=shadow.inhibit(shadow.raw_hidden(rates),'native_topk_2',{})
    masks={s:np.array([r['split']==s for r in rows]) for s in ('train','validation')};tx,ty=hidden[masks['train']],labels[masks['train']];vx,vy=hidden[masks['validation']],labels[masks['validation']]
    basis,coef=fit_dual_ridge(tx,ty,ridge=RIDGE,actions=4);class_w=basis.T@coef;params=drift_parameters(tx.shape[1]);runs=[];output.parent.mkdir(parents=True,exist_ok=True)
    for condition in CONDITIONS:
      train_obs=apply_drift(tx,condition,params);val_obs=apply_drift(vx,condition,params)
      for n in ANCHORS:
        ix=anchor_indices(ty,n);cal=AffineCalibrator.fit(train_obs[ix],tx[ix]);corrected=cal.infer(val_obs); shuffled=AffineCalibrator.fit(train_obs[ix],tx[ix[np.random.default_rng(SEED+n).permutation(len(ix))]]).infer(val_obs)
        path=output.parent/f'calibrator-{condition}-n{n}.safetensors';cal.save(path);restored=AffineCalibrator.load(path);err=float(np.max(np.abs(restored.infer(val_obs)-corrected)));assert err==0
        runs.append({'condition':condition,'anchors':n,'uncorrected':metrics(val_obs,vx,l2_rows(val_obs)@class_w,vy),'corrected':metrics(corrected,vx,l2_rows(corrected)@class_w,vy),'sham_shuffled':metrics(shuffled,vx,l2_rows(shuffled)@class_w,vy),'artifact':{'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'roundtrip_max_error':err}})
    report={'status':'simplified paired-anchor affine ridge calibration on simulated KC drift; not Neuralink, Degenhart, or NoMAD reproduction','protocol':{'ridge':RIDGE,'seed':SEED,'anchors':list(ANCHORS),'conditions':list(CONDITIONS),'calibration':'old train only; validation evaluation only; settings frozen before results','decoder':'fixed supervised class probe trained on clean old train','sham':'clean anchor pairing shuffled deterministically'},'runs':runs,'provenance':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'results/gemma-bridge/dataset.json',ROOT/'results/gemma-bridge/embeddings.safetensors',checkpoint(601)]}}
    output.write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 study=sub.add_parser('study');study.add_argument('--output',type=Path,default=OUT/'stability.json')
 fit=sub.add_parser('fit');fit.add_argument('--observed',required=True);fit.add_argument('--reference',required=True);fit.add_argument('--calibration',required=True);fit.add_argument('--ridge',type=float,default=RIDGE);fit.add_argument('--time-tolerance',type=float,default=0.0);fit.add_argument('--manifest')
 apply=sub.add_parser('apply');apply.add_argument('--calibration',required=True);apply.add_argument('--input',required=True);apply.add_argument('--output',required=True);apply.add_argument('--manifest')
 verify=sub.add_parser('verify');verify.add_argument('--calibration',required=True);verify.add_argument('--input',required=True);verify.add_argument('--output',required=True);verify.add_argument('--manifest')
 a=p.parse_args()
 if a.command=='study':result=run(a.output)['runs']
 elif a.command=='fit':result=fit_recordings(a.observed,a.reference,a.calibration,a.ridge,a.time_tolerance,a.manifest)
 elif a.command=='apply':result=apply_recording(a.calibration,a.input,a.output,a.manifest)
 else:result=verify_recording(a.calibration,a.input,a.output,a.manifest)
 print(json.dumps(result,indent=2))
