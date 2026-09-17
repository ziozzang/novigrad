#!/usr/bin/env python3
"""Reproduce the external recording import/calibration demo on frozen KC rates."""
import argparse,json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
from delayed_credit import ROOT,semantic_ports
from inhibition_mechanism import ShadowEngine,checkpoint,fit_dual_ridge,l2_rows,sha256
from neural_link_stability import (RIDGE,AffineCalibrator,RecordingBatch,apply_drift,
 apply_recording,drift_parameters,fit_recordings,load_recording,save_recording,verify_recording)

DEFAULT_OUT=ROOT/'results/neural-link-bridge/import-demo.json'

def run(output=DEFAULT_OUT):
 output=Path(output);output=output if output.is_absolute() else ROOT/output;output.parent.mkdir(parents=True,exist_ok=True)
 data_dir=output.parent/'import-demo-recordings';data_dir.mkdir(parents=True,exist_ok=True)
 rows=json.loads((ROOT/'results/gemma-bridge/dataset.json').read_text());emb=load_file(str(ROOT/'results/gemma-bridge/embeddings.safetensors'))['embeddings'];labels=np.array([r['label'] for r in rows],np.int64)
 shadow=ShadowEngine(checkpoint(601));hidden=shadow.inhibit(shadow.raw_hidden(semantic_ports(emb)),'native_topk_2',{})
 train=np.array([r['split']=='train' for r in rows]);val=np.array([r['split']=='validation' for r in rows]);tx,vx=hidden[train],hidden[val];ty,vy=labels[train],labels[val]
 params=drift_parameters(tx.shape[1]);train_observed=apply_drift(tx,'gain_offset',params);val_observed=apply_drift(vx,'gain_offset',params)
 channels=load_file(str(checkpoint(601)))['hidden_ids'].astype(np.int64);train_ids=np.flatnonzero(train).astype(np.int64);val_ids=np.flatnonzero(val).astype(np.int64);order=np.random.default_rng(771).permutation(len(train_ids))
 paths={'reference':data_dir/'train-reference.safetensors','observed':data_dir/'train-observed-shuffled.safetensors','validation':data_dir/'validation-observed.safetensors','calibration':data_dir/'affine.safetensors','corrected':data_dir/'validation-corrected.safetensors'}
 save_recording(paths['reference'],RecordingBatch(train_ids,train_ids.astype(float)*.01,tx,channels,ty))
 save_recording(paths['observed'],RecordingBatch(train_ids[order],train_ids[order].astype(float)*.01,train_observed[order],channels,ty[order]))
 save_recording(paths['validation'],RecordingBatch(val_ids,val_ids.astype(float)*.01,val_observed,channels,vy))
 fit_manifest=fit_recordings(paths['observed'],paths['reference'],paths['calibration'],RIDGE)
 apply_manifest=apply_recording(paths['calibration'],paths['validation'],paths['corrected']);verified=verify_recording(paths['calibration'],paths['validation'],paths['corrected'],str(paths['corrected'])+'.manifest.json')
 corrected=load_recording(paths['corrected']);cal=AffineCalibrator.load(paths['calibration']);direct=cal.infer(val_observed);exact=float(np.max(np.abs(np.asarray(direct,np.float32)-corrected.neural_features)))
 basis,coef=fit_dual_ridge(tx,ty,ridge=RIDGE,actions=4);class_w=basis.T@coef
 def metric(x):
  cosine=np.sum(x*vx,1)/np.maximum(np.linalg.norm(x,axis=1)*np.linalg.norm(vx,axis=1),1e-12)
  return {'cosine_to_clean_mean':float(cosine.mean()),'fixed_probe_accuracy':float(np.mean((l2_rows(x)@class_w).argmax(1)==vy))}
 try:
  load_recording(paths['validation'],channel_ids=channels[::-1]);negative={'rejected':False}
 except ValueError as error:negative={'rejected':True,'reason':str(error)}
 source=Path(__file__);report={'status':'complete external import -> ID-aligned fit -> save/load -> apply -> verify on simulated KC recordings','scope':'engineering demo; not implanted data or reproduction of cited BCI methods','data':{'train_anchors':32,'validation_rows':12,'features':tx.shape[1],'drift':'gain_offset','train_observed_rows_shuffled':True},'checks':{'serialized_vs_direct_max_error':exact,'verified':verified['verified'] and exact==0,'channel_order_negative_control':negative},'metrics':{'uncorrected':metric(val_observed),'corrected':metric(corrected.neural_features)},'artifacts':{str(p.relative_to(ROOT)):{'sha256':sha256(p)} for p in paths.values()},'manifests':{'fit':fit_manifest,'apply':apply_manifest},'provenance':{str(p.relative_to(ROOT)):sha256(p) for p in [source,Path(__file__).with_name('neural_link_stability.py'),ROOT/'results/gemma-bridge/dataset.json',ROOT/'results/gemma-bridge/embeddings.safetensors',checkpoint(601)]}}
 output.write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT_OUT);args=parser.parse_args();print(json.dumps(run(args.out),indent=2))
