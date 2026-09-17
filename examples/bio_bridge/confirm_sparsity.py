"""New-case confirmation of a fixed 2%-versus-20% sparsity contrast; no retraining."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from safetensors.numpy import save_file,load_file
from novigrad import Engine
from embedding_scenarios import GOALS,sha,prototype_scores
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gemma_bridge'))
from bridge import SemanticPorts
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/bio-bridge'


def main():
    casespath=Path(__file__).with_name('sparsity_confirmation.json')
    rows=json.loads(casespath.read_text())
    model=SentenceTransformer('/Users/a405394/models/google_embeddinggemma-300m',device='mps',local_files_only=True)
    embeddings=model.encode([r['text'] for r in rows],batch_size=16,prompt_name='Classification',normalize_embeddings=True)
    torch.mps.synchronize();del model;torch.mps.empty_cache()
    save_file({'embeddings':embeddings.astype(np.float32)},str(OUT/'confirmation-features.safetensors'))
    x=SemanticPorts(319,128).encode(embeddings).tolist();labels=np.array([GOALS.index(r['class']) for r in rows])
    results=[]
    for seed in [601,602,603]:
        for fraction in [.02,.2]:
            path=OUT/f'sparse-d128-f{fraction}-s{seed}.safetensors'
            engine=Engine.load(path);before=engine.weights
            pred=np.asarray(engine.infer_batch(x)).argmax(axis=1)
            assert before==engine.weights
            results.append({'seed':seed,'active_fraction':fraction,'accuracy':float(np.mean(pred==labels)),'predictions':pred.tolist(),'checkpoint':path.name,'checkpoint_sha256':sha(path),'weights_unchanged':True})
    summary={str(f):{'mean_accuracy':float(np.mean([r['accuracy'] for r in results if r['active_fraction']==f])),'by_seed':[r['accuracy'] for r in results if r['active_fraction']==f]} for f in [.02,.2]}
    diffs=[next(r['accuracy'] for r in results if r['seed']==s and r['active_fraction']==.02)-next(r['accuracy'] for r in results if r['seed']==s and r['active_fraction']==.2) for s in [601,602,603]]
    old=ROOT/'results/gemma-bridge'
    original=json.loads((old/'dataset.json').read_text());e=load_file(str(old/'embeddings.safetensors'))['embeddings']
    train=np.array([r['split']=='train' for r in original]);y=np.array([r['label'] for r in original])[train]
    prototypes={str(d):float(np.mean(prototype_scores(e[train],y,embeddings,d).argmax(axis=1)==labels)) for d in [128,768]}
    report={'supervised_prototype_accuracy':prototypes,'scope':'2%/20% contrast selected after original48known evaluation, then assessed on32 newly authored texts without reading priorcases/modeloutputs by caseauthor. No model retraining, checkpoint selection or new threshold search. Limited same-author-agent/domain confirmation, not broad biological proof.','cases_sha256':sha(casespath),'features_sha256':sha(OUT/'confirmation-features.safetensors'),'summary':summary,'paired_accuracy_deltas_by_training_seed':diffs,'runs':results}
    (OUT/'sparsity-confirmation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
