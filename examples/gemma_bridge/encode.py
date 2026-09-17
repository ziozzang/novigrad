"""Prepare frozen original-float32 Gemma features; no encoder fine-tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import importlib.metadata
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from safetensors.numpy import save_file
from data import records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('results/gemma-bridge'))
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    rows = records(); texts = [r['text'] for r in rows]
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    start = time.perf_counter()
    model = SentenceTransformer(str(args.model), device=device, local_files_only=True)
    load = time.perf_counter() - start
    model.encode(texts[:2], prompt_name='Classification', normalize_embeddings=True)
    if device == 'mps': torch.mps.synchronize()
    start = time.perf_counter()
    embeddings = model.encode(texts, batch_size=16, prompt_name='Classification', normalize_embeddings=True)
    if device == 'mps': torch.mps.synchronize()
    elapsed = time.perf_counter()-start
    timings = []
    for text in texts[:8]:
        start=time.perf_counter()
        model.encode([text], prompt_name='Classification', normalize_embeddings=True)
        if device == 'mps': torch.mps.synchronize()
        timings.append((time.perf_counter()-start)*1000)
    save_file({'embeddings': embeddings.astype(np.float32)}, str(args.out/'embeddings.safetensors'))
    (args.out/'dataset.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
    hashes={}
    for path in sorted(args.model.rglob('*.safetensors')):
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''): digest.update(chunk)
        hashes[str(path.relative_to(args.model))]=digest.hexdigest()
    report={'model':str(args.model),'weight_sha256':hashes,'device':device,
        'parameter_dtypes':sorted({str(v.dtype) for v in model.parameters()}),
        'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','sentence-transformers']},
        'rows':len(rows),'shape':list(embeddings.shape),'load_seconds':load,'encode_seconds':elapsed,
        'batch_size':16,'single_text_ms':timings,'prompt_name':'Classification','encoder_frozen':True}
    (args.out/'encoder.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
