"""Encode the finalized bilingual holdout without predictions or label fitting."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path('/Users/a405394/models/google_embeddinggemma-300m')
DEFAULT_CASES = ROOT / 'examples/bio_bridge/precise_holdout.json'
DEFAULT_OUT = ROOT / 'results/precise-bridge'
CLASSES = ('water', 'food', 'warmth', 'rest')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def provenance_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def validate_cases(rows):
    if not isinstance(rows, list) or len(rows) != 64:
        raise ValueError('expected exactly 64 holdout rows')
    required = {'id', 'text', 'class', 'language', 'family', 'pair_id'}
    if any(not isinstance(row, dict) or not required <= row.keys() or
           any(not isinstance(row[key], str) or not row[key].strip() for key in required)
           for row in rows):
        raise ValueError('every row requires nonempty string metadata and text')
    if len({row['id'] for row in rows}) != 64 or len({row['text'] for row in rows}) != 64:
        raise ValueError('duplicate row IDs or texts')
    counts = Counter((row['class'], row['language']) for row in rows)
    if counts != Counter({(label, lang): 8 for label in CLASSES for lang in ('en', 'ko')}):
        raise ValueError('expected eight rows per class and language')
    pairs = defaultdict(list)
    families = defaultdict(set)
    for row in rows:
        pairs[row['pair_id']].append(row)
        families[row['class']].add(row['family'])
    if len(pairs) != 32 or any(len(value) != 8 for value in families.values()):
        raise ValueError('expected 32 pairs and eight families per class')
    if len({frozenset(value) for value in families.values()}) != 1:
        raise ValueError('all classes must cover the same eight families')
    seen = set()
    for pair in pairs.values():
        if len(pair) != 2 or {row['language'] for row in pair} != {'en', 'ko'}:
            raise ValueError('each pair requires one English and one Korean row')
        identity = {(row['class'], row['family']) for row in pair}
        if len(identity) != 1 or next(iter(identity)) in seen:
            raise ValueError('pair categories/families must match and be unique')
        seen.update(identity)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=DEFAULT_CASES)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    rows = validate_cases(json.loads(args.cases.read_text()))
    output = args.out / 'holdout-embeddings.safetensors'
    provenance = args.out / 'provenance.json'
    if output.exists() or provenance.exists():
        raise FileExistsError('refusing to overwrite existing holdout artifacts')
    if not args.model.is_dir():
        raise ValueError('model must be an existing local directory')

    import numpy as np
    import torch
    from safetensors.numpy import save_file
    from sentence_transformers import SentenceTransformer

    model_files = sorted(path for path in args.model.rglob('*') if path.is_file()
                         and path.suffix in ('.safetensors', '.json', '.model', '.txt'))
    hashes = {str(path.relative_to(args.model)): sha256(path) for path in model_files}
    if not any(name.endswith('.safetensors') for name in hashes):
        raise ValueError('local model has no Safetensors weights')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    start = time.perf_counter()
    model = SentenceTransformer(str(args.model), device=device, local_files_only=True)
    load_seconds = time.perf_counter() - start
    dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if dtypes != ['torch.float32']:
        raise ValueError(f'expected original FP32 model, got {dtypes}')
    texts = [row['text'] for row in rows]
    model.encode(texts[:2], batch_size=2, prompt_name='Classification', normalize_embeddings=True)
    if device == 'mps':
        torch.mps.synchronize()
    start = time.perf_counter()
    embeddings = model.encode(texts, batch_size=16, prompt_name='Classification',
                              normalize_embeddings=True).astype(np.float32, copy=False)
    if device == 'mps':
        torch.mps.synchronize()
    encode_seconds = time.perf_counter() - start
    if embeddings.shape != (64, 768) or not np.isfinite(embeddings).all():
        raise ValueError('expected finite 64x768 embeddings')
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-5):
        raise ValueError('embeddings are not normalized')
    args.out.mkdir(parents=True, exist_ok=True)
    save_file({'embeddings': embeddings}, str(output), metadata={
        'format': 'novigrad.semantic_holdout_embeddings', 'version': '1',
        'prompt_name': 'Classification', 'normalized': 'true', 'dtype': 'float32'})
    report = {
        'purpose': 'Frozen held-out semantic scenarios; encoding only, no predictions or fitted mapping.',
        'external_validity': 'Authored, not external data; 32 correlated English/Korean translation pairs.',
        'generator': provenance_path(__file__), 'generator_sha256': sha256(__file__),
        'cases_path': provenance_path(args.cases), 'cases_sha256': sha256(args.cases),
        'model': provenance_path(args.model), 'model_file_sha256': hashes,
        'weight_sha256': {name: digest for name, digest in hashes.items() if name.endswith('.safetensors')},
        'embedding_path': provenance_path(output), 'embedding_sha256': sha256(output),
        'feature_bytes_sha256': hashlib.sha256(embeddings.tobytes(order='C')).hexdigest(),
        'row_ids': [row['id'] for row in rows],
        'row_order': 'embedding row i corresponds to cases JSON row i',
        'rows': 64, 'pairs': 32, 'shape': list(embeddings.shape), 'dtype': str(embeddings.dtype),
        'normalized': True, 'norm_min': float(norms.min()), 'norm_max': float(norms.max()),
        'prompt_name': 'Classification', 'prompt_text': model.prompts['Classification'],
        'device': device, 'parameter_dtypes': dtypes, 'batch_size': 16,
        'model_load_seconds': load_seconds, 'encode_seconds': encode_seconds, 'encoder_frozen': True,
        'versions': {name: importlib.metadata.version(name) for name in
                     ('torch', 'transformers', 'sentence-transformers', 'safetensors', 'numpy')},
    }
    provenance.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
