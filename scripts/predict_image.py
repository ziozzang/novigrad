#!/usr/bin/env python3
"""Image bundle -> label-free adapter -> one Rust batch -> class/sequence decode."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageOps
from safetensors.numpy import save_file
from image_adapter import ImageRateAdapter


def predict(bundle_path, image_path, binary, invert=False, api_url=None, model=None):
    bundle_path = Path(bundle_path)
    bundle = json.loads(bundle_path.read_text())
    if bundle.get('format') != 'nobi.image_bundle.v1':
        raise ValueError('unsupported image bundle')
    adapter = ImageRateAdapter.load(bundle_path.parent/bundle['adapter'])
    image = Image.open(image_path).convert('L')
    if invert:
        image = ImageOps.invert(image)
    length = int(bundle['sequence_length'])
    if length == 1:
        images = np.asarray(image.resize((28,28),Image.Resampling.BILINEAR),dtype=np.uint8)[None]
    else:
        if image.size != (length*28,28):
            raise ValueError(f'this fixed-cell example requires exactly {length*28}x28 pixels')
        images = np.asarray(image,dtype=np.uint8).reshape(28,length,28).transpose(1,0,2)
    features = adapter.transform(images)
    classes = bundle['class_labels']
    if api_url:
        if not model:
            raise ValueError('--model is required with --api-url')
        url = api_url.rstrip('/') + '/v1/models/' + urllib.parse.quote(model, safe='') + '/infer'
        body = {'inputs':features.tolist(),'input_ids':[str(int(i)) for i in adapter.input_ids]}
        request = urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'},method='POST')
        with urllib.request.urlopen(request,timeout=30) as response:
            result = json.load(response)
        indices = result['actions']
        probabilities = result['probabilities']
    else:
        with tempfile.TemporaryDirectory(prefix='novigrad-input-') as temp:
            path = Path(temp)/'input.safetensors'
            save_file({'inputs':features,'input_ids':adapter.input_ids},str(path))
            result = subprocess.run([str(Path(binary).resolve()),str((bundle_path.parent/bundle['model']).resolve()),str(path)],capture_output=True,text=True,check=True)
        rows = list(csv.DictReader(io.StringIO(result.stdout),delimiter='\t'))
        indices = [int(row['predicted']) for row in rows]
        probabilities = [[float(row[f'p{i}']) for i in range(len(classes))] for row in rows]
    if len(indices) != length or len(probabilities) != length:
        raise ValueError('unexpected number of backend predictions')
    if any(type(i) is not int or not 0 <= i < len(classes) for i in indices):
        raise ValueError('backend class index out of range')
    p = np.asarray(probabilities,dtype=np.float64)
    if p.shape != (length,len(classes)) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(axis=1),1,atol=1e-6):
        raise ValueError('invalid backend probabilities')
    labels = [classes[index] for index in indices]
    return {'prediction':labels[0] if length==1 else ''.join(labels),'indices':indices,'probabilities':probabilities,'sequence_length':length,'segmentation':bundle['segmentation']}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('bundle',type=Path)
    parser.add_argument('image',type=Path)
    parser.add_argument('--binary',type=Path,default=Path('target/release/classify_features'))
    parser.add_argument('--invert',action='store_true')
    parser.add_argument('--api-url',help='Use a running novi_api instead of a local Rust binary')
    parser.add_argument('--model',help='Registered API model name')
    args=parser.parse_args()
    print(json.dumps(predict(args.bundle,args.image,args.binary,args.invert,args.api_url,args.model),indent=2))
