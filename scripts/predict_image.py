#!/usr/bin/env python3
"""Image bundle -> label-free adapter -> one Rust batch -> class/sequence decode."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageOps
from safetensors.numpy import save_file
from image_adapter import ImageRateAdapter


def predict(bundle_path, image_path, binary, invert=False):
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
    with tempfile.TemporaryDirectory(prefix='novigrad-input-') as temp:
        path = Path(temp)/'input.safetensors'
        save_file({'inputs':features,'input_ids':adapter.input_ids},str(path))
        result = subprocess.run([str(Path(binary).resolve()),str((bundle_path.parent/bundle['model']).resolve()),str(path)],capture_output=True,text=True,check=True)
    rows = list(csv.DictReader(io.StringIO(result.stdout),delimiter='\t'))
    if len(rows) != length:
        raise ValueError('unexpected number of backend predictions')
    classes = bundle['class_labels']
    indices = [int(row['predicted']) for row in rows]
    probabilities = [[float(row[f'p{i}']) for i in range(len(classes))] for row in rows]
    labels = [classes[index] for index in indices]
    return {'prediction':labels[0] if length==1 else ''.join(labels),'indices':indices,'probabilities':probabilities,'sequence_length':length,'segmentation':bundle['segmentation']}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('bundle',type=Path)
    parser.add_argument('image',type=Path)
    parser.add_argument('--binary',type=Path,default=Path('target/release/classify_features'))
    parser.add_argument('--invert',action='store_true')
    args=parser.parse_args()
    print(json.dumps(predict(args.bundle,args.image,args.binary,args.invert),indent=2))
