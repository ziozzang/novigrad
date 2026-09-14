#!/usr/bin/env python3
"""Label-free image -> signed PCA rate ports, persisted as Safetensors.

One reusable adapter for object images and character crops. Fit uses training
images only. No pretrained model, OCR service, or target label enters features.
"""
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


class ImageRateAdapter:
    def __init__(self, mean, scale, projection, component_scale, input_ids, port_feature_indices=None):
        self.mean = mean
        self.scale = scale
        self.projection = projection
        self.component_scale = component_scale
        self.input_ids = input_ids
        self.port_feature_indices = (np.arange(len(input_ids)-1) % (2*projection.shape[1])).astype(np.int64) if port_feature_indices is None else port_feature_indices

    @staticmethod
    def features(images):
        images = np.asarray(images)
        if images.ndim != 3 or images.shape[1:] != (28, 28) or images.dtype != np.uint8:
            raise ValueError('images must be uint8 [N,28,28]')
        # Gradients and 7x7 local orientation histograms; 14x14 pooled pixels.
        # Arrays are batched in callers to bound memory.
        x = images.astype(np.float32) / 255
        dx = np.empty_like(x); dy = np.empty_like(x)
        dx[:, :, 1:-1] = (x[:, :, 2:] - x[:, :, :-2]) * .5
        dx[:, :, 0] = x[:, :, 1] - x[:, :, 0]
        dx[:, :, -1] = x[:, :, -1] - x[:, :, -2]
        dy[:, 1:-1, :] = (x[:, 2:, :] - x[:, :-2, :]) * .5
        dy[:, 0, :] = x[:, 1, :] - x[:, 0, :]
        dy[:, -1, :] = x[:, -1, :] - x[:, -2, :]
        magnitude = np.hypot(dx, dy)
        angle = np.mod(np.arctan2(dy, dx), np.pi) * (9 / np.pi)
        lower = np.floor(angle).astype(np.int32) % 9
        fraction = angle - np.floor(angle)
        hist = np.empty((len(x), 7, 7, 9), dtype=np.float32)
        for orientation in range(9):
            votes = magnitude * ((lower == orientation) * (1-fraction) + (((lower+1) % 9) == orientation) * fraction)
            hist[..., orientation] = votes.reshape(len(x), 7, 4, 7, 4).sum(axis=(2, 4))
        hist /= np.sqrt(np.sum(hist*hist, axis=-1, keepdims=True) + .04)
        pooled = x.reshape(len(x), 14, 2, 14, 2).mean(axis=(2,4))
        return np.concatenate([pooled.reshape(len(x), -1), hist.reshape(len(x), -1)], axis=1)

    @classmethod
    def fit(cls, images, input_ids, components=None, whitening=1.0):
        input_ids = np.asarray(input_ids, dtype=np.uint64)
        if input_ids.ndim != 1 or len(input_ids) < 3 or len(set(input_ids.tolist())) != len(input_ids):
            raise ValueError('input_ids must be unique vector of at least three IDs')
        features = np.concatenate([cls.features(images[i:i+512]) for i in range(0,len(images),512)])
        components = min(components or (len(input_ids)-1)//2, (len(input_ids)-1)//2, features.shape[1], len(features)-1)
        if not 0 <= whitening <= 1:
            raise ValueError("whitening must be in [0,1]")
        if components < 1:
            raise ValueError('need at least two training images')
        mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(features.std(axis=0, dtype=np.float64), .05).astype(np.float32)
        standardized = ((features-mean)/scale).astype(np.float64)
        covariance = standardized.T @ standardized / (len(features)-1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        projection = eigenvectors[:, -components:][:, ::-1].copy()
        # Canonical signs keep saved projections stable across LAPACK sign choices.
        for column in range(components):
            anchor = np.argmax(np.abs(projection[:, column]))
            if projection[anchor, column] < 0:
                projection[:, column] *= -1
        component_scale = np.power(np.maximum(eigenvalues[-components:][::-1], .01), .5*whitening).astype(np.float32)
        return cls(mean, scale, projection.astype(np.float32), component_scale, input_ids)

    def transform(self, images):
        output = np.empty((len(images), len(self.input_ids)), dtype=np.float32)
        width = self.projection.shape[1]
        for start in range(0,len(images),512):
            raw = self.features(images[start:start+512])
            z = ((raw-self.mean)/self.scale) @ self.projection
            z /= self.component_scale
            z = np.clip(z / 3, -1, 1)
            rates = output[start:start+len(raw)]
            rates.fill(0)
            signed = np.concatenate([np.maximum(z,0), np.maximum(-z,0)],axis=1)
            rates[:, :-1] = signed[:, self.port_feature_indices]
            rates[:, -1] = 1
        return output

    def save(self, path):
        save_file({'mean': self.mean, 'scale': self.scale, 'projection': self.projection,
                   'component_scale': self.component_scale, 'input_ids': self.input_ids, 'port_feature_indices':self.port_feature_indices}, str(path),
                  metadata={'format':'nobi.image_rate.v1','image_shape':'28,28',
                            'features':'pooled14x14+unsigned9bin-cell4-gradient',
                            'fit_scope':'training images only; no labels',
                            'rate_encoding':'PCA fractional whitening; clip +/-3 scale; repeated positive/negative ports; constant final port'})

    @classmethod
    def load(cls, path):
        with safe_open(str(path), framework='numpy') as file:
            if file.metadata().get('format') != 'nobi.image_rate.v1':
                raise ValueError('unsupported image adapter')
            arrays = {key:file.get_tensor(key) for key in file.keys()}
        return cls(**arrays)


def prepare(npz_path, input_edges, output_dir, components=None, whitening=1.0):
    data = np.load(npz_path, allow_pickle=False)
    # Same exact sorted external port order as Rust Engine::load.
    input_ids = np.array(sorted({int(line.split('\t')[0]) for line in Path(input_edges).read_text().splitlines()}), dtype=np.uint64)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    train = data['train_images']
    sequence = train.ndim == 4 or train.shape[-1] != 28
    sequence_length = train.shape[1] if train.ndim == 4 else train.shape[-1]//28
    def flatten_images(x):
        if not sequence:
            return x
        if x.ndim == 4:
            return x.reshape(-1,28,28)
        return x.reshape(len(x),28,sequence_length,28).transpose(0,2,1,3).reshape(-1,28,28)
    adapter = ImageRateAdapter.fit(flatten_images(train), input_ids, components, whitening)
    adapter.save(output_dir/'adapter.safetensors')
    tensors = {'input_ids':input_ids}
    for split in ['train','validation','test']:
        images = data[split+'_images']
        tensors[split+'_inputs'] = adapter.transform(flatten_images(images))
        tensors[split+'_labels'] = data[split+'_labels'].reshape(-1).astype(np.int64)
    save_file(tensors,str(output_dir/'features.safetensors'),metadata={'format':'nobi.classifier_dataset.v1','source_npz':Path(npz_path).name,'sequence_length':str(sequence_length if sequence else 1)})
    # Bundle is task metadata, not any answer-bearing feature input.
    bundle={'format':'nobi.image_bundle.v1','adapter':'adapter.safetensors','model':'model.safetensors','sequence_length':int(sequence_length if sequence else 1),'cell_size':[28,28],'segmentation':'fixed non-overlapping cells' if sequence else 'whole image','class_labels':(['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'] if 'fashion' in Path(npz_path).stem else [str(i) for i in range(10)]), 'components':adapter.projection.shape[1], 'whitening':whitening}
    (output_dir/'bundle.json').write_text(json.dumps(bundle,indent=2)+'\n')
    # Independent split counts and adapter roundtrip.
    restored = ImageRateAdapter.load(output_dir/'adapter.safetensors')
    np.testing.assert_array_equal(adapter.transform(flatten_images(data['test_images'])[:16]), restored.transform(flatten_images(data['test_images'])[:16]))
    print(json.dumps({'features':str(output_dir/'features.safetensors'),'input_ports':len(input_ids),'pca_components':adapter.projection.shape[1],'sequence_length':bundle['sequence_length'],'samples':{s:len(tensors[s+'_labels']) for s in ['train','validation','test']}}))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('images',type=Path)
    parser.add_argument('--input-edges',type=Path,default=Path('data/pn_kc.tsv'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--components',type=int,default=64)
    parser.add_argument('--whitening',type=float,default=.5)
    args=parser.parse_args()
    prepare(args.images,args.input_edges,args.out,args.components,args.whitening)
