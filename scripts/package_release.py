#!/usr/bin/env python3
"""Package prebuilt Apple Silicon binaries and selected self-contained model bundles."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
BINARIES=('novi','novi_engine','novi_rt','novi_api','train_connectome','train_classifier','classify_features','odor_motor')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',default='0.1.1')
    parser.add_argument('--binary-dir',type=Path,default=ROOT/'target/distribution/release')
    parser.add_argument('--out',type=Path,default=ROOT/'dist/v0.1.1')
    args=parser.parse_args()
    if not re.fullmatch(r'\d+\.\d+\.\d+',args.version):raise ValueError('version must be X.Y.Z')
    args.out.mkdir(parents=True,exist_ok=True)
    name=f'novigrad-v{args.version}-aarch64-apple-darwin'
    folder=args.out/name
    folder.mkdir(exist_ok=True)
    for binary in BINARIES:shutil.copy2(args.binary_dir/binary,folder/binary)
    for doc in ('LICENSE','NOTICE.md'):shutil.copy2(ROOT/doc,folder/doc)
    (folder/'README.md').write_text(f'''# Novigrad v{args.version} — Software-defined Bionic NPU

Author: jioh jung <jung@jioh.net>. Source code license: MIT.

Apple Silicon CPU executables built with target-cpu=apple-m1, tested on Apple M2 Ultra.
These are unsigned, unnotarized research binaries. Source builds use target-cpu=native by default.

- `novi`: synthetic learning and full-graph activation demo.
- `novi_engine`: generic sparse-port train/eval/infer/bench CLI.
- `novi_rt`: soft real-time periodic inference; no hard deadline guarantee.
- `novi_api`: local HTTP inference and opt-in learning/checkpoints.
- `train_classifier`, `classify_features`: generic feature training/inference.
- `train_connectome`, `odor_motor`: controlled reward-learning examples.

The model archive extracts at the source repository root. Optimized vision models are
`examples/vision/models/fashion-optimized/` and `captcha-optimized/`.
Image adapters require Python dependencies from `requirements-vision.txt` in the source repository.

Full English/Korean documentation: https://github.com/ziozzang/novigrad/tree/v{args.version}

한국어: 실행 파일 이름은 novi입니다. 모델 묶음은 소스 저장소 루트에 풀어 사용합니다.
주기 실행은 soft real-time이며, 생물학적 운동 제어나 hard RTOS를 입증한 것은 아닙니다.
''')
    metadata={'version':args.version,'author':'jioh jung <jung@jioh.net>','license':'MIT',
              'target':'aarch64-apple-darwin','target_cpu':'apple-m1','tested_on':'Apple M2 Ultra',
              'rustc':subprocess.check_output(['rustc','--version'],text=True).strip(),
              'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'cargo_lock_sha256':hashlib.sha256((ROOT/'Cargo.lock').read_bytes()).hexdigest(),
              'binaries':{binary:hashlib.sha256((folder/binary).read_bytes()).hexdigest() for binary in BINARIES}}
    (folder/'BUILD.json').write_text(json.dumps(metadata,indent=2)+'\n')
    binary_archive=args.out/f'{name}.tar.gz'
    with tarfile.open(binary_archive,'w:gz') as archive:archive.add(folder,arcname=name)
    model_archive=args.out/f'novigrad-v{args.version}-models.tar.gz'
    with tarfile.open(model_archive,'w:gz') as archive:
        for task in ('fashion','captcha','fashion-optimized','captcha-optimized'):
            for filename in ('adapter.safetensors','model.safetensors','bundle.json'):
                path=Path('examples/vision/models')/task/filename
                archive.add(ROOT/path,arcname=str(path))
        for task in ('binary','fourway','xor'):
            for filename in (f'{task}.safetensors',f'{task}_ports.tsv'):
                archive.add(ROOT/'results/v1-confirmation'/filename,arcname=f'models/connectome/{filename}')
        archive.add(ROOT/'results/neuromodulation/reward-seed1-episodes300.safetensors',arcname='models/neuromodulation/odor-motor.safetensors')
        for filename in ('LICENSE','NOTICE.md','results/connectome-data-manifest.json','results/vision-data-manifest.json',
                         'results/improvement/selection.json','results/improvement/holdout-manifest.json',
                         'results/improvement/final-summary.json'):
            archive.add(ROOT/filename,arcname=f'models/provenance/{Path(filename).name}')
    archives=[binary_archive,model_archive]
    (args.out/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in archives))
    print(json.dumps({p.name:p.stat().st_size for p in archives},indent=2))

if __name__=='__main__':main()
