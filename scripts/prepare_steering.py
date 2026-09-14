#!/usr/bin/env python3
"""Download a small, session-disjoint CARLA imitation demo from pinned HF viewer JPEGs.

Uses image pixels only; steering is a target, never an adapter input. This is an
offline imitation demo, not an autonomous driving or closed-loop evaluation.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = 'urjc-deepracer/carla-expert-racing'
REVISION = '4f993c8f052a9c94fcabbe4da63fc3eacf097c18'
DIRECTORY = ROOT / 'data' / 'steering'
OUTPUT = DIRECTORY / 'carla_racing.npz'
MANIFEST = ROOT / 'results' / 'steering' / 'data-manifest.json'
SOURCE_SPLITS = {'train': 'train', 'validation': 'valid', 'test': 'test'}
SOURCE_COUNTS = {'train': 20697, 'valid': 18623, 'test': 7776}
MAX_DOWNLOAD = 200_000_000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Downloader:
    def __init__(self):
        self.bytes = 0
        self.lock = threading.Lock()
        self.retry_lock = threading.Lock()
        self.retry_after = 0.0

    def get(self, url, limit=1_000_000):
        for attempt in range(8):
            with self.retry_lock:
                delay = max(0.0, self.retry_after - time.monotonic())
            if delay:
                time.sleep(delay)
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'Novigrad-steering-demo/0.1'})
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = response.read(limit + 1)
                with self.lock:
                    self.bytes += len(data)
                    require(self.bytes <= MAX_DOWNLOAD, '200 MB download cap exceeded')
                require(len(data) <= limit, 'Response exceeds per-file size cap')
                return data
            except (urllib.error.URLError, TimeoutError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code not in (429, 500, 502, 503, 504):
                    raise
                if attempt == 7:
                    raise
                if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                    with self.retry_lock:
                        self.retry_after = max(self.retry_after, time.monotonic() + 45)
                    print(json.dumps({'http_status': 429, 'retry_after_seconds': 45, 'attempt': attempt + 1}), flush=True)
                else:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError('Unreachable download state')

    def json(self, url):
        return json.loads(self.get(url))


def viewer_url(endpoint, **parameters):
    return 'https://datasets-server.huggingface.co/' + endpoint + '?' + urllib.parse.urlencode({'dataset': REPO, 'config': 'default', **parameters})


def fetch_row(downloader, split, index):
    cache = DIRECTORY / 'source' / split
    cache.mkdir(parents=True, exist_ok=True)
    metadata_path = cache / f'{index:06d}.json'
    image_path = cache / f'{index:06d}.jpg'
    if metadata_path.exists() and image_path.exists():
        metadata = json.loads(metadata_path.read_text())
        jpeg = image_path.read_bytes()
        require(metadata['revision'] == REVISION and metadata['row_index'] == index and metadata['source_split'] == split,
                'Cached sample provenance mismatch')
        require(digest(jpeg) == metadata['jpeg_sha256'], 'Cached JPEG hash mismatch')
    else:
        block = getattr(downloader, 'blocks', {}).get((split, index))
        response = block if block is not None else downloader.json(viewer_url('rows', split=split, offset=index, length=1))
        require(not response.get('partial') and response['num_rows_total'] == SOURCE_COUNTS[split], 'Viewer split changed')
        matching = [item for item in response['rows'] if item['row_idx'] == index]
        require(len(matching) == 1, 'Viewer returned incorrect row')
        row = matching[0]['row']
        asset = row['image_path']['src']
        parsed = urllib.parse.urlsplit(asset)
        expected_path = f'/cached-assets/{REPO}/--/{REVISION}/--/default/{split}/{index}/image_path/image.jpg'
        require(parsed.scheme == 'https' and parsed.netloc == 'datasets-server.huggingface.co' and parsed.path == expected_path,
                'Viewer image source revision or row does not match pin')
        steer = float(row['steer'])
        timestamp = float(row['timestamp'])
        require(np.isfinite(steer) and -1 <= steer <= 1 and np.isfinite(timestamp), 'Invalid steering target or timestamp')
        require(isinstance(row['experiment_id'], str) and row['experiment_id'], 'Missing experiment_id')
        jpeg = downloader.get(asset)
        metadata = {'repository': REPO, 'revision': REVISION, 'source_split': split, 'row_index': index,
                    'experiment_id': row['experiment_id'], 'timestamp': timestamp, 'steer': steer,
                    'asset_path': expected_path, 'jpeg_sha256': digest(jpeg), 'jpeg_bytes': len(jpeg),
                    'source_width': row['image_path']['width'], 'source_height': row['image_path']['height']}
        with Image.open(io.BytesIO(jpeg)) as image:
            require(image.format == 'JPEG' and image.size == (metadata['source_width'], metadata['source_height']), 'Unexpected viewer image')
        # Persist independent cache entries for safe retry after a network failure.
        image_path.with_suffix('.jpg.part').write_bytes(jpeg)
        image_path.with_suffix('.jpg.part').replace(image_path)
        metadata_path.with_suffix('.json.part').write_text(json.dumps(metadata, indent=2) + '\n')
        metadata_path.with_suffix('.json.part').replace(metadata_path)
    with Image.open(io.BytesIO(jpeg)) as image:
        prepared = np.asarray(image.convert('L').resize((28, 28), Image.Resampling.BILINEAR), dtype=np.uint8)
    return metadata, prepared


def prepare(counts, workers):
    if OUTPUT.exists():
        require(MANIFEST.exists(), 'Dataset exists without provenance manifest')
        manifest = json.loads(MANIFEST.read_text())
        require(manifest['requested_counts'] == counts and manifest['revision'] == REVISION, 'Existing dataset configuration differs')
        require(manifest['artifact_sha256'] == digest(OUTPUT.read_bytes()), 'Existing dataset checksum mismatch')
        print(json.dumps({'verified_existing': counts, 'dataset': str(OUTPUT.relative_to(ROOT))}))
        return
    downloader = Downloader()
    repo_info = downloader.json(f'https://huggingface.co/api/datasets/{REPO}?blobs=true')
    require(repo_info['sha'] == REVISION, 'HF main revision changed; viewer cannot be used without a fresh provenance audit')
    source_groups = {}
    for split in SOURCE_SPLITS.values():
        stats = downloader.json(viewer_url('statistics', split=split))
        require(not stats.get('partial') and stats['num_examples'] == SOURCE_COUNTS[split], 'Unexpected split statistics')
        group_stats = next(item['column_statistics'] for item in stats['statistics'] if item['column_name'] == 'experiment_id')
        require(group_stats['nan_count'] == 0, 'Source rows have missing session groups')
        source_groups[split] = group_stats['frequencies']
    for first, second in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        require(not set(source_groups[first]) & set(source_groups[second]), 'Source splits share experiment IDs')
    selected = {}
    tasks = []
    downloader.blocks = {}
    require(counts == {'train': 100, 'validation': 50, 'test': 50}, 'This bounded demo uses fixed 100/50/50 sampling')
    for split, source in SOURCE_SPLITS.items():
        require(0 < counts[split] <= SOURCE_COUNTS[source], 'Requested count outside source bounds')
        if split == 'train':
            indices = np.linspace(0, SOURCE_COUNTS[source] - 1, 300, dtype=np.int64)[:100]
        else:
            offsets = [0, 2000] if split == 'validation' else [0, 4000]
            indices = np.concatenate([offset + np.linspace(0, 99, 25, dtype=np.int64) for offset in offsets])
        # Fetch one metadata page for each group of missing image rows. This
        # preserves the exact fixed indices but lowers clean-clone request count.
        missing_pages = {}
        for index in indices:
            cache = DIRECTORY / 'source' / source
            if not ((cache / f'{index:06d}.json').exists() and (cache / f'{index:06d}.jpg').exists()):
                missing_pages.setdefault(int(index) // 100 * 100, []).append(int(index))
        for offset, page_indices in sorted(missing_pages.items()):
            block = downloader.json(viewer_url('rows', split=source, offset=offset, length=100))
            # Signed URLs remain only in memory; per-row cache metadata retains
            # the stable revision-qualified asset path without query credentials.
            for index in page_indices:
                downloader.blocks[(source, index)] = block
        require(len(np.unique(indices)) == counts[split], 'Repeated selected row')
        selected[split] = indices
        tasks.extend((split, source, int(index)) for index in indices)
    completed = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_row, downloader, source, index): (split, index) for split, source, index in tasks}
        for number, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            try:
                completed[key] = future.result()
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
            if number % 50 == 0 or number == len(tasks):
                print(json.dumps({'prepared': number, 'total': len(tasks), 'downloaded_bytes': downloader.bytes}), flush=True)
    arrays = {}
    records = {}
    actual_groups = {}
    for split, indices in selected.items():
        ordered = [completed[(split, int(index))] for index in indices]
        records[split] = [metadata for metadata, _ in ordered]
        arrays[f'{split}_images'] = np.stack([image for _, image in ordered])
        steering = np.array([metadata['steer'] for metadata, _ in ordered], dtype=np.float32)
        arrays[f'{split}_steer'] = steering
        arrays[f'{split}_labels'] = np.where(steering < -.2, 0, np.where(steering > .2, 2, 1)).astype(np.int64)
        arrays[f'{split}_groups'] = np.array([metadata['experiment_id'] for metadata, _ in ordered])
        arrays[f'{split}_source_indices'] = indices
        arrays[f'{split}_timestamps'] = np.array([metadata['timestamp'] for metadata, _ in ordered], dtype=np.float64)
        actual_groups[split] = set(arrays[f'{split}_groups'].tolist())
        require(actual_groups[split] <= set(source_groups[SOURCE_SPLITS[split]]), 'Sample group absent from source statistics')
    for first, second in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        require(not actual_groups[first] & actual_groups[second], 'Selected splits share experiment IDs')
        require(not {r['jpeg_sha256'] for r in records[first]} & {r['jpeg_sha256'] for r in records[second]}, 'Exact JPEG duplicates across selected splits')
    serialized = io.BytesIO()
    np.savez_compressed(serialized, **arrays)
    artifact = serialized.getvalue()
    manifest = {
        'repository': REPO, 'revision': REVISION, 'declared_license': repo_info.get('cardData', {}).get('license'),
        'source_card_url': f'https://huggingface.co/datasets/{REPO}/blob/{REVISION}/README.md',
        'purpose': 'Small offline image-to-steering imitation demo; not closed-loop driving evaluation',
        'image_source': 'Hugging Face Dataset Viewer JPEG derivative, not original Parquet embedded image bytes',
        'input_fields': ['image_path pixels'], 'target_field': 'steer',
        'excluded_input_fields': ['maneuver', 'mask_path', 'steer', 'throttle', 'brake', 'heading', 'speed', 'timestamp', 'experiment_id'],
        'preprocessing': 'Whole-image PIL convert L then resize 28x28 BILINEAR; no fitted preprocessing',
        'classes': ['left', 'straight', 'right'], 'thresholds': [-0.2, 0.2],
        'class_rule': 'left: steer < -0.2; right: steer > 0.2; straight: inclusive [-0.2,0.2]',
        'sampling': 'train: np.linspace(0,20696,300,dtype=int64)[:100]; validation: offsets 0,2000 plus np.linspace(0,99,25,dtype=int64); test: offsets 0,4000 plus same25 offsets; no target-based selection',
        'scope_note': 'Reduced from planned300/150/150 to100/50/50 after HF viewer HTTP429; training spans4 sessions, validation/test use2 fixed100-row windows each; temporally correlated within windows',
        'requested_counts': counts, 'source_counts': SOURCE_COUNTS, 'source_group_counts': source_groups,
        'selected_group_counts': {split: dict(Counter(r['experiment_id'] for r in rows)) for split, rows in records.items()},
        'selected_class_counts': {split: np.bincount(arrays[f'{split}_labels'], minlength=3).tolist() for split in counts},
        'exclusion_assertions': {'all_source_experiment_ids_disjoint': True, 'selected_experiment_ids_disjoint': True,
                                 'selected_cross_split_exact_jpeg_hashes_disjoint': True, 'all_viewer_image_paths_match_pinned_revision': True},
        'artifact': str(OUTPUT.relative_to(ROOT)), 'artifact_sha256': digest(artifact),
        'arrays_sha256': {name: digest(array.tobytes()) for name, array in arrays.items()},
        'preparation_script_sha256': digest(Path(__file__).read_bytes()),
        'source_parquet_files': {f['rfilename']: {'bytes': f['size'], 'sha256': f.get('lfs', {}).get('sha256')}
                                 for f in repo_info['siblings'] if f['rfilename'].endswith('.parquet')},
        'samples': records,
    }
    with np.load(io.BytesIO(artifact), allow_pickle=False) as saved:
        for name, array in arrays.items():
            require(np.array_equal(saved[name], array), f'NPZ roundtrip mismatch: {name}')
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text())
        require(existing['artifact_sha256'] == manifest['artifact_sha256'] and existing['samples'] == records, 'Regeneration differs from tracked provenance')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('xb') as output:
        output.write(artifact)
    if not MANIFEST.exists():
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST.open('x') as output:
            output.write(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'dataset': str(OUTPUT.relative_to(ROOT)), 'counts': counts,
                      'class_counts': manifest['selected_class_counts'], 'downloaded_bytes': downloader.bytes}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=int, default=100)
    parser.add_argument('--validation', type=int, default=50)
    parser.add_argument('--test', type=int, default=50)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    require(1 <= args.workers <= 12, 'workers must be between 1 and 12')
    prepare({'train': args.train, 'validation': args.validation, 'test': args.test}, args.workers)


if __name__ == '__main__':
    main()
