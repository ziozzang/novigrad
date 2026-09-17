#!/usr/bin/env python3
"""Bounded original ORN data import; never infer paired identities from summaries."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/orn-data-pilot'
BASE = 'https://datadryad.org'
VERSION = 452144
DOI = '10.5061/dryad.9p8cz8x0j'
MAX_BYTES = 2 * 1024**2
FILES = {
    'README.md': (4856149, 27257, '637e7806f1a46d1442c35bc8d14e31a2d23f5072b90563cf9b26d0d1495b8335'),
    'Source_data_table_for_Figure_2_non_formatted.xlsx': (4856086, 35321, 'eaff9022e0b29f09b506507fd188b6258ee7fbae2b712b707fbaedc80272a25f'),
}
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def request_bytes(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Novigrad-ORN-research-import/1.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        if int(response.headers.get('Content-Length', 0)) > MAX_BYTES:
            raise ValueError('remote file exceeds per-file limit')
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError('response exceeds per-file limit')
    return data


def validate_source(data, size, checksum):
    if len(data) != size or sha(data) != checksum:
        raise ValueError('source size/checksum differs from pinned original')


def read_xlsx(data):
    """Read cell values with original addresses; no formulas or macros executed."""
    result = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sum(item.file_size for item in archive.infolist()) > 20 * 1024**2:
            raise ValueError('expanded workbook exceeds limit')
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            root = ET.fromstring(archive.read('xl/sharedStrings.xml'))
            strings = [''.join(node.itertext()) for node in root.findall('s:si', NS)]
        rels = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        targets = {r.attrib['Id']: r.attrib['Target'] for r in rels}
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        for sheet in workbook.findall('s:sheets/s:sheet', NS):
            rid = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            target = targets[rid]
            if '..' in PurePosixPath(target).parts:
                raise ValueError('invalid workbook relationship')
            target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            root = ET.fromstring(archive.read(target))
            cells = []
            for cell in root.findall('.//s:sheetData/s:row/s:c', NS):
                value = cell.find('s:v', NS)
                kind = cell.attrib.get('t')
                if cell.find('s:f', NS) is not None:
                    cells.append({'address': cell.attrib['r'], 'formula': True, 'value': None})
                    continue
                if kind == 'inlineStr':
                    inline = cell.find('s:is', NS)
                    content = ''.join(inline.itertext()) if inline is not None else ''
                elif value is None:
                    continue
                elif kind == 's':
                    content = strings[int(value.text)]
                else:
                    content = value.text
                cells.append({'address': cell.attrib['r'], 'value': content})
            result[sheet.attrib['name']] = cells
    return result


def require_paired_rows(rows):
    """Fail closed: row order and similarly named columns are not pairing keys."""
    if not rows:
        raise ValueError('no explicitly paired measurements')
    keys = set()
    for row in rows:
        required = ('experiment_id', 'fly_id', 'pair_id', 'calcium_source_cell', 'spike_source_cell')
        if any(not isinstance(row.get(k), str) or not row[k] for k in required):
            raise ValueError('missing explicit experiment/fly/pair identity or source cells')
        key = (row['experiment_id'], row['fly_id'], row['pair_id'])
        if key in keys:
            raise ValueError('duplicate paired observation')
        keys.add(key)
    return rows


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / 'source'
    source.mkdir(exist_ok=True)
    metadata_url = BASE + '/api/v2/datasets/doi%3A10.5061%2Fdryad.9p8cz8x0j'
    files_url = BASE + f'/api/v2/versions/{VERSION}/files'
    metadata_data = request_bytes(metadata_url)
    listing_data = request_bytes(files_url)
    metadata = json.loads(metadata_data)
    listing = json.loads(listing_data)['_embedded']['stash:files']
    if metadata['license'] != 'https://spdx.org/licenses/CC0-1.0.html':
        raise ValueError('dataset license changed')
    indexed = {row['path']: row for row in listing}
    for name, (_, size, digest) in FILES.items():
        row = indexed[name]
        if row['size'] != size or row['digestType'] != 'sha-256' or row['digest'] != digest:
            raise ValueError('pinned version file manifest differs')
    (source / 'dataset-api.json').write_bytes(metadata_data)
    (source / 'version-files-api.json').write_bytes(listing_data)
    records = []
    for name, (file_id, size, digest) in FILES.items():
        destination = source / name
        urls = [BASE + f'/downloads/file_stream/{file_id}', BASE + f'/api/v2/files/{file_id}/download']
        record = {'name': name, 'file_id': file_id, 'expected_bytes': size,
                  'expected_sha256': digest, 'source_urls': urls, 'attempts': []}
        if destination.exists():
            validate_source(destination.read_bytes(), size, digest)
        else:
            for url in urls:
                try:
                    data = request_bytes(url)
                    validate_source(data, size, digest)
                    destination.write_bytes(data)
                    record['attempts'].append({'url': url, 'status': 'downloaded'})
                    break
                except (urllib.error.URLError, TimeoutError) as error:
                    record['attempts'].append({'url': url, 'status': 'unavailable',
                                               'http_status': getattr(error, 'code', None), 'error': str(error)})
        record['downloaded'] = destination.exists()
        if destination.exists():
            record['sha256'] = sha(destination.read_bytes())
        records.append(record)
    complete = all(row['downloaded'] for row in records)
    if complete:
        workbook = source / 'Source_data_table_for_Figure_2_non_formatted.xlsx'
        write_json(args.out / 'workbook-cells.json', read_xlsx(workbook.read_bytes()))
    report = {
        'doi': DOI, 'dryad_version_id': VERSION, 'license': metadata['license'],
        'attribution': 'Xiao Y, Wu ST, Xuan Y, Rifkin SA, Su CY. Dataset published 2026-07-10. DOI: ' + DOI,
        'dataset_title': metadata['title'], 'source_metadata_url': metadata_url,
        'source_version_files_url': files_url,
        'metadata_sha256': sha(metadata_data), 'version_files_sha256': sha(listing_data),
        'generator_sha256': sha(Path(__file__).read_bytes()), 'files': records,
        'status': 'schema_review_required' if complete else 'blocked_original_file_access',
        'fit_executed': False, 'prediction_rows': 0,
        'reason': ('Workbook cells exported; verify explicit same-record calcium/spike pairing and biological fly IDs before fitting.' if complete else
                   'Metadata verified, but original files unavailable via public and API download paths. Workbook schema and paired fly identities cannot be validated from documentation alone.'),
        'predeclared_design': {'split': 'leave-one-biological-fly-out; all repeats/ROIs together',
            'models': ['training_mean', 'calcium_only_OLS', 'calcium_only_ridge_alpha_1', 'training_pair_shuffle_ridge_alpha_1'],
            'normalization': 'training-fold mean/std only', 'shuffle_seed': 20260917,
            'selection': 'none; no held-out tuning', 'optional_concentration': 'only if explicit paired-row metadata validates it',
            'metrics': ['per-fly MAE', 'macro-fly MAE', 'all grouped raw predictions']},
        'boundary': 'No fabricated joins, no behavior or goal inference, no model fit without verified individual pairing.'}
    write_json(args.out / 'import-report.json', report)
    print(json.dumps({'status': report['status'], 'fit_executed': False, 'files_downloaded': sum(r['downloaded'] for r in records)}))


if __name__ == '__main__':
    main()
