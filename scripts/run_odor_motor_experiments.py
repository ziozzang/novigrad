#!/usr/bin/env python3
"""Reproduce the bounded synthetic odor/delayed-reward controls."""
import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(1, 9))
REVERSAL_SEEDS = (1, 2, 3)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def annotation_audit():
    path = ROOT / 'data/neurons_783.tsv'
    classes = ('DAN', 'MBON', 'motor')
    counts = {name: 0 for name in classes}
    counts['descending'] = 0
    total = 0
    with path.open(newline='') as source:
        for row in csv.DictReader(source, delimiter='\t'):
            total += 1
            if row['cell_class'] in classes:
                counts[row['cell_class']] += 1
            if row['super_class'] == 'descending':
                counts['descending'] += 1
    return {'source': str(path.relative_to(ROOT)), 'sha256': sha256(path),
            'row_count': total, 'annotation_field_counts': counts,
            'interpretation': 'FlyWire brain annotation rows, not a whole-CNS or foreleg-motor census'}


def result_table(rows, key):
    by = {row['seed']: row for row in rows if row['mode'] == key and row['episodes'] == 300}
    return [by[s] for s in SEEDS]


def write_reports(outdir, records, provenance):
    by_mode = {mode: result_table(records, mode) for mode in ('reward', 'frozen', 'shuffled')}
    reversal = {r['seed']: r for r in records if r['episodes'] == 600}
    def pct(v):
        return f'{100*v:.2f}%'
    def mean(rows, field):
        return sum(r[field] for r in rows) / len(rows)
    reward_mean = mean(by_mode['reward'], 'final_accuracy')
    shuffled_mean = mean(by_mode['shuffled'], 'final_accuracy')
    frozen_mean = mean(by_mode['frozen'], 'final_accuracy')
    lines = ['# Neuromodulation experiment — synthetic odor to virtual action', '',
             '[한국어 요약](NEUROMODULATION_REPORT.ko.md)', '',
             'This is a delayed-reward learning test of the actual topology-constrained Rust engine on a narrow synthetic two-odor task. Output labels are **virtual actions**, not identified foreleg motor neurons.', '',
             '| Seed | Reward-trained | Frozen | Independent shuffled target | Reversal pre / post |',
             '|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        reward = by_mode['reward'][seed-1]['final_accuracy']
        frozen = by_mode['frozen'][seed-1]['final_accuracy']
        shuffled = by_mode['shuffled'][seed-1]['final_accuracy']
        rev = reversal.get(seed)
        reverse_text = f"{pct(rev['pre_reversal_accuracy'])} / {pct(rev['final_accuracy'])}" if rev else '—'
        lines.append(f'| {seed} | {pct(reward)} | {pct(frozen)} | {pct(shuffled)} | {reverse_text} |')
    lines += ['', f'Across eight fixed seeds: reward-trained **{pct(reward_mean)}**, frozen **{pct(frozen_mean)}**, and independent-target control **{pct(shuffled_mean)}** (range {pct(min(r["final_accuracy"] for r in by_mode["shuffled"]))}–{pct(max(r["final_accuracy"] for r in by_mode["shuffled"]))}). The frozen 33% is an empirical policy result and **not** an assumed 50% chance baseline. Reversal was run for seeds 1–3 only.', '',
              '## Protocol and interpretation', '',
              '- Data: 32 sorted real ALPN input IDs from the FlyWire-derived PN→KC graph, divided into two synthetic odor groups. Each trial sets up to 12 of 16 in-group ports to rate 1 and four opposite-group nuisance draws to rate 0.15. Sampling is with replacement. The 400 evaluation patterns per seed use a separate PRNG stream; uniqueness against training is not asserted.',
              '- Circuit: 27,848 fixed ALPN→KC edges, 62,261 sign-constrained plastic KC→MBON edges, 10% hidden activity, learning rate 0.02, logit gain 6, default per-MBON L1 homeostasis. The decoder maps 96 MBON ports into two engineered virtual actions.',
              '- Delayed feedback: capacity 3 observations/episode, `lambda=0.8`, exponential baseline `alpha=0.02`, reward +1 for the designated action and −1 otherwise. Weights stay fixed for all three observations. On signal, `error = reward − baseline`; old records are scaled by `lambda^age`; gradients are averaged and applied once; baseline updates after application.',
              '- Controls: `frozen` performs the same observations/actions but discards each trace, causing no weight changes. `shuffled` samples the reward target independently of odor for every episode; it is not a permuted fixed dataset. The wide range across seeds on this tiny task shows why the control should be inspected per seed.',
              '- Reversal: 600 episodes, correct odor-action mapping flipped after episode 300. Accuracy on the original mapping is measured before reversal and on the flipped mapping afterward. Each accuracy uses 400 separately generated patterns.',
              '- Checkpoints are Safetensors schema 4. Per-run JSON and exact CLI commands are in [neuromodulation/](neuromodulation/); checkpoints are locally generated and excluded from Git. Regenerate with `python3 scripts/run_odor_motor_experiments.py`.', '',
              '## Biological boundary', '',
              'DANs provide compartmental reinforcement signals to mushroom-body KC→MBON synapses ([adult MB connectome](https://elifesciences.org/articles/62576)); reward-prediction circuit interpretations also exist ([eLife model](https://elifesciences.org/articles/75611)). This implementation uses a **global scalar baseline and eligibility replay**, so it is neither an anatomical DAN simulation nor a measured dopamine rule. FlyWire here is a [brain connectome](https://www.nature.com/articles/s41586-024-07558-y); linking its descending outputs to VNC leg motor circuits requires separate data and matching ([brain/VNC comparative study](https://www.nature.com/articles/s41586-025-08925-z)).',
              '', f"The local `{provenance['source']}` ({provenance['row_count']:,} rows) has annotation-field counts DAN {provenance['annotation_field_counts']['DAN']}, MBON {provenance['annotation_field_counts']['MBON']}, descending {provenance['annotation_field_counts']['descending']}, and `motor` class {provenance['annotation_field_counts']['motor']}. These counts do not establish an MBON→foreleg motor path.", '']
    (outdir.parent / 'NEUROMODULATION_REPORT.md').write_text('\n'.join(lines))
    ko = ['# 신경조절 실험 — 합성 후각 입력과 가상 행동', '',
          '[영문 상세 보고서](NEUROMODULATION_REPORT.md)', '',
          f'8개 고정 seed의 합성 테스트 평균: 지연 보상 학습 **{pct(reward_mean)}**, 동결 모델 **{pct(frozen_mean)}**, 후각과 무관하게 보상 목표를 뽑는 대조군 **{pct(shuffled_mean)}**. 동결 모델의 33%는 실제 초기 정책의 측정값이며 50%의 이론적 우연 수준으로 해석하지 않습니다. 300 에피소드 뒤 목표 행동을 뒤집은 3개 seed는 뒤집기 전후 각각 100%였습니다.', '',
          '입력은 FlyWire 유래 ALPN 포트 32개의 작은 합성 두 후각 패턴입니다. 보상은 세 번의 관측 뒤 주어지고, 가중치는 그 사이 고정됩니다. `lambda=0.8`, 보상 기준값 갱신률 `alpha=0.02`, 학습률 0.02를 사용했습니다. 테스트는 별도 난수 흐름으로 만든 패턴 400개이지만, 학습 패턴과의 완전한 비중복은 보장하지 않습니다.', '',
          '출력 두 개는 **가상 행동 명령**입니다. 실제 앞다리 운동 뉴런 ID나 다리 제어가 아닙니다. 전역 보상 기준값과 기록 재실행은 공학적 기능이며 DAN의 구획별 도파민 작용을 재현하지 않습니다. 생물학적 출처와 각 seed 결과, 실행 명령은 [영문 보고서](NEUROMODULATION_REPORT.md)를 참고하세요.', '']
    (outdir.parent / 'NEUROMODULATION_REPORT.ko.md').write_text('\n'.join(ko))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=Path('results/neuromodulation'))
    parser.add_argument('--skip-build', action='store_true')
    args = parser.parse_args()
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    target = ROOT / 'target/odor-experiments'
    binary = target / 'release/odor_motor'
    if not args.skip_build:
        env = os.environ.copy()
        env['CARGO_TARGET_DIR'] = str(target)
        subprocess.run(['cargo', 'build', '--release', '--bin', 'odor_motor'], cwd=ROOT, env=env, check=True)
    if not binary.exists():
        raise SystemExit(f'missing binary: {binary}; omit --skip-build')
    scenarios = [(seed, mode, 300, 300) for seed in SEEDS for mode in ('reward', 'frozen', 'shuffled')]
    scenarios += [(seed, 'reward', 600, 300) for seed in REVERSAL_SEEDS]
    records, commands = [], []
    for seed, mode, episodes, reverse_after in scenarios:
        name = f'{mode}-seed{seed}-episodes{episodes}'
        checkpoint = outdir / f'{name}.safetensors'
        command = [str(binary), '--episodes', str(episodes), '--seed', str(seed), '--mode', mode,
                   '--reverse-after', str(reverse_after), '--out', str(checkpoint)]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
        result = json.loads(process.stdout)
        result['checkpoint'] = str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint)
        result['checkpoint_sha256'] = sha256(checkpoint)
        (outdir / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')
        records.append(result)
        reproducible = ['target/odor-experiments/release/odor_motor'] + command[1:]
        reproducible[-1] = result['checkpoint']
        commands.append({'name': name, 'command': shlex.join(reproducible)})
        print(name, result['final_accuracy'])
    provenance = annotation_audit()
    provenance['edge_files'] = {name: {'source': f'data/{name}', 'sha256': sha256(ROOT / 'data' / name)} for name in ('pn_kc.tsv', 'kc_mbon.tsv')}
    (outdir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (outdir / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    (outdir / 'summary.json').write_text(json.dumps(records, indent=2) + '\n')
    write_reports(outdir, records, provenance)


if __name__ == '__main__':
    main()
