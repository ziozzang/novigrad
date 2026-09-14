#!/usr/bin/env python3
"""Build an auditable local report from every final experiment row."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'results/v1-confirmation'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    with (DIRECTORY / 'metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 72
    summary = []
    for task in ['binary', 'fourway', 'xor']:
        groups = {mode: [r for r in rows if r['task'] == task and r['mode'] == mode] for mode in ['reward', 'frozen', 'shuffled']}
        for group in groups.values():
            assert sorted(int(r['seed']) for r in group) == list(range(101, 109))
        mean = lambda mode, key: statistics.mean(float(r[key]) for r in groups[mode])
        reward = mean('reward', 'after_expected')
        reversal = mean('reward', 'reversal_expected')
        assert reward >= .85 and reversal >= .85
        for mode in ['frozen', 'shuffled']:
            assert reward - mean(mode, 'after_expected') >= .20
            assert reversal - mean(mode, 'reversal_expected') >= .20
        assert all(float(r['after_expected']) >= .75 and float(r['reversal_expected']) >= .75 and int(r['changed_weights']) > 0 for r in groups['reward'])
        assert all(int(r['changed_weights']) == 0 and r['before_expected'] == r['after_expected'] for r in groups['frozen'])
        summary.append(dict(task=task, before_greedy=mean('reward','before_greedy'), after_greedy=mean('reward','after_greedy'), expected=reward, frozen=mean('frozen','after_expected'), random_reward=mean('shuffled','after_expected'), before_reversal=mean('reward','before_reversal_expected'), reversal=reversal, reversal_greedy=mean('reward','reversal_greedy'), minimum_seed_expected=min(float(r['after_expected']) for r in groups['reward']), changed_weights_min=min(int(r['changed_weights']) for r in groups['reward']), changed_weights_max=max(int(r['changed_weights']) for r in groups['reward'])))
    portable = json.loads((ROOT / 'results/safetensors-verification.json').read_text())
    assert len(portable) == 4 and all(r['passed'] for r in portable)
    manifest = {'platform': platform.platform(), 'machine': platform.machine(), 'rustc': subprocess.check_output(['rustc','--version'], text=True).strip(), 'configuration': {key: rows[0][key] for key in ['episodes','reversal_episodes','eval_trials','learning_rate','logit_gain']}, 'seeds': list(range(101,109)), 'results': summary, 'artifacts': {str(path.relative_to(ROOT)): digest(path) for path in [DIRECTORY/'metrics.csv', DIRECTORY/'summary.txt', *sorted(DIRECTORY.glob('*.safetensors')), ROOT/'data/pn_kc.tsv', ROOT/'data/kc_mbon.tsv', ROOT/'data/training_circuit.json', ROOT/'Cargo.lock']}, 'final_source_sha256': {str(path.relative_to(ROOT)): digest(path) for path in [ROOT/'src/plastic.rs', ROOT/'src/bin/train_connectome.rs', ROOT/'src/bin/nobi_engine.rs']}, 'safetensors_verification': portable, 'criteria_passed': True}
    (DIRECTORY/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    lines = ['# nobi v1 학습 검증', '', '실제 FlyWire ALPN→KC→MBON 연결을 사용한 rate 모델에서 보상 학습과 규칙 전환 후 재학습을 검증했다. 최종 확인은 튜닝에 사용하지 않은 seed 101–108, 세 과제, 세 보상 조건의 72회 실행이다.', '', '각 실행: 학습 2,000회 + 재학습 2,000회. 각 평가: 독립 잡음 입력 1,024개. 학습률 0.02, gain 12, 출력별 가중치 합 제한. 값은 8개 seed의 평균이며 아래 정답 확률은 모델이 정답 행동에 배정한 확률이다.', '', '| 과제 | 학습 전 greedy | 학습 후 greedy | 정답 확률 | 보상 0 | 무작위 보상 | 규칙 전환 직후 | 재학습 후 |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    names={'binary':'두 입력 구분', 'fourway':'4개 입력 분류', 'xor':'XOR'}
    for r in summary:
        lines.append('| '+names[r['task']]+' | '+' | '.join(f"{100*r[key]:.2f}%" for key in ['before_greedy','after_greedy','expected','frozen','random_reward','before_reversal','reversal'])+' |')
    lines += ['', '모든 과제에서 사전 설정한 기준을 통과했다: 정상 보상과 재학습의 평균 정답 확률 ≥85%, 각 대조군보다 ≥20%p 높음, 각 seed ≥75%, 학습 가중치 변화 있음. 보상 0 조건은 가중치가 한 개도 바뀌지 않았다. 평가에서도 가중치가 보존된다.', '', '기존 KC→MBON 연결의 가중치를 실제로 갱신했다. 출력별 가중치 합 정규화는 비활성 시냅스에도 영향을 주므로 changed_weights를 보상 eligibility가 생긴 연결 수로 해석하지 않는다.', '', '## 저장·재개', '', '세 실제 회로 모델과 범용 예제 모델을 공식 Python Safetensors로 읽었다. NumPy 독립 forward와 Rust CLI 추론이 일치했다. 실제 회로 실행기는 저장 전후 예측 및 동일 후속 입력·행동·보상 흐름에서 이어서 학습한 가중치의 bit 단위 일치도 확인했다.', '', f"독립 추론 비교 최대 절대 오차: {max(r['max_absolute_error'] for r in portable):.3g} (허용 2e-6 + 상대 오차).", '', '## M2 Ultra 처리량', '', '아래는 고정 입력 forward + 보상 0의 hot-loop 측정이다. 파일 로딩은 분리했으며 실제 보상 갱신 처리량과 같지 않다.', '', '```text', (ROOT/'results/benchmark.txt').read_text().strip(), '```', '', '학습 실험의 실행 시간은 metrics.csv에 있다. 평가·체크포인트 검증·일부 재현 실행을 포함하므로 순수 학습 처리량으로 사용하지 않는다.', '', '## 재현과 한계', '', '- [전체 측정 CSV](v1-confirmation/metrics.csv), [판정](v1-confirmation/summary.txt), [입력·모델·코드 해시](v1-confirmation/manifest.json).', '- 학습 모델: [binary](v1-confirmation/binary.safetensors), [fourway](v1-confirmation/fourway.safetensors), [xor](v1-confirmation/xor.safetensors).', '- 최종 코드로 seed 101의 9개 조건을 재실행했으며 시간 외 모든 지표가 기존 실행과 같았다. 이 한 seed에서는 무작위 보상도 이진 재학습 점수가 높아 단일 seed 판정은 실패했다. 최종 성공 판정은 사전에 정한 8개 seed 평균으로 수행한다 ([재현 기록](reproduction.json)).', '- 초기 비정규화 모델의 재학습 실패도 v1-validation/ 및 v1-final/에 보존했다. 성능 기준은 낮추지 않았으며 엔진 정규화를 수정했다.', '- 실제 연결 제약을 쓰지만 입력 패턴과 보상은 합성이다. 출력 행동 매핑도 공학적으로 지정했다. 전체 초파리 뇌나 실제 도파민 작용을 학습한 것이 아니다.', '- 고정 ALPN→KC와 가소성 KC→MBON의 두 레이어 모델이다. 임의 깊이·재귀 회로 학습이나 생물학적 시간 상수는 이 결과로 검증되지 않는다.', '- XOR은 네 조합 모두를 학습하고 새 잡음으로 평가했다. 보지 않은 조합 추론이나 실제 감각 데이터 일반화의 증거는 아니다.', '- 연결망의 우수성을 임의 연결과 비교한 결과는 아니다. 일반적인 용도를 지원하는 첫 CPU 엔진으로 범위를 둔다.', '']
    (ROOT/'results/V1_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
