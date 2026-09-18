"""Build the bilingual research note from primary-source synthesis and measured artifacts.

Requires Markdown 3.8.2 and matplotlib; never reruns or fits a controller/model.
"""
from pathlib import Path
import gzip
import hashlib
import html
import json
import re

import markdown
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'novigrad-robotics-v1'
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results/robot-token-bridge'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_results():
    path = RESULTS / 'results.json'
    raw = path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix('.json.gz').read_bytes())
    lock = json.loads((RESULTS / 'protocol-lock.json').read_text())
    run = json.loads((RESULTS / 'run-manifest.json').read_text())
    assert sha(raw) == run['results_sha256']
    assert sha((RESULTS / 'protocol-lock.json').read_bytes()) == run['protocol_lock_sha256']
    for name, digest in lock['source_sha256'].items():
        assert sha((ROOT / name).read_bytes()) == digest, name
    verify = json.loads((RESULTS / 'verification.json').read_text())
    assert verify['exact_full_trace_replay'] and verify['results_sha256'] == sha(raw)
    return json.loads(raw), sha(raw)


def figure(results):
    names = {'continuous_true_disturbance': 'Continuous feedback',
             'bins16_true_disturbance': '16-bin feedback',
             'bins64_true_disturbance': '64-bin feedback',
             'bins256_true_disturbance': '256-bin feedback',
             'bins64_actioncopy_disturbance': 'Command-only state',
             'bins64_wrongframe_disturbance': 'Wrong frame (rejected)',
             'bins64_oneshot_disturbance': 'Single expiring pulse',
             'bins64_slow2hz_disturbance': '64-bin / 2 Hz updates',
             'bins64_true_no_disturbance': '64-bin / no disturbance',
             'bins64_actuator_no_effect_disturbance': '64-bin / disabled actuator'}
    labels = [names[k] for k in results]
    success = [v['metrics']['success_rate'] * 100 for v in results.values()]
    error = [v['metrics']['mean_final_error_rad'] * 180 / 3.141592653589793 for v in results.values()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, layout='constrained')
    axes[0].barh(labels, success, color='#326d9d')
    axes[1].barh(labels, error, color='#ad663c')
    axes[0].invert_yaxis()
    axes[0].set(xlabel='Final heading within 0.15 rad (%)', xlim=(0, 100))
    axes[1].set(xlabel='Mean final absolute heading error (degrees)')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Typed host-codec diagnostic — no language model or learned circuit', fontsize=12)
    path = OUT / 'robotics-control.svg'
    fig.savefig(path, metadata={'Date': None})
    fig.savefig(OUT / 'robotics-control.png', dpi=160)
    plt.close(fig)
    svg = '\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n'
    path.write_text(svg)
    return svg[svg.index('<svg'):]


CSS = '''
*{box-sizing:border-box}body{margin:0;background:#f4f5f3;color:#17212b;font:16px/1.7 system-ui,sans-serif}
main{max-width:1120px;margin:auto;padding:36px 28px}header{border-bottom:3px solid #326d9d;padding-bottom:24px}
h1{font-size:clamp(28px,5vw,48px);line-height:1.2}h2{margin-top:42px;border-bottom:1px solid #ccd3d7;padding-bottom:8px}
h3{margin-top:28px}a{color:#1c587f;overflow-wrap:anywhere}.eyebrow{letter-spacing:.12em;font-size:12px;text-transform:uppercase}
.boundary{background:#e6eef4;border-left:5px solid #326d9d;padding:18px 22px;margin:24px 0}
.table-scroll{max-width:100%;overflow:auto}table{border-collapse:collapse;min-width:680px;width:100%;font-size:14px}
td,th{padding:10px;border-bottom:1px solid #cdd6da;text-align:left}th{background:#e6ebee}
pre{overflow:auto;background:#e8edef;padding:18px;font-size:13px}code{overflow-wrap:anywhere}svg{width:100%;height:auto}
details{border:1px solid #ccd3d7;margin:18px 0;padding:12px}summary{cursor:pointer;font-weight:600}
footer{font-size:13px;color:#4d5c67;border-top:1px solid #ccd3d7;margin-top:36px;padding-top:14px}
@media(max-width:600px){main{padding:22px 16px}body{font-size:15px}.boundary{padding:14px}table{font-size:12px}}
@media print{body{background:white}main{max-width:none;padding:0}a{color:inherit}pre{white-space:pre-wrap}h2,h3{break-after:avoid}svg{break-inside:avoid}}
'''


def main():
    data, result_hash = load_results()
    chart = figure(data['results'])
    smoke_path = ROOT / 'results/robot-simulator-smoke/results.json'
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    manifest = {'scope': 'Research synthesis plus interface and simulator feasibility; no new LM training or biological validation',
                'numeric_results_sha256': result_hash, 'sources': {}, 'outputs': {}}
    manifest['sources'][str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__).read_bytes())
    for name in ('protocol-lock.json', 'run-manifest.json', 'verification.json'):
        p = RESULTS / name
        manifest['sources'][str(p.relative_to(ROOT))] = sha(p.read_bytes())
    for ko in (False, True):
        suffix = '.ko' if ko else ''
        source = ROOT / f'examples/bio_bridge/ROBOTICS_BRIDGE_RESEARCH{suffix}.md'
        document = markdown.markdown(source.read_text(), extensions=['tables', 'fenced_code'])
        document = re.sub(r'href="(?!https?://|#)([^"]+)"', r'href="../../examples/bio_bridge/\1"', document)
        document = document.replace('<table>', '<div class="table-scroll"><table>').replace('</table>', '</table></div>')
        document = document.replace('<h1>', '<h2>').replace('</h1>', '</h2>')
        title = '생물학에서 양방향 로보틱스 브리지로' if ko else 'From biological mechanisms to a bidirectional robotics bridge'
        boundary = ('문헌의 생물학적 관찰, 제안하는 언어모델 연결 구조, 실제 실행한 공학 검증을 구분합니다. 아래 방향 제어 실험에는 임베딩·언어모델·실제 신경 회로가 없습니다.' if ko else
                    'Biological observations, proposed language-model interfaces, and executed engineering checks are distinct. The heading-control diagnostic below contains no embedding model, language model, or reconstructed neural circuit.')
        columns = ['조건', '성공 / 사례', '평균 최종 오차 (rad)', '거부 명령 수'] if ko else ['Condition', 'Success / cases', 'Mean final error (rad)', 'Rejected commands']
        rows = []
        for name, values in data['results'].items():
            m = values['metrics']
            rows.append(f'<tr><td>{html.escape(name)}</td><td>{m["successes"]} / {m["cases"]}</td><td>{m["mean_final_error_rad"]:.4f}</td><td>{m["command_errors"]}</td></tr>')
        table = '<div class="table-scroll"><table><thead><tr>' + ''.join(f'<th>{x}</th>' for x in columns) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
        notes = ('목표는 명시적으로 주어진 방향이며 1.5초에 바뀝니다. 8개 난수 seed × 16개 초기 조건은 짝지어진 합성 사례로, 독립적인 학습 반복이나 동물 표본이 아닙니다. 성공은 3초 시점 오차가 0.15 rad 미만인 경우입니다. 자기 명령만 적분한 상태는 외부 회전을 놓치도록 구성했습니다. 잘못된 좌표계 조건은 모든 명령을 거부하는 규약 검사이며, 일회 명령 조건은 만료 뒤 정지하는 펄스입니다. 지속 기억이나 생물학적 회상 비교가 아닙니다. 초기 검사에서 발견한 명령 만료 경계 오류와 수정 전 자료를 함께 보존했습니다.' if ko else
                 'Goals are explicit requested headings and switch at 1.5 s. Eight RNG seeds × 16 initial states are paired synthetic cases, not independent training runs or animals. Success means final error below 0.15 rad at 3 s. Command-only state integration deliberately omits external rotation. Wrong-frame commands are rejected by contract; one-shot is an expiring motor pulse, not a persistent-memory or biological-recall test. The first check exposed an expiry-boundary bug; its pre-correction artifacts are retained.')
        simulation = ('별도 물리 시뮬레이터 검증' if ko else 'Separate physics simulator readiness')
        if smoke is not None:
            smoke_manifest = json.loads(smoke_path.with_name('manifest.json').read_text())
            assert sha(smoke_path.read_bytes()) == smoke_manifest['files']['results.json']
            assert sha((ROOT / smoke_manifest['script']['path']).read_bytes()) == smoke_manifest['script']['sha256']
            smoke_section = ('실제 실행 기록을 아래에 원문으로 포함합니다. 짧은 물리 step 검증이며 언어모델 브리지, 학습, 탐색 성공을 검증하지 않습니다.' if ko else
                             'The actual execution record is reproduced below. This short physics-step check does not establish an LM bridge, learning, or navigation success.')
            smoke_section += f' FlyGym {smoke["dependencies"]["flygym"]} / MuJoCo {smoke["dependencies"]["mujoco"]}; {smoke["steps"]:,} steps; {smoke["simulated_seconds"]:.1f} s simulation / {smoke["step_wall_seconds"]:.3f} s wall time ({smoke["real_time_factor"]:.3f}×).'
            smoke_section += '<details><summary>' + ('실행 기록' if ko else 'Execution record') + '</summary><pre>' + html.escape(json.dumps(smoke, ensure_ascii=False, indent=2)) + '</pre></details>'
            manifest['sources'][str(smoke_path.relative_to(ROOT))] = sha(smoke_path.read_bytes())
        else:
            smoke_section = '아직 완료된 실행 기록이 없습니다.' if ko else 'No completed simulator execution record is available.'
        switch = 'robotics.html' if ko else 'robotics.ko.html'
        output = OUT / f'robotics{suffix}.html'
        output.write_text(f'''<!doctype html><html lang="{'ko' if ko else 'en'}"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{CSS}</style><main>
<header><p class="eyebrow">Novigrad · Research and reproducible feasibility · 2026-09-18</p><h1>{title}</h1><a href="{switch}">{'English' if ko else '한국어'}</a></header>
<div class="boundary">{boundary}</div>
<section><h2>{'실행한 토큰·제어 규약 검사' if ko else 'Executed token and control-contract diagnostic'}</h2><p>{notes}</p>{table}{chart}<p><a href="../../results/robot-token-bridge/README{suffix}.md">{'방법·한계·재현' if ko else 'Methods, limitations and replay'}</a> · <a href="../../results/robot-token-bridge/results.json.gz">{'전체 압축 경로' if ko else 'All compressed traces'}</a></p></section>
<section><h2>{simulation}</h2><div>{smoke_section}</div><a href="../../results/robot-simulator-smoke/README{suffix}.md">{'출처와 의존성 기록' if ko else 'Provenance and dependencies'}</a></section>
<section>{document}</section><footer>jioh jung · Novigrad · {'동료 심사 전 연구 설계·공학 검증 문서' if ko else 'Pre-peer-review research design and engineering checks'}<br>Result SHA-256: <code>{result_hash}</code></footer></main></html>''')
        manifest['sources'][str(source.relative_to(ROOT))] = sha(source.read_bytes())
        manifest['outputs'][output.name] = sha(output.read_bytes())
    manifest['matplotlib'] = matplotlib.__version__
    manifest['markdown'] = markdown.__version__
    for name in ('robotics-control.svg', 'robotics-control.png'):
        manifest['outputs'][name] = sha((OUT / name).read_bytes())
    (OUT / 'robotics-report-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
