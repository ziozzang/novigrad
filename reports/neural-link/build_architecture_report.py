#!/usr/bin/env python3
"""Render bilingual supplementary HTML from frozen architecture JSON only."""
import hashlib
import html
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results/architecture-bridge'
DEST = Path(__file__).resolve().parent
LABELS = ('water', 'food', 'warmth', 'rest')


def esc(value):
    return html.escape(str(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{esc(c)}</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def percent(value):
    return str((Decimal(str(value)) * 100).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)) + '%'


def stats(values):
    return f'{percent(mean(values))} [{percent(min(values))}–{percent(max(values))}]'


def load():
    names = ['protocol-lock.json', 'evaluation-lock.json', 'holdout-final.json', 'adversarial-final.json',
             'adapters/report.json', 'mixing-sanity.json', 'verification.json']
    paths = [DATA / name for name in names]
    paths += [ROOT / f'examples/bio_bridge/architecture_{name}.json' for name in ('holdout', 'adversarial')]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        raise SystemExit('Final artifacts not ready: ' + ', '.join(missing))
    records = {name: json.loads((DATA / name).read_text()) for name in names}
    for source, key in [('protocol-lock.json', 'frozen'), ('evaluation-lock.json', 'hashes')]:
        for relative, expected in records[source][key].items():
            if sha(ROOT / relative) != expected:
                raise ValueError(f'locked artifact changed: {relative}')
    verification = records['verification.json']
    if verification['protocol_sha256'] != sha(DATA / 'protocol-lock.json') or verification['evaluation_sha256'] != sha(DATA / 'evaluation-lock.json'):
        raise ValueError('verification lock hashes differ')
    if verification['saved_inference_exact'] != {'holdout': True, 'adversarial': True}:
        raise ValueError('saved inference replay did not pass')
    cases = {name: json.loads((ROOT / f'examples/bio_bridge/architecture_{name}.json').read_text()) for name in ('holdout', 'adversarial')}
    for name, rows in cases.items():
        final = records[f'{name}-final.json']
        if final['case_ids'] != [r['id'] for r in rows] or final['labels'] != [LABELS.index(r['class']) for r in rows]:
            raise ValueError('case order or labels mismatch')
    return records, cases, paths


def architecture_svg(ko):
    labels = {
        'title': '고정 표현과 읽기 전용 비교 경로' if ko else 'Frozen representations and read-only comparison paths',
        'encoder': '고정 EmbeddingGemma' if ko else 'Frozen EmbeddingGemma',
        'ports': 'PCA · signed ports', 'pn': 'PN', 'kc': 'KC · top-k', 'mbon': 'MBON',
        'readout': 'Offline 지도 readout' if ko else 'Offline supervised readout',
        'context': '외부 문맥 규칙' if ko else 'External context rule',
        'baseline': '직접 embedding 기준선 (우회)' if ko else 'Direct embedding baseline (bypass)',
        'adapter': '별도 저랭크/full 선형 adapter' if ko else 'Separate low-rank/full linear adapter',
        'note': '점선: 읽기 전용 특징 · 내부 신경 상태 쓰기 아님' if ko else 'Dashed: read-only features · not neural state write-in',
        'note2': 'Adapter는 회로 우회 · transformer LoRA 아님' if ko else 'Adapter bypasses circuit · not transformer LoRA',
    }
    def box(x,y,w,label,fill='#edf2f2'):
        return f'<rect x="{x}" y="{y}" width="{w}" height="48" rx="7" fill="{fill}" stroke="#577780"/><text x="{x+w/2}" y="{y+29}" text-anchor="middle">{esc(label)}</text>'
    svg = '<svg viewBox="0 0 1080 410" role="img" aria-labelledby="architecture-title" xmlns="http://www.w3.org/2000/svg" style="width:100%;min-width:720px;font:15px system-ui;background:#fff"><title id="architecture-title">'+esc(labels['title'])+'</title><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#456570"/></marker></defs>'
    for args in [(20,40,200,labels['encoder']),(265,40,185,labels['ports']),(490,40,80,labels['pn']),(615,40,125,labels['kc']),(780,40,100,labels['mbon']), (565,170,260,labels['readout']), (855,170,200,labels['context']), (20,265,370,labels['baseline']), (430,265,400,labels['adapter'])]:
        svg += box(*args)
    for path in ['M220 64 H265','M450 64 H490','M570 64 H615','M740 64 H780','M855 194 H825','M120 88 V265','M220 76 H240 V245 H630 V265']:
        svg += f'<path d="{path}" stroke="#456570" fill="none" stroke-width="2" marker-end="url(#arrow)"/>'
    for path in ['M530 88 V130 H600 V170','M677 88 V170','M830 88 V130 H795 V170']:
        svg += f'<path d="{path}" stroke="#456570" fill="none" stroke-width="2" stroke-dasharray="6 5" marker-end="url(#arrow)"/>'
    svg += '<text x="540" y="352" text-anchor="middle">'+esc(labels['note'])+'</text><text x="540" y="380" text-anchor="middle">'+esc(labels['note2'])+'</text></svg>'
    return '<div class="scroll">'+svg+'</div>'


def render(lang, records, cases, paths):
    ko = lang == 'ko'
    def tr(en, kr): return kr if ko else en
    sections = []
    def section(title, content): sections.append(f'<section><h2>{esc(title)}</h2>{content}</section>')
    caveat = tr('No quantum jump is demonstrated by this supplementary study. Context results are offline supervised probes, not native reward learning or neural recurrence. Site pooling uses fixed landmarks, not learned tokens. The mHC check is algebraic only. Adapter training does not fine-tune EmbeddingGemma.',
                '이 보충 연구는 비약적 성능 향상을 입증하지 않는다. 문맥 결과는 offline 지도 probe이며 native 보상 학습이나 신경 재귀가 아니다. 위치별 pooling은 학습 토큰이 아닌 고정 landmark다. mHC 검사는 대수적 확인뿐이며 EmbeddingGemma를 미세조정하지 않았다.')
    intro = f'<p class="notice">{esc(caveat)}</p>'
    rows = []
    for name, data in cases.items():
        result = records[f'{name}-final.json']
        rows.append([name, len(data), len({r['pair_id'] for r in data}), len({r['family'] for r in data}), percent(result['direct_embedding']['accuracy'])])
    section(tr('Frozen data and direct baseline', '고정 자료와 직접 기준선'), table([tr('Set','세트'),tr('Texts','문장'),tr('Bilingual pairs','번역 쌍'),tr('Families','상황군'),tr('Direct embedding accuracy','직접 임베딩 정확도')], rows) + '<p>' + esc(tr('Authored data, not external observations. Translation pairs, context-expanded rows, and seeds are dependent; mean–range across seeds is descriptive, not a confidence interval. All variants were retained without choosing a validation winner.', '직접 작성한 자료이며 외부 관측값이 아니다. 번역 쌍·문맥 확장 행·seed는 독립 표본이 아니다. Seed 평균–범위는 신뢰구간이 아닌 기술 통계다. 검증 승자를 고르지 않고 모든 조건을 유지했다.')) + '</p>')
    stress_models = records['adversarial-final.json']['probes']['cyclic']['models']
    direct = stress_models['interaction_embedding']['evaluation']['accuracy']
    kc = stress_models['interaction_kc']['evaluation']['accuracy']
    matched = stress_models['matched_interaction_kc']['evaluation']['accuracy']
    finding = tr(
        f'On the adversarial set under the cyclic rule, direct embedding interaction reaches {percent(direct)}, compared with KC interaction {percent(kc)} and coefficient-count-matched KC interaction {percent(matched)}. The stronger direct baseline wins this stress comparison. Larger bridge ranks remain subject to a four-logit linear rank ceiling; these results do not establish a quantum jump.',
        f'적대적 세트의 cyclic 규칙에서 직접 임베딩 interaction은 {percent(direct)}, KC interaction은 {percent(kc)}, 계수 수를 맞춘 KC interaction은 {percent(matched)}다. 이 스트레스 비교에서는 강한 직접 기준선이 앞선다. 큰 bridge rank에도 4개 logit 선형 rank 상한이 적용되며, 이 결과는 비약적 향상을 입증하지 않는다.')
    section(tr('Main finding: the strong baseline survives', '주요 결과: 강한 기준선이 유지됨'), '<p class="notice">'+esc(finding)+'</p>')
    adapter_config = records['adapters/report.json']['fixed_hyperparameters']
    ridge = records['holdout-final.json']['sites']['ridge']
    method = tr(
        f'The development protocol uses the original train32 and reused validation12. Ridge is fixed at {ridge}. Direct adapters run {adapter_config["epochs"]} epochs with {adapter_config["optimizer"]}, learning rate {adapter_config["learning_rate"]}, seeds {adapter_config["seeds"]}, using MPS training and CPU load-only evaluation. No native reward learning occurs here. This is a procedural one-shot final evaluation, not a claim of fully blinded research.',
        f'개발 절차는 기존 train32와 재사용 validation12를 사용한다. Ridge는 {ridge}로 고정했다. 직접 adapter는 {adapter_config["epochs"]} epoch, {adapter_config["optimizer"]}, 학습률 {adapter_config["learning_rate"]}, seed {adapter_config["seeds"]}로 MPS에서 학습하고 CPU에서 저장 모델을 불러 평가했다. Native 보상 학습은 하지 않았다. 절차상 최종 평가는 한 번 수행했지만 완전 맹검 연구라는 뜻은 아니다.')
    section(tr('Methods and computation paths', '방법과 계산 경로'), '<p>'+esc(method)+'</p>'+architecture_svg(ko))
    family_rows = []
    for family in sorted({row['family'] for row in cases['adversarial']}):
        values = []
        for model in ('interaction_kc', 'interaction_embedding'):
            evaluation = stress_models[model]['evaluation']
            selected = [i for i, base in enumerate(evaluation['base_case_indices']) if cases['adversarial'][base]['family'] == family]
            accuracy = sum(evaluation['predictions'][i] == evaluation['targets'][i] for i in selected) / len(selected)
            values.append(percent(accuracy))
        subset = [row for row in cases['adversarial'] if row['family'] == family]
        family_rows.append([family, len(subset), len({row['pair_id'] for row in subset}), *values])
    section(tr('Adversarial families: cyclic action accuracy', '적대적 상황군: cyclic 행동 정확도'),
        '<p>'+esc(tr('Each text is evaluated across the same external contexts; expanded rows are not independent cases. Family scores pool those context rows, keeping the original text/pair counts visible.', '각 문장을 같은 외부 문맥들에서 평가하므로 확장 행은 독립 사례가 아니다. 상황군 점수는 문맥 행을 합산하며 원래 문장·번역 쌍 수를 함께 표시한다.'))+'</p>'+
        table(['Family','Texts','Pairs','KC interaction','Direct embedding interaction'],family_rows))
    context_rows = []
    for name in cases:
        report = records[f'{name}-final.json']['probes']
        for rules in ('cyclic', 'alternate'):
            for model, row in report[rules]['models'].items():
                context_rows.append([name, rules, model, row['architecture']['coefficient_count'], percent(row['evaluation']['accuracy']), percent(row['shuffled_eval_context']['accuracy']), percent(row['missing_context']['accuracy']), percent(row['trained_with_shuffled_context']['accuracy'])])
    section(tr('Context comparisons: action targets', '문맥 비교: 행동 정답'), '<p>' + esc(tr('These targets depend on the external context rule; they are not semantic class labels. Coefficient matching does not match effective rank or geometry. A host latch is deterministic external state, not recurrent neural memory.', '정답 행동은 외부 문맥 규칙에 따라 달라지며 의미 범주 라벨과 다르다. 계수 수를 맞춰도 유효 rank·기하는 같지 않다. Host latch는 결정론적 외부 상태이며 신경 재귀 기억이 아니다.')) + '</p>' + table(['Set','Rule','Model',tr('Coefficients','계수'),tr('Normal','정상'),tr('Shuffled eval context','평가 문맥 교란'),tr('Missing context','문맥 없음'),tr('Shuffled train context','학습 문맥 교란')], context_rows))
    site_rows = []
    for name in cases:
        runs = records[f'{name}-final.json']['sites']['runs']
        for model in runs[0]['models']:
            values = [run['models'][model]['accuracy'] for run in runs]
            site_rows.append([name, model, len(values), stats(values)])
    section(tr('Readout sites: seed mean [minimum–maximum]', '읽기 위치: seed 평균 [최소–최대]'), table(['Set','Site/model','Seeds',tr('Semantic accuracy','의미 정확도')], site_rows) + '<p>' + esc(tr('Saved supervised read-only probes use matched readout width. Multi-site paths can bypass the sparse KC bottleneck via raw/PN features; MBON uses a previously trained checkpoint. Decodability is not evidence of causal use.', '저장된 지도 읽기 probe는 readout 폭을 맞춘다. 다중 위치 경로는 raw/PN 특징으로 희소 KC 병목을 우회할 수 있고 MBON은 이전 학습 checkpoint를 사용한다. 해독 가능성은 인과적 사용의 증거가 아니다.')) + '</p>')
    adapter_rows = []
    training = records['adapters/report.json']
    for kind, runs in training['models'].items():
        adapter_rows.append([kind, ', '.join(map(str, sorted({r['parameters'] for r in runs}))), ', '.join(map(str, sorted({r['effective_linear_rank'] for r in runs}))), stats([r['validation_accuracy'] for r in runs]), *[stats([r['accuracy'] for r in records[f'{name}-final.json']['adapters'][kind]]) for name in cases]])
    section(tr('Low-rank bridge capacity, not transformer LoRA', '저랭크 bridge 용량: transformer LoRA 아님'), table(['Adapter',tr('Trainable parameters','학습 파라미터'),tr('Effective rank','유효 rank'),'Validation','Holdout','Adversarial'], adapter_rows) + '<p>' + esc(tr('These direct linear embedding adapters bypass the connectome. Their complete input-to-four-logit map has rank at most four, so increasing adapter rank cannot add output rank. This rank ceiling limits conclusions about capacity saturation; it is not evidence that larger nonlinear models or real transformer LoRA cannot help.', '직접 선형 임베딩 adapter는 connectome을 우회한다. 전체 입력→4개 logit 변환의 rank는 최대 4이므로 adapter rank를 높여도 출력 rank를 늘릴 수 없다. 이 상한 때문에 용량 포화 해석이 제한되며, 더 큰 비선형 모델이나 실제 transformer LoRA가 도움이 없다는 증거는 아니다.')) + '</p>')
    mix = records['mixing-sanity.json']
    section(tr('mHC-inspired algebraic sanity check', 'mHC에서 착안한 대수적 검사'), table([tr('Quantity','항목'),tr('Value','값')], [[k, v] for k, v in mix.items() if k not in ('boundary','counterexample')]) + '<p>' + esc(tr('No mHC architecture was trained here. Stable uniform mixing can collapse stream rank; stable permutation can change stream identity. Numerical stability alone does not establish semantic correspondence.', '여기서는 mHC 구조를 학습하지 않았다. 안정적인 균등 혼합은 스트림 rank를 무너뜨릴 수 있고 안정적인 순열은 스트림 정체성을 바꿀 수 있다. 수치 안정성만으로 의미 대응이 입증되지 않는다.')) + '</p>')
    details = []
    final = records['adversarial-final.json']
    data = cases['adversarial']
    labels = final['labels']
    models = [('direct_embedding', final['direct_embedding']['predictions'])]
    for run in final['sites']['runs']:
        models += [(f"site/{name}/seed{run['seed']}", row['predictions']) for name,row in run['models'].items()]
    for name,runs in final['adapters'].items():
        models += [(f"adapter/{name}/seed{row['seed']}",row['predictions']) for row in runs]
    for model, predictions in models:
        failures = [[r['id'], r['language'], r['family'], r['text'], LABELS[truth], LABELS[pred]] for r,truth,pred in zip(data,labels,predictions) if truth != pred]
        details.append(f'<details><summary>{esc(model)} — {len(failures)} '+esc(tr('failures','오류'))+'</summary>'+table(['ID','Language','Family',tr('Full text','전문'),tr('Expected','정답'),tr('Predicted','예측')], failures)+'</details>')
    for rules in ('cyclic','alternate'):
        for model,entry in final['probes'][rules]['models'].items():
            row=entry['evaluation']; failures=[]
            for index,(truth,pred) in enumerate(zip(row['targets'],row['predictions'])):
                if truth != pred:
                    case=data[row['base_case_indices'][index]]
                    failures.append([case['id'],case['text'],row['contexts_used'][index],truth,pred])
            details.append(f'<details><summary>{esc("context/"+rules+"/"+model)} — {len(failures)} '+esc(tr('action errors','행동 오류'))+'</summary>'+table(['ID',tr('Full text','전문'),'Context',tr('Expected action','정답 행동'),tr('Predicted action','예측 행동')],failures)+'</details>')
    section(tr('Adversarial case audit: expand by model', '적대적 사례 점검: 모델별 펼치기'), '<p>'+esc(tr('All normal-evaluation failures are included, without cherry-picking. Semantic errors use category names; context errors use numeric action IDs. Search the browser page after opening a model section.', '정상 평가의 모든 오류를 선택 없이 포함했다. 의미 오류는 범주 이름으로, 문맥 오류는 숫자 행동 ID로 표시한다. 모델 항목을 펼친 뒤 브라우저 검색을 사용할 수 있다.'))+'</p>'+''.join(details))
    verification = records['verification.json']
    section(tr('Verification and provenance', '검증과 출처'), '<p>'+esc(tr('Saved inference replay is exact for both sets. This is not bitwise MPS retraining or biological replay.', '두 세트의 저장 모델 추론 재실행은 정확히 일치한다. MPS 재학습의 비트 단위 재현이나 생물학적 재현은 아니다.'))+'</p>'+table(['Artifact','SHA-256'], [[str(path.relative_to(ROOT)),sha(path)] for path in paths]+[[str(Path(__file__).resolve().relative_to(ROOT)),sha(Path(__file__))]]))
    refs=[('mHC','https://arxiv.org/html/2512.24880v2'),('LoRA','https://arxiv.org/abs/2106.09685'),('Prefix tuning','https://arxiv.org/abs/2101.00190'),('Prompt tuning','https://arxiv.org/abs/2104.08691'),('Flamingo','https://arxiv.org/abs/2204.14198'),('BLIP-2','https://arxiv.org/html/2301.12597v3')]
    section(tr('Primary references and interpretation', '원 연구와 해석 범위'), '<p>'+esc(tr('These sources motivate possible future methods; they are not claims that those methods were implemented in this study.', '다음 연구는 향후 방법의 근거이며 이번 연구에서 해당 방법을 구현했다는 뜻이 아니다.'))+'</p><ul>'+''.join(f'<li><a href="{url}">{esc(label)}</a></li>' for label,url in refs)+'</ul>')
    title=tr('Architecture study · supplementary report','구조 비교 연구 · 보충 보고서')
    css='''body{max-width:1180px;margin:0 auto;padding:44px 24px;font:16px/1.65 Georgia,"Noto Serif KR",serif;color:#18252e;background:#faf9f5}header{border-bottom:3px solid #183a47;padding-bottom:20px}h1{font-size:2.3rem;line-height:1.2}h2{font:700 1.35rem/1.4 system-ui;margin-top:40px}p{max-width:95ch}.notice{background:#edf2f2;border-left:4px solid #3e6970;padding:18px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font:13px/1.5 system-ui;margin:16px 0}th{background:#203e4b;color:white;text-align:left}th,td{padding:10px;border-bottom:1px solid #d6dcdc;vertical-align:top}td{overflow-wrap:anywhere}tr:nth-child(even){background:#f0f2ef}details{border:1px solid #cbd4d3;margin:10px 0;padding:12px}summary{cursor:pointer;font:600 14px/1.5 system-ui}a{color:#146073}footer{margin-top:45px;color:#53616a;font-size:13px}@media(max-width:600px){body{padding:20px 12px}h1{font-size:1.7rem}th,td{padding:7px}}@media print{body{max-width:none;background:white}details{break-inside:avoid}.scroll{overflow:visible}table{font-size:9px}}'''
    return f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{css}</style></head><body><header><p>Novigrad · supplementary methods and results</p><h1>{esc(title)}</h1><a href="architecture'+('.html' if ko else '.ko.html')+'">'+('English' if ko else '한국어')+'</a></header>'+intro+''.join(sections)+'<footer>'+esc(tr('Generated only from frozen JSON artifacts. No fitting or inference runs in this builder.', '고정 JSON 결과만으로 생성했다. 생성기는 학습이나 추론을 실행하지 않는다.'))+'</footer></body></html>'


def main():
    records,cases,paths=load()
    for language,name in [('en','architecture.html'),('ko','architecture.ko.html')]:
        destination=DEST/name
        destination.write_text(render(language,records,cases,paths))
        print(destination.relative_to(ROOT),sha(destination))


if __name__=='__main__': main()
