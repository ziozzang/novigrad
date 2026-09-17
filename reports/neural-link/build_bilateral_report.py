#!/usr/bin/env python3
"""Build the bilingual bilateral bridge supplement from frozen result JSON only."""
import hashlib,html,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'results/bilateral-bridge';OUT=ROOT/'reports/neural-link'
VARIANTS=('circuit_learned','circuit_fixed_queries','circuit_pooled_mlp','embedding_learned','label_oracle','circuit_learned_long','circuit_fixed_queries_long','circuit_pooled_mlp_long','embedding_learned_long')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pct(x):return '—' if x is None else f'{100*x:.1f}%'
def esc(x):return html.escape(str(x))
def td(x):return f'<td>{esc(x)}</td>'
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{esc(x)}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(td(x) for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def integrity():
 status=[];lock=DATA/'protocol-lock.json'
 if lock.exists():
  record=read(lock);bad=[name for name,digest in record.get('frozen',{}).items() if not (ROOT/name).exists() or sha(ROOT/name)!=digest]
  status.append(('Protocol lock',sha(lock),'PASS' if not bad else f'FAIL ({len(bad)})'))
 started=DATA/'final-started.json'
 if started.exists() and lock.exists():status.append(('Final→protocol',read(started).get('protocol_sha256',''), 'PASS' if read(started).get('protocol_sha256')==sha(lock) else 'FAIL'))
 evaluation=DATA/'evaluation-lock.json'
 if evaluation.exists():
  bad=[name for name,digest in read(evaluation).get('hashes',{}).items() if not (ROOT/name).exists() or sha(ROOT/name)!=digest]
  status.append(('Evaluation lock',sha(evaluation),'PASS' if not bad else f'FAIL ({len(bad)})'))
 return status

def temporal_rows(final):
 d=read(DATA/('temporal-final.json' if final else 'temporal/development.json'));rows=[]
 for key,values in d['models'].items():
  vals=values if isinstance(values,list) else [values]
  metrics=vals if final else [v['validation'] for v in vals]
  rows.append((key,len(vals),pct(sum(m['accuracy'] for m in metrics)/len(metrics)),pct(sum(m.get('zero_current_accuracy',0) for m in metrics)/len(metrics)),pct(sum(m.get('row_shuffle_accuracy',0) for m in metrics)/len(metrics))))
 return rows

def lm_rows(final):
 rows=[]
 for name in VARIANTS:
  path=DATA/(f'{name}-final.json' if final else f'lm/{name}-development.json')
  if not path.exists():continue
  d=read(path);a,z,s=(d[k] for k in ('actual','zero_prefix','row_shuffled_prefix'));m=d.get('training_mean_prefix')
  steps=480 if name.endswith('_long') else 120
  rows.append((name,steps,f"{a['correct_count']}/{a['valid_count']}/{a['generated_count']}",pct(a['rank_accuracy']),f"{z['correct_count']}/{z['valid_count']}/{z['generated_count']}",f"{s['correct_count']}/{s['valid_count']}/{s['generated_count']}",f"{m['correct_count']}/{m['valid_count']}/{m['generated_count']}" if m else '—'))
 return rows

def failures_html(lang):
 case_path=ROOT/'examples/bio_bridge/bilateral_holdout.json'
 cases=read(case_path) if case_path.exists() else []
 if isinstance(cases,dict):cases=cases.get('cases',[])
 by_id={r.get('id'):r for r in cases};blocks=[]
 for name in VARIANTS:
  p=DATA/f'{name}-final.json'
  if not p.exists():continue
  d=read(p);ids=d.get('case_ids',[]);items=[]
  for r in d['actual']['rows']:
   if r.get('correct',False):continue
   case_id=ids[r['index']] if r['index']<len(ids) else r['index'];case=by_id.get(case_id,{})
   items.append('<article class="failure"><b>'+esc(case_id)+'</b><p><strong>Text:</strong> '+esc(case.get('text','unavailable'))+'</p><p><strong>Target:</strong> '+esc(case.get('class',r['target']))+' · <strong>Parsed:</strong> '+esc(r.get('parsed_goal'))+' · <strong>Valid:</strong> '+esc(r.get('valid'))+' · <strong>Rank:</strong> '+esc(r.get('rank_prediction'))+'</p><pre>'+esc(r.get('free_output',''))+'</pre></article>')
  if items:blocks.append('<details><summary>'+esc(name)+' — '+str(len(items))+' failures</summary>'+''.join(items)+'</details>')
 return ''.join(blocks) or '<p>'+('최종 artifact 대기 중.' if lang=='ko' else 'Pending final artifacts.')+'</p>'


def policy_swap(lang):
 p=DATA/'policy-swap-final.json'
 if not p.exists():return '<p>Pending final policy-swap artifact.</p>'
 d=read(p);rows=[]
 if isinstance(d,dict):
  source=d.get('results',d)
  for name,value in source.items():
   if isinstance(value,dict):
    metrics=value.get('metrics',value)
    rows.append((name,metrics.get('n','—'),metrics.get('goal_stable','—'),metrics.get('correct_stable','—'),metrics.get('wrong_fixed_points','—'),metrics.get('native_action_correct_external_need','—')))
 return table(['policy / variant','n','goal stable','correct stable','wrong fixed points','native correct'],rows) if rows else '<pre>'+esc(json.dumps(d,ensure_ascii=False,indent=2))+'</pre>'

def posthoc(lang):
 p=DATA/'posthoc-analysis.json'
 if not p.exists():return '<p>Pending post-hoc artifact.</p>'
 d=read(p);comparison=[]
 for name,value in d['paired_long_comparisons'].items():
  pair=value['paired_32_enko_clusters'];family=value['optional_8_family_clusters']
  comparison.append((name,f"{100*pair['paired_accuracy_difference_left_minus_right']:+.2f} pp",f"[{100*pair['percentile_95_ci'][0]:+.2f}, {100*pair['percentile_95_ci'][1]:+.2f}]",f"[{100*family['percentile_95_ci'][0]:+.2f}, {100*family['percentile_95_ci'][1]:+.2f}]"))
 transitions=[]
 for name,value in d['step_120_to_480_transitions'].items():transitions.append((name,value['help'],value['hurt'],value['both_correct'],value['both_wrong'],value['net_help_minus_hurt']))
 intro=('사후 기술 분석이며 selection이나 superiority 검정이 아니다. 기본 bootstrap 단위는 32개 영·한 pair이고, 8개 scenario family 분석은 cluster 수가 매우 작다.' if lang=='ko' else 'This is post hoc descriptive analysis, not selection or a superiority test. The primary bootstrap unit is 32 bilingual pairs; the eight-family analysis has very few clusters.')
 finding=('learned-long과 fixed/pooled의 pair-bootstrap CI는 0을 포함한다. direct-embedding 대비 차이는 +20.31 pp, 95% CI [+6.25, +34.38]였지만 topology의 causal effect로 해석할 수 없다.' if lang=='ko' else 'The pair-bootstrap intervals for learned-long versus fixed and pooled include zero. Its contrast with direct embedding is +20.31 pp, 95% CI [+6.25, +34.38], but this cannot identify a causal effect of topology.')
 shuffle=('저장된 shuffle은 64개 중 20개가 같은 class로 배정되고 exact row fixed point는 0개다. 따라서 class 정보를 완전히 제거한 대조가 아니다.' if lang=='ko' else 'The saved shuffle assigns 20 of 64 rows to the same class and has zero exact row fixed points. It therefore does not remove all class information.')
 return '<p>'+intro+'</p>'+table(['comparison','difference','pair 95% CI','family 95% CI'],comparison)+'<p>'+finding+'</p><h3>120→480</h3>'+table(['mode','helped','hurt','both correct','both wrong','net help−hurt'],transitions)+'<p>'+shuffle+'</p>'

def svg(lang):
 labels=['authored text / 작성 문장','frozen embedding / 고정 임베딩','site-time packet / 사이트·시간 패킷','4 latent queries / 잠재 쿼리 4개','FP32 adapter / FP32 어댑터','frozen BF16 LM / 고정 BF16 LM','tool call / 도구 호출']
 x=[10,165,320,475,630,785,940];boxes=''.join(f'<g><rect x="{a}" y="30" width="130" height="62" rx="10"/><text x="{a+65}" y="55">{labels[i].split(" / ")[0 if lang=="en" else 1]}</text></g>' for i,a in enumerate(x));arrows=''.join(f'<path d="M {a+130} 61 H {x[i+1]-7}"/><path d="M {x[i+1]-14} 55 l 7 6 -7 6"/>' for i,a in enumerate(x[:-1]));return f'<svg viewBox="0 0 1080 122" role="img" aria-label="data flow">{boxes}{arrows}</svg>'
def render(lang):
 ko=lang=='ko';final=all((DATA/f'{n}-final.json').exists() for n in VARIANTS);temporal_final=(DATA/'temporal-final.json').exists();verified=(DATA/'verification.json').exists()
 title='양방향 신경 링크: 엄격한 보충 보고서' if ko else 'Bilateral neural link: rigorous supplement'
 strings={
 'boundary':('이 보고서의 typed site/time 경로는 모의 특징을 pooling한다. 생물학적 token, 실제 신경 기록, 사고 해독의 증거가 아니다. 실제 LM 경로는 별도 실험으로, 고정 FunctionGemma에 학습된 soft prefix를 입력한다.' if ko else 'The typed site/time path pools simulated features. They are not biological tokens, real neural recordings, or evidence of thought decoding. The actual-LM path is a separate experiment: a learned soft prefix enters frozen FunctionGemma.'),
 'stage':('최종 64개 평가와 검증 파일 사용' if final else '개발 결과만 표시 — 최종 평가 대기 중') if ko else ('Final 64-case evaluation included' if final else 'Development results only — final evaluation pending'),
 }
 dev=read(DATA/'temporal/development.json');latest=read(DATA/'baseline.json')['latest_only']['accuracy']
 loop=''
 if (DATA/'closed-loop-final.json').exists():
  lr=[]
  for name,d in read(DATA/'closed-loop-final.json').items():
   m=d['metrics'];lr.append((name,m['n'],m['goal_stable'],m['correct_stable'],m['wrong_fixed_points'],m['native_evaluated'],m['native_action_correct_external_need']))
  loop='<h2>'+('폐루프 결과' if ko else 'Closed-loop outcomes')+'</h2>'+('<p>Tick 2는 LLM goal → host의 old-train class prototype 조회 → site features → LLM goal 순서다. native action은 side diagnostic이며 다음 prefix에 들어가지 않는다. 따라서 목표 안정성은 native action 정답성과 다르고 잘못된 목표가 안정된 고정점이 될 수 있다. action feedback, error correction, learned recurrence가 아니다.</p>' if ko else '<p>Tick 2 follows LLM goal → host lookup of the old-train class prototype → site features → LLM goal. The native action is a side diagnostic and never enters the next prefix. Goal stability therefore differs from native-action correctness, and an incorrect generated goal can become a stable wrong fixed point. This is not action feedback, error correction, or learned recurrence.</p>')+table(['variant','n','goal stable','correct stable','wrong fixed points','native n','native correct'],lr)
 body=f'''<header><p class="kicker">NOVIGRAD · BILATERAL SUPPLEMENT</p><h1>{title}</h1><p>{strings['stage']}</p></header><main>
<section class="warning"><strong>{'해석 경계' if ko else 'Interpretation boundary'}</strong><p>{strings['boundary']}</p></section>
<section><h2>{'두 경로를 분리해서 읽기' if ko else 'Keep the two paths separate'}</h2>{svg(lang)}<div class="twocol"><article><h3>Typed site/time</h3><p>{'32차원 모의 site 특징, site ID와 상대시간을 learned/fixed-query/pooled resampler가 4개 latent로 줄인다. 이것은 supervised temporal classification이다.' if ko else 'Learned, fixed-query, and pooled resamplers reduce simulated 32-D site features, site IDs, and relative times to four latents. This is supervised temporal classification.'}</p></article><article><h3>Actual LM soft prefix</h3><p>{'32차원 latent 네 개를 각각 640차원으로 투영해 4×640 soft prefix를 만든다. 원본 BF16 LM은 고정되고 adapter는 FP32이다. base model fine-tuning이 아니다.' if ko else 'Four 32-D latents are each projected to 640 dimensions, forming a 4×640 soft prefix. The original BF16 LM stays frozen while the adapter trains in FP32. This is not base-model fine-tuning.'}</p></article></div></section>
<section><h2>{'데이터와 프로토콜' if ko else 'Data and protocol'}</h2><ul><li>{'기존 개발: train 32 / validation 12.' if ko else 'Earlier development: train 32 / validation 12.'}</li><li>{'새 최종 세트: 저자가 작성한 64개 문장, 32개 영·한 쌍. 번역 쌍은 독립 표본이 아니다.' if ko else 'New final set: 64 authored texts in 32 English–Korean pairs. Translation pairs are dependent.'}</li><li>{'9개 variant: 120-step 5개와 480-step long 4개. hyperparameter selection 없이 모두 평가.' if ko else 'Nine variants: five at 120 steps and four long variants at 480 steps; all are evaluated without selection.'}</li><li>{'LM 학습 seed는 하나다. 효과의 seed 안정성을 추정하지 않는다.' if ko else 'LM training uses one seed, so it does not estimate seed stability.'}</li><li>{'120/480 step은 같은 seed stream을 사용한다. long 설계는 개발 결과를 본 뒤 추가되었으며 최종 selection은 없었지만 완전히 사전 독립적인 설계가 아니다.' if ko else 'The 120- and 480-step runs share one seed stream. Long runs were added after development results were seen; although there is no final-set selection, the design was development-informed.'}</li><li>{'label oracle 120-step은 label leakage positive control이며 480-step 모델과 동일 budget의 upper bound가 아니다.' if ko else 'The 120-step label oracle is a leakage positive control, not a same-budget upper bound for the 480-step models.'}</li></ul></section>
<section><h2>{'방법 상세' if ko else 'Method details'}</h2><p>{'Temporal 분류기는 3개 site×3개 time의 9 token을 사용한다. 실제 LM 실험은 각 예제를 current-time의 static 3 token packet으로 구성하므로 두 입력 체계를 혼동하면 안 된다.' if ko else 'Temporal classification uses nine tokens: three sites across three times. The actual-LM experiment instead uses a static three-token packet at the current time; these input regimes must remain distinct.'}</p><p>{'각 site의 319/5,177/96차원 입력은 고정된 seeded projection을 거쳐 32차원 특징이 된다. resampler는 4×32 latent를 만들고, 학습된 projection이 각 32차원 latent를 640차원으로 바꿔 4×640 soft prefix를 만든다.' if ko else 'The 319-, 5,177-, and 96-dimensional site inputs each pass through a fixed seeded projection into 32-D features. The resampler produces four 32-D latents, and a learned projection maps each latent to 640 dimensions, yielding a 4×640 soft prefix.'}</p><p>{'도구 공간은 set_goal 함수 하나와 4개 enum 값으로 닫혀 있다. open-ended tool selection 실험이 아니다.' if ko else 'The tool space is closed: one set_goal function with four enum values. This is not open-ended tool selection.'}</p></section><section><h2>{'시간 resampler' if ko else 'Temporal resamplers'}</h2><p>{'분석적 latest-only 기준선' if ko else 'Analytic latest-only baseline'}: <b>{pct(latest)}</b>. {'개발 temporal packet은 조건별 384개 train / 48개 validation 관측을 만든다. 100% latest-only 분석 comparator는 개발 전용이며 이에 대응하는 final comparator는 평가하지 않았다.' if ko else 'Development temporal packets contain 384 train / 48 validation observations per condition. The 100% latest-only analytic comparator belongs to development only; no corresponding final comparator was evaluated.'}</p><p>{('최종 authored 세트는 role/time distractor 분포가 달라 개발 대비 명확한 성능 하락을 보인다. 이는 분포 이동이며 일반화 성공으로 과장해서는 안 된다.' if temporal_final else '최종 표가 아직 완성되지 않았다.') if ko else ('The final authored set changes the role/time-distractor distribution and shows a clear drop from development. This is distribution shift and must not be presented as broad generalization.' if temporal_final else 'The complete final table is not yet available.')}</p>{table(['model','seeds','accuracy','zero current','row shuffle'],temporal_rows(temporal_final))}</section>
<section><h2>{'실제 LM 결과' if ko else 'Actual LM results'}</h2><p>{'주 지표는 전체 LM 호출의 자유 생성 tool call이다. 4-candidate likelihood ranking은 보조 지표다. warm 응답은 16 token, 다른 응답은 15 token이므로 ranking에는 길이 편향 가능성이 있다.' if ko else 'The primary metric is the freely generated tool call from a full LM invocation. Four-candidate likelihood ranking is secondary. The warm response has 16 tokens while the others have 15, creating a possible length bias.'}</p>{table(['variant','steps','actual correct/valid/n','candidate rank','zero c/v/n','shuffle c/v/n','train-mean c/v/n'],lm_rows(final))}<p>{'zero와 training-mean 생성은 각각 동일 prefix의 물리적 출력 하나를 행들에 재사용하며 shuffle도 source-prefix cache를 공유할 수 있다. 표시된 행은 논리적 평가이며 독립 생성 trial이 아니다.' if ko else 'Zero and training-mean generation each reuse one identical-prefix physical output across rows; shuffled controls may also share source-prefix cache identities. The displayed rows are logical evaluations, not independent generation trials.'}</p></section>
<section><h2>{'사후 분석' if ko else 'Post hoc analysis'}</h2>{posthoc(lang)}</section><section><h2>{'케이스별 실제 생성 실패' if ko else 'Per-case actual-generation failures'}</h2>{failures_html(lang)}</section>{loop}<section><h2>{'정책 교환 대조' if ko else 'Policy-swap control'}</h2>{policy_swap(lang)}<p>{'601→701 교환은 MBON feature와 native policy를 동시에 바꾸므로 native-policy의 causal effect를 격리하지 않는다.' if ko else 'The 601→701 swap changes MBON features and the native policy together, so it does not isolate a causal effect of the native policy.'}</p></section>
<section><h2>{'재현성과 무결성' if ko else 'Replay and integrity'}</h2>{table(['check','sha256','status'],integrity())}<p>{('검증 완료 후 candidate 64개 전부의 likelihood/rank를 replay하고, 4개 조건마다 사전 선언된 8개 index를 자유 생성으로 replay한다(variant당 논리적 32행).' if ko else 'Once verification is complete, replay covers candidate likelihoods and ranks for all 64 rows, plus eight predeclared free-generation rows in each of four conditions (32 logical rows per variant).')} <b>{(('검증 파일 있음' if verified else '검증 대기') if ko else ('verification present' if verified else 'verification pending'))}</b></p><p>{'120-step LM 소스는 source-snapshots에 보관되어 있다. 학습 전후 전체 RAM base-model hash와 base gradient 부재는 480-step 4개 variant의 불변성을 입증하며, 이 강한 근거를 120-step run에 소급 주장하지 않는다.' if ko else 'The archived 120-step LM source is preserved under source-snapshots. Complete in-RAM base-model hashes before/after training and absent base gradients establish immutability for the four 480-step variants; that stronger evidence is not retroactively claimed for the 120-step runs.'}</p></section>
<section><h2>{'연구 근거와 한계' if ko else 'Research basis and limits'}</h2><p><a href="../../examples/bio_bridge/BILATERAL_RESEARCH{'.ko' if ko else ''}.md">BILATERAL_RESEARCH</a> · <a href="../../examples/bio_bridge/PAIRED_DATA_RESEARCH.md">PAIRED_DATA_RESEARCH</a></p><p>{'문헌은 closed-loop 규율과 paired-data 선택을 뒷받침한다. 현재 software feature와 생물학적 신경 활동의 대응을 입증하지 않는다. ORN 실제 데이터 importer와 test는 성공했지만 원격 데이터 접근은 HTTP 403/401로 차단되어 ORN model이나 metric은 보고하지 않는다.' if ko else 'The sources motivate closed-loop discipline and paired-data choices. They do not establish correspondence between these software features and biological neural activity. The ORN real-data importer and tests succeeded, but remote dataset access returned HTTP 403/401; therefore no ORN model or metric is reported here.'}</p></section></main>'''
 css='''body{margin:0;background:#081018;color:#dce8ed;font:16px/1.55 system-ui}header,main{max-width:1120px;margin:auto;padding:34px}header{padding-top:70px}h1{word-break:keep-all;font-size:clamp(2.4rem,7vw,5.5rem);line-height:.95;max-width:900px}h2{margin-top:45px;color:#7ee0c3}.kicker{color:#ffba69;letter-spacing:.18em}.warning{border-left:5px solid #ffba69;background:#13222b;padding:20px}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:18px}.twocol article{background:#101d25;padding:18px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #29404c;text-align:left}th{color:#ffba69}svg{width:100%;background:#101d25;border-radius:14px}svg rect{fill:#17313b;stroke:#7ee0c3}svg text{fill:#dce8ed;text-anchor:middle;font-size:10px}svg path{stroke:#ffba69;fill:none}a{color:#7ee0c3}details{background:#101d25;margin:10px 0;padding:14px}summary{cursor:pointer;color:#ffba69;font-weight:700}.failure{border-top:1px solid #29404c;padding:12px 0}.failure pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#081018;padding:10px}@media(max-width:700px){.twocol{grid-template-columns:1fr}header,main{padding:20px}}@media print{body{background:white;color:#17232b;font-size:11pt}header{padding-top:10px}h1{font-size:32pt}h2{color:#17634f}.warning,.twocol article,details,.failure pre{background:#f3f6f7}a{color:#17634f}summary,th,.kicker{color:#7d4a00}.scroll{overflow:visible}table{font-size:9pt}td{overflow-wrap:anywhere}.failure{break-inside:avoid}svg{background:#f3f6f7}svg rect{fill:#edf4f1;stroke:#17634f}svg text{fill:#17232b}svg path{stroke:#7d4a00}}'''
 return f'<!doctype html><html lang="{lang}"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{title}</title><style>{css}</style>{body}</html>'
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/'bilateral.html').write_text(render('en'));(OUT/'bilateral.ko.html').write_text(render('ko'))
 print('wrote bilateral.html and bilateral.ko.html')
if __name__=='__main__':main()
