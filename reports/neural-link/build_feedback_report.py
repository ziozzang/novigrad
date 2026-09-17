#!/usr/bin/env python3
"""Build bilingual causal-feedback assay reports from immutable artifacts."""
import hashlib,html,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'results/causal-feedback';OUT=ROOT/'reports/neural-link'
LM_MODES=('original','self_writeback','belief_half','belief_full','static_prior_prefix','flipped_feedback','permuted_prototypes','uniform_prefix')
REFERENCES=('analytic_fixed_model','prior_ranked_no_repeat')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def esc(x):return html.escape(str(x))
def pct(x):return f'{100*x:.1f}%'
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{esc(x)}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{esc(x)}</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def integrity():
 rows=[];p=DATA/'protocol-lock.json'
 if p.exists():
  d=read(p);bad=[n for n,h in d.get('frozen',{}).items() if not (ROOT/n).exists() or sha(ROOT/n)!=h];rows.append(('protocol lock',sha(p),'PASS' if not bad else f'FAIL ({len(bad)})'))
 for name in ('evaluation-lock.json','verification.json'):
  q=DATA/name
  if q.exists():
   if name=='evaluation-lock.json':bad=[n for n,h in read(q).items() if not (DATA/n).exists() or sha(DATA/n)!=h];status='PASS' if not bad else f'FAIL ({len(bad)})'
   else:status='PASS' if read(q).get('all_hashes_pass') else 'FAIL'
   rows.append((name,sha(q),status))
 for label,path in (('posthoc-analysis',DATA/'posthoc-analysis.json'),('matched-reference',DATA/'matched-reference.json'),('actuator-noise report',ROOT/'results/actuator-noise/report.json'),('actuator-noise verification',ROOT/'results/actuator-noise/verification.json'),('hard-routing results',ROOT/'results/hard-feedback-routing/results.json'),('hard-routing verification',ROOT/'results/hard-feedback-routing/verification.json')):
  if path.exists():rows.append((label,sha(path),'PRESENT'))
 return rows
def summary_rows(modes):
 p=DATA/'results.json'
 if not p.exists():return []
 rows=[]
 for key,value in read(p).items():
  model,policy,scenario,mode=key.split('/')
  if mode not in modes:continue
  m=value['metrics'];rows.append((model,policy,scenario,mode,f"{m['successes']}/{m['n']}",pct(m['success_rate']),f"{m['mean_decisions']:.2f}",f"{m['mean_utility']:.3f}",m['initial_failures'],m['rescues'],m['invalid_episodes'],m['repeated_executed_actions']))
 return rows
def posthoc_section(lang):
 p=DATA/'posthoc-analysis.json'
 if not p.exists():return '<p>Pending.</p>'
 d=read(p);rows=[]
 for model,values in d['paired_native701_immediate_contrasts'].items():
  for label in ('belief_full_minus_static_prior_prefix','belief_full_minus_permuted_prototypes','belief_full_minus_prior_ranked_no_repeat'):
   v=values[label];ci=v['percentile_95_ci'];rows.append((model,label,f"{100*v['paired_success_difference_left_minus_right']:+.2f} pp",f"[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]"))
 note=('32개 authored 영·한 pair bootstrap이며 training run이나 topology 표본을 resample한 것이 아니다. 단일 training seed라 population/superiority 추론을 지원하지 않는다.' if lang=='ko' else 'The bootstrap resamples 32 authored bilingual pairs, not training runs or topology samples. With one training seed, it does not support population or superiority inference.')
 return '<p>'+note+'</p>'+table(['model','contrast','difference','paired 95% CI'],rows)

def matched_section(lang):
 p=DATA/'matched-reference.json'
 if not p.exists():return '<p>Pending.</p>'
 d=read(p);rows=[]
 for key,v in d['comparisons'].items():
  if '/native701/' not in key:continue
  rows.append((key.rsplit('/',1)[0],key.rsplit('/',1)[1],f"{100*v['success_gap_vs_belief_full']:+.2f} pp",v['helped_vs_belief_full'],v['hurt_vs_belief_full'],v['first_invalid_count'],f"{v['mean_physical_actions']:.2f}"))
 note=('이 matched reference는 첫 두 모델 결과를 본 뒤 설계한 사후 falsification이며 confirmatory 분석이 아니다. 동일한 저장 first action을 강제한 뒤 host Bayes 또는 no-repeat가 이어진다.' if lang=='ko' else 'This matched reference was designed post hoc after observing the first two model results. It is a stronger falsification, not confirmatory analysis: the exact saved first action is forced, followed by host Bayes or no-repeat.')
 return '<p>'+note+'</p>'+table(['model/policy','controller','gap vs belief-full','helped','hurt','first invalid','mean actions'],rows)

def hard_routing_section(lang):
 root=ROOT/'results/hard-feedback-routing';p=root/'results.json';vpath=root/'verification.json'
 if not p.exists():return '<p>Pending.</p>'
 d=read(p);verification=read(vpath) if vpath.exists() else {};rows=[]
 for model,value in d['comparisons'].items():
  rows.append((model,value['continuous_successes'],value['hard_successes'],f"{100*value['success_gap_hard_minus_continuous']:+.2f} pp",value['hard_helped'],value['hard_hurt'],f"{value['continuous_mean_physical_actions']:.2f}",f"{value['hard_mean_physical_actions']:.2f}"))
 intro=('연속 posterior mixture 결과를 본 뒤 선택한 사후 가설이다. belief_full에서 host posterior argmax가 class를 선택하고 해당 old-train labeled prototype 하나를 prefix로 보낸다. 이는 LM reasoning이 아니라 host hard routing이며 학습은 없다.' if lang=='ko' else 'This hypothesis was selected post hoc after seeing continuous-mixture results. Under belief_full, the host posterior argmax selects a class and routes one old-train labeled prototype as the prefix. This is host hard routing, not LM reasoning, and nothing is trained.')
 caveat=('learned 모델에서는 hard routing이 11개 case를 돕고 1개를 해쳤다. 따라서 단조 향상이 아니다. native701/immediate에서만 측정했으며 fresh generalization이나 architecture superiority 근거가 아니다. 손해를 본 case는 bilateral-water-night_shift-ko다. 연속 command 3→2→0은 성공했지만 hard 3→2→1→3은 실패했다. 마지막 host belief는 rest .427655, water .399485였고 fixed reliability .9 updater가 실패 action을 제거하지 않기 때문이다. 이는 controller 한계이며 반드시 decoder failure는 아니다.' if lang=='ko' else 'For the learned model, hard routing helped 11 cases and hurt one, so improvement is not monotonic. It was measured only under native701/immediate and provides no fresh-generalization or architecture-superiority evidence. The harmed case was bilateral-water-night_shift-ko: continuous commands 3→2→0 succeeded, while hard commands 3→2→1→3 failed. The final host belief still assigned rest .427655 versus water .399485 because the fixed-reliability .9 updater does not eliminate failed actions; this diagnoses a controller limitation, not necessarily a decoder failure.')
 replay=('3×64 전체 environment/host/cache trace는 exact replay됐다. 156개 새 prefix가 실제 생성되었고, museum family의 사전 지정 8개 case에서 새 prefix 16개(5/4/7)를 fresh LM으로 exact replay했다. 모든 family/문장 재생성은 아니다.' if lang=='ko' else 'All 3×64 environment/host/cache traces replayed exactly. The assay made 156 new actual generations; fresh LM replay exactly checked 16 new prefixes (5/4/7) reached by eight prespecified museum-family cases. This is not regeneration across every family or sentence.')
 status=('PASS' if verification.get('selected_actual_generation_exact') and verification.get('complete_environment_host_hard_prefix_cache_replay_exact') and verification.get('all_evaluation_hashes_pass') else 'PENDING/FAIL')
 return '<p>'+intro+'</p>'+table(['model','continuous','hard','gap','helped','hurt','continuous actions','hard actions'],rows)+'<p>'+caveat+'</p><p>'+replay+'</p><p><a href="../../results/hard-feedback-routing/results.json">results.json</a> · <a href="../../results/hard-feedback-routing/verification.json">verification.json</a> · verification: <b>'+status+'</b></p>'

def noise_section(lang):
 p=ROOT/'results/actuator-noise/report.json'
 if not p.exists():return '<p>Pending.</p>'
 d=read(p);groups={}
 for c in d['conditions']:groups.setdefault((c['policy'],c['expected_total']),[]).append(c['all_classes']['intended_agreement'])
 rows=[]
 for (policy,count),values in sorted(groups.items()):rows.append((policy,count,len(values),pct(sum(values)/len(values))))
 # Log-spaced count axis; agreement averaged over three fixed seeds.
 counts=sorted({k[1] for k in groups});parts=[]
 for policy,color in (('native601','#ffb86a'),('native701','#7de0c3')):
  pts=[]
  for i,count in enumerate(counts):pts.append(f"{55+i*150},{185-140*sum(groups[(policy,count)])/len(groups[(policy,count)])}")
  parts.append(f'<polyline points="{" ".join(pts)}" stroke="{color}"/><text x="60" y="{215+(0 if policy=="native601" else 15)}" fill="{color}">{policy}</text>')
 for i,count in enumerate(counts):parts.append(f'<text x="{55+i*150}" y="202">{count}</text>')
 svg='<svg class="curves" viewBox="0 0 720 240" role="img"><path d="M 50 35 V 190 H 680"/>'+''.join(parts)+'</svg>'
 note=('별도 독립 진단: old-train class prototype 4개에 대해 class당 256 Poisson replicate, seed 3개다. 이는 추상 count noise이며 검증된 생물학적 time window, 실제 fly data, 또는 generic OOD가 아니다.' if lang=='ko' else 'Separate diagnostic: 256 Poisson replicates per class over four old-train class prototypes and three seeds. This is abstract count noise, not a validated biological time window, real fly data, or generic OOD test.')
 return '<p>'+note+'</p>'+table(['policy','expected count','seeds','intended agreement'],rows)+svg+'<p><a href="../../results/actuator-noise/report.json">noise report.json</a> · <a href="../../results/actuator-noise/verification.json">verification.json</a></p>'

def focused_rows():
 p=DATA/'results.json'
 if not p.exists():return []
 data=read(p);rows=[]
 wanted=('original','belief_full','static_prior_prefix','permuted_prototypes','uniform_prefix','prior_ranked_no_repeat')
 for model in ('circuit_pooled_mlp_long','circuit_learned_long','circuit_fixed_queries_long'):
  for mode in wanted:
   value=data.get(f'{model}/native701/immediate/{mode}')
   if value:
    m=value['metrics'];rows.append((model,mode,f"{m['successes']}/{m['n']}",pct(m['success_rate']),m['rescues'],m['invalid_episodes'],f"{m['mean_utility']:.3f}"))
 return rows

def curves(lang):
 p=DATA/'results.json'
 if not p.exists():return '<p>Run pending.</p>'
 data=read(p);models=('circuit_pooled_mlp_long','circuit_learned_long','circuit_fixed_queries_long');modes=(('original','#ffb86a'),('belief_full','#7de0c3'),('prior_ranked_no_repeat','#95a9ff'))
 parts=[]
 for panel,model in enumerate(models):
  ox=panel*350;parts.append(f'<text class="title" x="{ox+175}" y="18">{model.replace("circuit_","")}</text>')
  for mode,color in modes:
   value=data.get(f'{model}/native701/immediate/{mode}');points=[]
   if value:
    for i,v in enumerate(value['metrics']['success_by_step']):points.append(f'{ox+42+i*82},{190-v*145}')
   parts.append(f'<polyline points="{" ".join(points)}" stroke="{color}"/><text x="{ox+45}" y="{210+modes.index((mode,color))*13}" fill="{color}">{mode}</text>')
  parts.append(f'<path d="M {ox+40} 40 V 192 H {ox+310}"/>')
 return '<svg class="curves" viewBox="0 0 1050 255" role="img">'+''.join(parts)+'</svg>'

def failures(lang):
 p=DATA/'results.json'
 if not p.exists():return '<p>'+('실행 대기 중.' if lang=='ko' else 'Run pending.')+'</p>'
 case_rows=read(DATA/'cases.json') if (DATA/'cases.json').exists() else [];blocks=[]
 for key,value in read(p).items():
  mode=key.rsplit('/',1)[-1]
  if '/native701/immediate/' not in key or mode not in ('original','belief_full','static_prior_prefix','permuted_prototypes','uniform_prefix'):continue
  bad=[r for r in value['cases'] if not r['success'] or r['invalid']]
  if not bad:continue
  items=[]
  for case in bad:
   trace=[]
   for step in case['trace']:
    trace.append(f"step {step['step']}: command={step.get('command')} executed={step.get('executed_action')} reward={step.get('delivered_observation',{}).get('reward')} physical_success={step.get('physical_success')} invalid={step.get('invalid')}")
   source=case_rows[case['case_index']] if case['case_index']<len(case_rows) else {}
   raw='\n'.join(f"step {z['step']} raw: {z.get('raw')}" for z in case['trace'])
   items.append('<article><b>case '+str(case['case_index'])+' · target '+esc(source.get('class',case['target']))+'</b><p><strong>Text:</strong> '+esc(source.get('text','unavailable'))+'</p><pre>'+esc('\n'.join(trace)+'\n'+raw)+'</pre></article>')
  blocks.append('<details><summary>'+esc(key)+' — '+str(len(bad))+' failures/invalid</summary>'+''.join(items)+'</details>')
 return ''.join(blocks)
def diagram(lang):
 labels=(['frozen initial prefix','frozen LM goal','601/701 actuator map','static-goal environment','binary reward','host Bayes update','norm-matched next prefix'] if lang=='en' else ['고정 initial prefix','고정 LM goal','601/701 actuator map','고정-goal 환경','binary reward','host Bayes update','norm-matched next prefix'])
 boxes=''.join(f'<g><rect x="{10+i*150}" y="24" width="125" height="58" rx="9"/><text x="{72+i*150}" y="49">{esc(v)}</text></g>' for i,v in enumerate(labels));arrows=''.join(f'<path d="M {135+i*150} 53 H {153+i*150}"/><path d="M {146+i*150} 47 l 7 6 -7 6"/>' for i in range(6));return f'<svg viewBox="0 0 1050 110" role="img">{boxes}{arrows}<path class="return" d="M 975 83 V 103 H 80 V 83"/></svg>'
def render(lang):
 ko=lang=='ko';lock=read(DATA/'protocol-lock.json');ready=(DATA/'results.json').exists();verified=(DATA/'verification.json').exists();maps=lock['maps']
 title='인과 피드백 진단' if ko else 'Causal feedback diagnostic'
 stage=('검증 완료' if verified else '실행 완료, 검증 대기' if ready else '실행 진행 중') if ko else ('Verified' if verified else 'Run complete; verification pending' if ready else 'Run in progress')
 headers=['model','actuator','scenario','mode','success/n','rate','decisions','utility','first failures','rescues','invalid','repeat actions']
 maprows=[]
 for name,v in maps.items():
  reachable=sorted(set(v['actions']));maprows.append((name,v['actions'],reachable,[x for x in range(4) if x not in reachable],f"{16*len(reachable)}/64"))
 ram='—'
 if (DATA/'run-manifest.json').exists():
  m=read(DATA/'run-manifest.json');ram=str(m.get('base_memory_before_sha256',m.get('base_memory_before_after_sha256','—')))+' / '+str(m.get('base_memory_after_sha256',m.get('base_memory_before_after_sha256','—')))
 body=f'''<header><p class="kicker">NOVIGRAD · POST-HOC MECHANISTIC ASSAY</p><h1>{title}</h1><p>{stage}</p></header><main>
<section class="warning"><h2>{'주장 경계' if ko else 'Claim boundary'}</h2><p>{'이것은 고정된 goal과 terminal success를 가진 4-action bandit 진단이다. 물리 자원 dynamics, reward-prediction error, LM 학습, 생물학적 학습 또는 recurrence를 모델링하지 않는다. Host가 Bayesian belief를 계산하고 label prototype prefix를 만든다.' if ko else 'This is a four-action bandit diagnostic with a static goal and terminal success. It does not model physical-resource dynamics, reward-prediction error, LM learning, biological learning, or recurrence. The host computes the Bayesian belief and constructs label-prototype prefixes.'}</p></section>
<section><h2>{'데이터 흐름' if ko else 'Data flow'}</h2>{diagram(lang)}<p>{'Target label은 환경과 evaluator에만 전달된다. 그러나 prior label 후보, old-train class prototype, command→execution calibration은 supervised host 정보다. 따라서 “hidden-label-free learning”이 아니다.' if ko else 'The target label enters only the environment and evaluator. Candidate labels in the prior, old-train class prototypes, and command-to-execution calibration are still supervised host information. This is not hidden-label-free learning.'}</p></section>
<section><h2>{'프로토콜' if ko else 'Protocol'}</h2><ul><li>{'같은 authored 64개를 재사용한 post-hoc mechanistic assay이며 fresh generalization set이 아니다.' if ko else 'This reuses the same authored 64 cases as a post-hoc mechanistic assay; it is not a fresh generalization set.'}</li><li>{'모든 prefix intervention은 initial prefix norm에 맞춘다. gain은 0(original), .5, 1로 고정되며 선택하지 않는다.' if ko else 'Every prefix intervention is matched to the initial-prefix norm. Gains are fixed at 0 (original), .5, and 1 without selection.'}</li><li>{'고정 updater reliability=.9, action cost=.05이다. immediate feedback에도 .9를 써서 analytic reference도 환경-optimal Bayes가 아니다.' if ko else 'The updater fixes reliability=.9 and action cost=.05. Reliability remains .9 under immediate deterministic feedback, so the analytic reference is not environment-optimal Bayes.'}</li><li>{'Delayed/noisy는 belief_full과 analytic_fixed_model만 평가하므로 전체 mode robustness 순위가 아니다.' if ko else 'Delayed/noisy scenarios cover only belief_full and analytic_fixed_model, so they do not rank robustness across all modes.'}</li><li>{'실제 success는 noisy/flipped/delayed reward와 무관하게 즉시 terminal이다. termination 자체가 informative하며 완전한 deception trajectory가 아니다.' if ko else 'Physical success terminates immediately despite noisy, flipped, or delayed reward. Termination itself is informative, so this is not a fully blinded deception trajectory.'}</li><li>{'모든 LM 조건의 첫 입력은 동일하고 첫 행동이 맞으면 즉시 종료된다. 따라서 누적 success는 구조상 original보다 낮아질 수 없고, uniform/permuted도 추가 시도만으로 향상될 수 있다. belief_full−original 차이만으로 feedback information gain을 주장할 수 없다.' if ko else 'Every LM condition has the same first input and stops immediately when that first action succeeds. Cumulative success therefore cannot fall below original by construction, and even uniform or permuted controls can improve through extra attempts. Belief-full minus original alone cannot identify information gain from feedback.'}</li><li>{'true-positive feedback 뒤에는 다음 정책 행동이 없다. 피드백 경로는 실패 관측에 따른 candidate 제거와 기만 positive만 시험하며 reward learning이 아니다. Noise의 false negative도 실제 성공 뒤 행동을 망가뜨릴 수 없어 일반 noisy-reward RL로 해석할 수 없다.' if ko else 'No policy action follows true-positive feedback. The feedback path tests candidate elimination after failures and deceptive positives, not reward learning. A noisy false negative cannot spoil behavior after physical success, so this is not a general noisy-reward RL test.'}</li></ul></section>
<section><h2>{'Actuator reachability' if not ko else 'Actuator 도달성'}</h2>{table(['map','commands→executed','reachable','unreachable','static-goal ceiling'],maprows)}<p>{'601은 실행 action 1,2에 도달하지 못해 균형 64개 중 최대 32개만 성공 가능하다. 701은 4개 action 모두 도달한다. 이는 actuator-map sensitivity이며 native policy의 causal effect가 아니다.' if ko else 'Map 601 cannot execute actions 1 or 2, so its ceiling is 32 successes among 64 balanced goals. Map 701 reaches all four actions. This tests actuator-map sensitivity, not a causal effect of a native policy.'}</p></section>
<section><h2>{'핵심 701 / immediate 비교' if ko else 'Focused 701 / immediate comparison'}</h2><p>{'Original, feedback, static/permuted/uniform controls와 no-repeat reference를 함께 읽어야 한다.' if ko else 'Read feedback alongside original, static/permuted/uniform controls, and the no-repeat reference.'}</p>{table(['model','mode','success/n','rate','rescues','invalid','utility'],focused_rows())}{curves(lang)}</section><section><h2>{'사후 paired 분석' if ko else 'Post hoc paired analysis'}</h2>{posthoc_section(lang)}</section><section><h2>{'First-action matched reference' if not ko else 'First-action matched reference'}</h2>{matched_section(lang)}</section><section><h2>{'Hard posterior routing' if not ko else 'Hard posterior routing'}</h2>{hard_routing_section(lang)}</section><section><h2>{'Actuator count-noise 진단' if ko else 'Actuator count-noise diagnostic'}</h2>{noise_section(lang)}</section><section><h2>{'실제 LM 조건' if ko else 'Actual-LM conditions'}</h2><p>{'original은 동일 deterministic prefix/output을 반복하며 독립 retry가 아니다. self_writeback은 실행 action이 아니라 LM이 명령한 goal prototype을 쓴다. static_prior_prefix는 reward를 사용하지 않지만 supervised prior mixture를 주입한다.' if ko else 'Original repeats the same deterministic prefix/output and is not an independent retry. Self-writeback uses the LM-commanded goal prototype, not the executed action. Static-prior-prefix ignores rewards but still injects a supervised prior mixture.'}</p><details><summary>{'전체 actual-LM condition 표' if ko else 'All actual-LM conditions'}</summary>{table(headers,summary_rows(LM_MODES))}</details></section>
<section><h2>{'Host analytic references' if ko else 'Host analytic references'}</h2><p>{'이 조건은 LM generation validity와 prefix interface를 우회하므로 capacity-matched LM baseline이 아니다.' if ko else 'These conditions bypass LM generation validity and the prefix interface; they are not capacity-matched LM baselines.'}</p><details><summary>{'전체 analytic reference 표' if ko else 'All analytic references'}</summary>{table(headers,summary_rows(REFERENCES))}</details></section>
<section><h2>{'Control 해석' if ko else 'Control interpretation'}</h2><p>{'permuted prototype과 uniform prefix는 semantic injection 대조다. flipped feedback에서 실패는 positive로 바뀌어 계속되지만 실제 성공은 negative로 전달된 뒤 즉시 종료되므로 부분적 deception이다. 환경은 static goal이며 reward history를 통해 host posterior만 바뀐다.' if ko else 'Permuted-prototype and uniform-prefix conditions probe semantic injection. Under flipped feedback, failures become positive and continue, while physical success is delivered as negative and immediately terminates; deception is partial. The environment goal stays static and reward history changes only the host posterior.'}</p></section>
<section><h2>{'실패·invalid trace' if ko else 'Failure and invalid traces'}</h2><p>{'가독성을 위해 native701/immediate의 original, belief-full, static, permuted, uniform 조건만 아래에 펼쳐 보이며 전체 trace는 raw JSON에 있다.' if ko else 'For readability, the expandable records below cover native701/immediate original, belief-full, static, permuted, and uniform conditions; raw JSON contains every trace.'}</p><p><a href="../../results/causal-feedback/results.json">results.json</a> · <a href="../../results/causal-feedback/generation-cache.json">generation-cache.json</a></p>{failures(lang)}</section>
<section><h2>{'재현성과 동결' if ko else 'Replay and freezing'}</h2>{table(['artifact','sha256','status'],integrity())}<p>Base RAM before / after: {esc(ram)}</p><div class="warning"><h3>{'Portable replay provenance gap' if not ko else 'Portable replay provenance 누락'}</h3><p>{'원래 model protocol inventory가 function/chat_template.jinja를 포함하지 않았다. 따라서 model weight는 사전 동결됐지만 chat template dependency가 결과 전에 inventory로 동결됐다고 소급 주장할 수 없다. 첫 portable copy는 목록 파일만 복사해 tokenizer chat template가 없어 실제 생성에 실패했으며 이 실패 기록은 보존한다. 이후 post-outcome supplementary lock인 model-runtime-dependencies.json이 원본 template 13,792 bytes, SHA-256 db61fb01017bd82401d3ffca4f8e066cd56ff6d38d2a30c4058770c0bf7ab49b를 고정하고 wrapper가 hash를 요구한다. 기존 수치 결과는 변경하지 않았으며 이후 exact replay는 현재 고정 template를 별도로 검증한다.' if ko else 'The original model protocol inventory omitted function/chat_template.jinja. The model weights were frozen, but the chat-template dependency cannot be retroactively claimed as pre-outcome frozen. The first portable copy retained only listed files and actual generation failed because the tokenizer had no chat template; that failed attempt is retained. A post-outcome supplementary lock, model-runtime-dependencies.json, now pins the original 13,792-byte template at SHA-256 db61fb01017bd82401d3ffca4f8e066cd56ff6d38d2a30c4058770c0bf7ab49b, and the wrapper requires that hash. Original numeric results remain unchanged; later exact replay separately validates the currently pinned template.'}</p></div><p>{'전체 environment/host/prefix dynamics는 저장된 LM 문자열로 2,926개 unique 생성 prefix(903/951/1,072)를 exact replay한다. 실제 LM 재호출은 사전 지정된 museum family 8개 case(4 goals×영·한)에서 만난 358개 prefix(108/95/155)를 exact replay했다. 모든 문장, 모든 family, 전체 64개를 재생성한 것은 아니다.' if ko else 'The complete environment/host/prefix dynamics replay exactly from saved LM strings across 2,926 unique generated prefixes (903/951/1,072). Fresh LM calls exactly replayed 358 selected prefixes (108/95/155) encountered by eight prespecified cases: four goals × English/Korean, all from the museum family. This does not regenerate every sentence, every family, or all 64 cases.'}</p></section></main>'''
 css='''html,body{max-width:100%;overflow-x:hidden}body{margin:0;background:#071017;color:#dce8ed;font:16px/1.55 system-ui}header,main{max-width:1180px;margin:auto;padding:34px}header{padding-top:68px}.kicker{letter-spacing:.18em;color:#ffb86a}h1{font-size:clamp(2.5rem,7vw,5.8rem);line-height:.94}h2{color:#7de0c3;margin-top:44px}.warning{border-left:5px solid #ffb86a;background:#13232c;padding:20px}*{box-sizing:border-box}section,details,.scroll{min-width:0;max-width:100%}.scroll{width:100%;overflow-x:auto;overflow-y:hidden;contain:inline-size}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #29404b;text-align:left}th{color:#ffb86a}details{background:#101e26;margin:9px 0;padding:12px}summary{cursor:pointer;color:#ffb86a;font-weight:700}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#071017;padding:10px}td,pre,p{overflow-wrap:anywhere}h1{word-break:keep-all}.curves polyline{fill:none;stroke-width:3}.curves .title{fill:#dce8ed;text-anchor:middle;font-size:11px}svg{width:100%;background:#101e26;border-radius:12px}svg rect{fill:#17333d;stroke:#7de0c3}svg text{fill:#dce8ed;text-anchor:middle;font-size:9px}svg path{stroke:#ffb86a;fill:none}.return{stroke-dasharray:5 4}@media(max-width:700px){header,main{padding:20px}}@media print{body{background:white;color:#111}header,main{max-width:none;padding:12px}.warning,details,svg{background:white}h2,.kicker,th,summary{color:#174f46}details{display:block}details>*{display:block!important}table{break-inside:auto}tr,article,section{break-inside:avoid}pre{background:#f4f4f4;color:#111}.scroll{overflow:visible}}'''
 return f'<!doctype html><html lang="{lang}"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{title}</title><style>{css}</style>{body}</html>'
def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'feedback.html').write_text(render('en'));(OUT/'feedback.ko.html').write_text(render('ko'));print('wrote feedback reports')
if __name__=='__main__':main()
