"""Replay real generated tool calls through novi into a host-grounded 2D task.

The language model runs once per unique request in deep_evaluate.py. Its exact
recorded output is reparsed here, then a fresh Session invokes native novi.
The policy is frozen; physical episodes reset pose, memory and random streams.
This is a simulation integration benchmark, not an online HTTP/robot service.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from navigation import CircularHeadingEstimator, AngularSteeringComparator, wrap_degrees
from protocol import Session, parse

GOALS = ('water','food','warmth','rest')
VARIANTS = ('oracle_goal','parsed_direct','novi_memory','novi_no_odometry','novi_swapped_decoder')
SEEDS = tuple(range(2101,2111))
SCENARIOS = ('visible','dropout')
ROOT = Path(__file__).resolve().parents[2]


def station_map(seed):
    """Host map: resource identities remain fixed; spatial bearings change."""
    rng = np.random.default_rng(seed)
    rotation = rng.uniform(-np.pi,np.pi)
    angles = rotation + np.arange(4)*np.pi/2
    return {g:4.0*np.array([np.cos(x),np.sin(x)]) for g,x in zip(GOALS,angles)}


def simulate(expected, selected, seed, scenario, use_odometry=True):
    if expected not in GOALS or selected not in (*GOALS,None):
        raise ValueError('unknown station')
    if scenario not in SCENARIOS:
        raise ValueError('unknown scenario')
    if selected is None:
        return {'success':False,'reward':0,'reached':None,'steps':0,'distance_to_requested':4.0,'control_seconds':[]}
    stations = station_map(seed)
    rng = np.random.default_rng(seed+10000)
    # Draw full exogenous streams before actions; variants share noise by tick.
    vision_noise = rng.normal(0,2.5,120)
    odo_noise = rng.normal(0,.6,120)
    disturbance = 3*np.sin(np.arange(120)*.31)+rng.normal(0,.5,120)
    heading = float(rng.uniform(-180,180))
    xy = np.zeros(2)
    estimate = CircularHeadingEstimator()
    steering = AngularSteeringComparator()
    last_motion = None
    times = []
    reached = None
    for step in range(120):
        visual = float(wrap_degrees(heading+vision_noise[step]))
        if scenario=='dropout' and 8<=step<70:
            visual = None
        odo = None if last_motion is None or not use_odometry else float(last_motion+odo_noise[step])
        # Localization and map are assumed exact host services. Heading is noisy.
        bearing = float(np.degrees(np.arctan2(*(stations[selected]-xy)[::-1])))
        start = time.perf_counter()
        estimated = estimate.update(visual,odo)
        steering.set_target(bearing)
        turn = steering.choose_turn(estimated)
        # Slow forward movement while misaligned: fixed engineered motor decoder.
        error = abs(float(wrap_degrees(bearing-estimated)))
        speed = .16*max(.15,float(np.cos(np.radians(error))))
        times.append(time.perf_counter()-start)
        last_motion = turn+disturbance[step]
        heading = float(wrap_degrees(heading+last_motion))
        xy += speed*np.array([np.cos(np.radians(heading)),np.sin(np.radians(heading))])
        for goal, location in stations.items():
            if np.linalg.norm(xy-location)<=.35:
                reached = goal
                break
        if reached is not None:
            break
    success = reached==expected
    return {'success':success,'reward':1 if success else (-1 if reached is not None else 0),'reached':reached,'steps':step+1,'distance_to_requested':float(np.linalg.norm(xy-stations[expected])),'control_seconds':times}


def dispatch(raw, backend):
    """Generated intent is never replaced with evaluator's expected goal."""
    session = Session(backend.decide)
    name,args = parse(raw)
    if name!='set_goal':
        return None,None
    session.call(name,args)
    action = session.call('choose_action',{})
    return args['goal'],action['meaning']


def write_report(path, report):
    """Keep one physical episode per line for reviewable, compact raw JSON."""
    metadata = {k:v for k,v in report.items() if k != 'runs'}
    header = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
    rows = ',\n'.join('    '+json.dumps(row, ensure_ascii=False, allow_nan=False) for row in report['runs'])
    path.write_text(header[:-2]+',\n  "runs": [\n'+rows+'\n  ]\n}\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--language',type=Path,default=ROOT/'results/function-bridge-deep/language-lora731.json')
    p.add_argument('--embedding-model',default='/Users/a405394/models/google_embeddinggemma-300m')
    p.add_argument('--out',type=Path,default=ROOT/'results/function-bridge-deep/grounded.json')
    a=p.parse_args()
    from runtime import NoviBackend
    backend=NoviBackend(a.embedding_model,ROOT/'results/gemma-bridge')
    before=list(backend.engine.weights)
    language=json.loads(a.language.read_text())
    cases=[r for r in language['cases'] if r['expected'] is not None and r['expected'][0]=='set_goal']
    rows=[];latencies=[];dispatches=[]
    for case in cases:
        start=time.perf_counter()
        try: parsed,neural=dispatch(case['raw'],backend)
        except ValueError: parsed,neural=None,None
        dispatches.append({'id':case['id'],'parsed_goal':parsed,'neural_goal':neural,'seconds':time.perf_counter()-start})
        expected=case['expected'][1]['goal']
        for variant in VARIANTS:
            selected={'oracle_goal':expected,'parsed_direct':parsed,'novi_memory':neural,'novi_no_odometry':neural,'novi_swapped_decoder':None if neural is None else GOALS[(GOALS.index(neural)+1)%4]}[variant]
            for scenario in SCENARIOS:
                for seed in SEEDS:
                    result=simulate(expected,selected,seed,scenario,variant!='novi_no_odometry')
                    latencies.extend(result.pop('control_seconds'))
                    rows.append({'case_id':case['id'],'language':case['language'],'expected':expected,'selected':selected,'semantic_correct':selected==expected,'variant':variant,'scenario':scenario,'seed':seed,**result})
    assert before==list(backend.engine.weights),'evaluation changed policy weights'
    aggregate={}
    for scenario in SCENARIOS:
        aggregate[scenario]={}
        for variant in VARIANTS:
            chosen=[r for r in rows if r['scenario']==scenario and r['variant']==variant]
            conditional=[r for r in chosen if r['semantic_correct']]
            seed_rates=[np.mean([r['success'] for r in chosen if r['seed']==seed]) for seed in SEEDS]
            aggregate[scenario][variant]={'count':len(chosen),'unique_physics_configurations':len({(r['expected'],r['selected'],r['seed'],r['scenario'],r['variant']!='novi_no_odometry') for r in chosen}),'arrival_rate':float(np.mean([r['success'] for r in chosen])),'conditional_on_correct_goal':float(np.mean([r['success'] for r in conditional])) if conditional else None,'mean_terminal_reward':float(np.mean([r['reward'] for r in chosen])),'arrival_rate_by_seed':list(map(float,seed_rates))}
    report={'scope':'Frozen policy, recorded actual language generations -> strict parser -> native novi -> host map -> heading controller -> physical target contact. No online language/HTTP call per episode, no learning or obstacles.','language_report':str(a.language),'language_report_sha256':hashlib.sha256(a.language.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256((ROOT/'results/gemma-bridge'/backend.bundle['checkpoint']).read_bytes()).hexdigest(),'seeds':SEEDS,'steps':120,'policy_unchanged':True,'aggregate':aggregate,'dispatches':dispatches,'control_latency_seconds':{'p50':float(np.median(latencies)),'p95':float(np.percentile(latencies,95)),'samples':len(latencies),'scope':'Python heading fusion/steering only, excludes model load, language generation and simulated sensors/physics'},'runs':rows,'limitations':['exact host position and known resource map','no obstacles or learned steering','same language output reused across seeds; seeds are not independent language observations','cases sharing expected/selected goal and seed duplicate physics; counts are descriptive workload rows, not independent trials','oracle_goal has only 4 goals x 10 seeds = 40 unique physics configurations per scenario','reward intentionally ranks abstention/timeout (0) above wrong resource contact (-1)','exogenous noise is coupled by tick, but closed-loop sensor values diverge between variants','novi semantic decoder is redundant after correct enum; direct baseline tests its cost','reward measures target contact but policy remains frozen']}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    write_report(a.out,report)
    print(json.dumps({'aggregate':aggregate,'control_latency_seconds':report['control_latency_seconds']},indent=2))

if __name__=='__main__':main()
