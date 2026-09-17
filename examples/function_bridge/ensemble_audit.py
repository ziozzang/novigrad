"""Exploratory MAGI-style vote on already generated calls; no extra inference.

Both majority and unanimity are reported. Evaluation reuse makes this a
post-hoc diagnostic, not an independently validated routing policy.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def vote(calls,unanimous=False):
    keys=[json.dumps(c,sort_keys=True) for c in calls]
    key,count=Counter(keys).most_common(1)[0]
    return json.loads(key) if count >= (len(calls) if unanimous else len(calls)//2+1) else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,default=Path('results/function-bridge-deep'))
    a=p.parse_args()
    reports=[json.loads((a.directory/f'language-lora{s}.json').read_text()) for s in [731,732,733]]
    assert all([c['id'] for c in r['cases']]==[c['id'] for c in reports[0]['cases']] for r in reports)
    rows=[]
    for cases in zip(*(r['cases'] for r in reports)):
        calls=[c['parsed'] for c in cases]
        row={'id':cases[0]['id'],'expected':cases[0]['expected'],'individual_calls':calls}
        row.update({kind:vote(calls,kind=='unanimous') for kind in ['majority','unanimous']})
        rows.append(row)
    summary={}
    for variant in ['majority','unanimous']:
        valid=[r for r in rows if r['expected'] is not None]
        unsupported=[r for r in rows if r['expected'] is None]
        summary[variant]={'valid_correct':sum(r[variant]==r['expected'] for r in valid),'valid_count':len(valid),'unsupported_rejected':sum(r[variant] is None for r in unsupported),'unsupported_count':len(unsupported),'valid_abstentions':sum(r[variant] is None for r in valid),'unsupported_state_mutation_calls':sum(r[variant] is not None and r[variant][0] in ('set_goal','observe_heading','choose_action') for r in unsupported)}
    agreement=float(np.mean([r['individual_calls'][0]==r['individual_calls'][1]==r['individual_calls'][2] for r in rows]))
    result={'scope':'Post-hoc reused holdout; three independently sampled LoRA training seeds, same base/data. Correlated errors expected. No new inference or production routing change.','all_three_agreement':agreement,'summary':summary,'cases':rows}
    (a.directory/'ensemble.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
