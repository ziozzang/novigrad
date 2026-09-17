"""Exploratory cross-model agreement; diagnostic only, not an API default."""
import json
from pathlib import Path
from function_scenarios import metrics
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/bio-bridge'


def gate(parsed, semantic_goal):
    if parsed is not None and parsed[0]=='set_goal' and parsed[1]['goal']!=semantic_goal:
        return None
    return parsed


def main():
    embedding=json.loads((OUT/'embedding.json').read_text())
    semantic={r['id']:r for r in embedding['function_semantic_predictions']}
    result={'scope':'Post-hoc reused function cases; freeze 768-dimension prototype margin calibrated on separate validation data. Filter only set_goal calls using latest user message, not historical context. Other operations pass original parser. No new generation, inference or production behavior change.','variants':{}}
    for name in ['base','policy','adapter','adapter-policy']:
        original=json.loads((OUT/f'function-{name}.json').read_text())
        rows=[]
        for c in original['cases']:
            pred=gate(c['parsed'],semantic[c['id']]['goal'])
            rows.append({**c,'original_parsed':c['parsed'],'parsed':pred,'embedding_goal':semantic[c['id']]['goal'],'correct':pred==c['expected']})
        result['variants'][name]={'before':original['overall'],'after':metrics(rows),'cases':rows}
    (OUT/'semantic-gate.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v['after'] for k,v in result['variants'].items()},indent=2))
if __name__=='__main__':main()
