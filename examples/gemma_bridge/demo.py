"""Run an actual language -> learned actuator decision with a saved bundle."""
import argparse,json,time
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from novigrad import Engine
from bridge import SemanticPorts
from data import CLASSES

p=argparse.ArgumentParser()
p.add_argument('--model',type=Path,required=True)
p.add_argument('--results',type=Path,default=Path('results/gemma-bridge'))
p.add_argument('--seed',type=int,default=101)
p.add_argument('text')
a=p.parse_args()
report=json.loads((a.results/'experiment.json').read_text())
run=next(r for r in report['runs'] if r['seed']==a.seed and r['kind']=='novi' and r['condition']=='reward')
engine=Engine.load(a.results/f'novi-{a.seed}.safetensors')
model=SentenceTransformer(str(a.model),device='mps',local_files_only=True)
start=time.perf_counter()
embedding=model.encode([a.text],prompt_name='Classification',normalize_embeddings=True)
rates=SemanticPorts(len(engine.input_ids)).encode(embedding)
probabilities=engine.infer(rates[0].tolist()); action=int(np.argmax(probabilities))
meaning=CLASSES[run['mapping'].index(action)]
print(json.dumps({'text':a.text,'actuator':action,'action_meaning':meaning,'probabilities':probabilities,
                  'seconds_excluding_load':time.perf_counter()-start},ensure_ascii=False,indent=2))
