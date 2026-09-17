"""Post-hoc untrained final-set control; no fitting or selection."""
import json
import numpy as np
from safetensors.numpy import load_file
from precise_bridge import ROOT,OUT,PortBridge,MODES,GOALS
from precise_native import new_engine,evaluate
from precise_pipeline import verify,write
from inhibition_mechanism import sha256

def main():
    verify();rows=json.loads((ROOT/'examples/bio_bridge/precise_holdout.json').read_text())
    x=load_file(str(OUT/'holdout-embeddings.safetensors'))['embeddings']
    y=np.array([GOALS.index(r['class']) for r in rows]);langs=np.array([r['language'] for r in rows])
    result={m:evaluate(new_engine(),PortBridge.load(OUT/f'ports-{m}.safetensors',m).encode(x),y,langs) for m in MODES}
    report={'status':'Post hoc after final results; no training, tuning, or alteration to frozen final report. Initial native model is deterministic, not three independent runs.','results':result,'source_sha256':sha256(__file__)}
    write(OUT/'untrained-final-posthoc.json',report);print(json.dumps({m:r['accuracy'] for m,r in result.items()}))
if __name__=='__main__':main()
