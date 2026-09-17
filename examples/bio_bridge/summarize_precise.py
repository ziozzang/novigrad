"""Summarize fixed, centered and PCA interfaces without selecting on final data."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/precise-bridge'

def main():
    semantic=json.loads((OUT/'final.json').read_text())['results']
    native=json.loads((OUT/'native-final.json').read_text())
    permutations=json.loads((OUT/'permutation-final.json').read_text())['results']
    rows=json.loads((ROOT/'examples/bio_bridge/precise_holdout.json').read_text())
    modes=['fixed','centered','pca'];summary={}
    for mode in modes:
        summary[mode]={'semantic_accuracy':semantic[mode]['all']['semantic_category_accuracy'],
                       'native_accuracy':native['aggregate'][mode]['accuracy'],
                       'native_mean_correct_probability':native['aggregate'][mode]['mean_correct_probability'],
                       'semantic_language_accuracy':{k:v['semantic_category_accuracy'] for k,v in semantic[mode]['languages'].items()}}
    # Resample whole authored families; preserves translated pairs and within-context dependence.
    family=np.array([r['family'] for r in rows]);families=sorted(set(family));differences=[]
    a=np.array([r['category']==r['expected_category'] for r in semantic['pca']['all']['cases']],float)
    b=np.array([r['category']==r['expected_category'] for r in semantic['fixed']['all']['cases']],float)
    differences=np.array([(a[family==f]-b[family==f]).mean() for f in families])
    bootstrap=np.random.default_rng(917).choice(differences,size=(10000,len(families)),replace=True).mean(1)
    report={'scope':'Authored64 translated32 scenarios in8families. Resampling interval is descriptive for this small authored sample, not population validity.',
            'modes':summary,'raw_semantic_accuracy':semantic['raw_embedding']['all']['semantic_category_accuracy'],
            'pca_minus_fixed_family_cluster_bootstrap':{'families':len(families),'mean_difference':float(differences.mean()),'percentile95':np.quantile(bootstrap,[.025,.975]).tolist(),'seed':917,'replicates':10000}}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained');x=np.arange(3)
    ax.bar(x-.18,[summary[m]['semantic_accuracy']*100 for m in modes],.36,label='Semantic reconstruction')
    ax.bar(x+.18,[summary[m]['native_accuracy']*100 for m in modes],.36,label='Native reward-trained action (3 seeds)')
    ax.axhline(report['raw_semantic_accuracy']*100,color='gray',linestyle='--',label='Raw embedding semantic baseline')
    ax.set_xticks(x,['Fixed MRL128','Train-centered MRL128','Train PCA (rank31)']);ax.set_ylim(0,105);ax.set_ylabel('Final authored-case accuracy (%)');ax.set_title('Train-only calibration of the embedding-to-fly interface');ax.legend(loc='lower right',fontsize=8)
    fig.savefig(OUT/'comparison.png',dpi=160);fig.savefig(OUT/'comparison.svg');plt.close(fig)
    p=OUT/'comparison.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
