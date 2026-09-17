"""Semantic readout and independent class-probe intervention summaries."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/thought-bridge'


def main():
    decoding=json.loads((OUT/'decoding.json').read_text())
    interventions=json.loads((OUT/'interventions.json').read_text())
    variants=decoding['variants'];splits=['test','korean','reused_confirmation']
    summary={'scope':'Input-semantic reconstruction, not subjective thoughts. Independent class probe interventions are distinct from the embedding decoder.',
        'decoding':{v:{s:{k:r[k] for k in ['semantic_category_accuracy','mean_cosine_to_input_embedding','catalog_identity_retrieval_accuracy']} for s,r in rows.items()} for v,rows in variants.items()},
        'posthoc_mbon_control':decoding['posthoc_mbon_control'], 'untrained_control':decoding['untrained_control'], 'interventions':interventions['aggregate']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    ax=axs[0];x=np.arange(3);w=.16
    for offset,name,label in [(-2,'raw_embedding','Original embedding'),(-1,'aligned','KC to embedding'),(0,'mbon_aligned','MBON to embedding'),(1,'mean_embedding','Mean-vector control')]:
        ax.bar(x+offset*w,[variants[name][s]['semantic_category_accuracy']*100 for s in splits],w,label=label)
    shuffled=[n for n in variants if n.startswith('shuffled_')]
    ax.bar(x+2*w,[np.mean([variants[n][s]['semantic_category_accuracy'] for n in shuffled])*100 for s in splits],w,label='Shuffled alignment (mean of 5)')
    ax.set_xticks(x,['English (24)','Korean (12)','Reused cases (32)']);ax.set_ylim(0,100);ax.set_ylabel('Semantic category accuracy (%)');ax.set_title('Embedding reconstruction loses information');ax.legend(fontsize=8)
    ax=axs[1];names=['top_contribution_25pct','random_active_25pct','random_l1matched25'];x=np.arange(3);w=.35
    for offset,metric,label in [(-w/2,'decoder_flip','Independent class-probe flip'),(w/2,'native_policy_flip','Native action flip')]:
        ax.bar(x+offset,[interventions['aggregate'][n][metric]*100 for n in names],w,label=label)
    ax.set_xticks(x,['Top 26 active KCs','Random 26 KCs','Random, same\nremoved L1']);ax.set_ylim(0,100);ax.set_ylabel('Changed decisions (%)');ax.set_title('Causal silencing in the engineered model');ax.legend(fontsize=8)
    fig.suptitle('Embedding-based state readout: decoding and causal use are separate',fontsize=14)
    fig.savefig(OUT/'semantic-readout.png',dpi=160);fig.savefig(OUT/'semantic-readout.svg');plt.close(fig)
    p=OUT/'semantic-readout.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    print('Wrote summary.json and semantic-readout plots')


if __name__=='__main__':main()
