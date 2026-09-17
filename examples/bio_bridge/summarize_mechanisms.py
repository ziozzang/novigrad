"""Descriptive mechanism comparisons; reused examples, no significance claims."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/mechanism-bridge'


def main():
    adaptation=json.loads((OUT/'adaptation.json').read_text())
    inhibition=json.loads((OUT/'inhibition.json').read_text())
    eligibility=json.loads((OUT/'eligibility.json').read_text())
    summary={'scope':'Descriptive mechanism contrasts; reused datasets; independent module tasks are not directly comparable accuracy benchmarks', 'adaptation':[]}
    for g in [.25,1.,4.]:
        for schedule in ['sustained8','pulse1']:
            for drift in [False,True]:
                for mode in adaptation['design']['modes']:
                    rows=[r for r in adaptation['runs'] if (r['background_strength'],r['schedule'],r['drift'],r['mode'])==(g,schedule,drift,mode)]
                    summary['adaptation'].append(dict(background_strength=g,schedule=schedule,drift=drift,mode=mode,
                        foreground_accuracy=float(np.mean([r['foreground']['accuracy'] for r in rows])),
                        foreground_correct_probability=float(np.mean([r['foreground']['mean_correct_probability'] for r in rows])),
                        accuracy_by_seed=[r['foreground']['accuracy'] for r in rows]))
    summary['inhibition_frozen']=inhibition['aggregate']['reused_confirmation_32']
    summary['inhibition_matched_supervised_probe']=inhibition['posthoc_matched_readout']
    summary['eligibility']=eligibility['mean_across_seeds']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for ax,drift in [(axs[0,0],False),(axs[0,1],True)]:
        for mode,label in [('direct','Direct'),('divisive_ema','Channel divisive'),('subtractive_ema','Channel subtraction'),('oracle_background','Known-background oracle')]:
            rows=[r for r in summary['adaptation'] if r['schedule']=='sustained8' and r['drift']==drift and r['mode']==mode]
            ax.plot([r['background_strength'] for r in rows],[r['foreground_accuracy']*100 for r in rows],marker='o',label=label)
        ax.set_ylim(0,100);ax.set_xlabel('Background / foreground amplitude');ax.set_ylabel('Foreground accuracy (%)')
        ax.set_title('Adaptation: '+('changing background' if drift else 'static background'));ax.legend(fontsize=8)
    ax=axs[1,0];variants=['native_topk_2','topk_20','subtractive_global','subtractive_local'];xx=np.arange(4);w=.36
    frozen=[summary['inhibition_frozen']['0.0'][v]['accuracy']['mean']*100 for v in variants]
    probe=[inhibition['posthoc_matched_readout']['variants'][v]['mixtures']['0.0']['accuracy']*100 for v in variants]
    ax.bar(xx-w/2,frozen,w,label='Frozen native readout (3 runs)');ax.bar(xx+w/2,probe,w,label='Matched supervised ridge (1 encoder)')
    ax.set_xticks(xx,['Top-k 2%','Top-k 20%','Global\nsubtraction','Local\nsubtraction']);ax.set_ylim(0,100);ax.set_ylabel('Clean confirmation accuracy (%)');ax.set_title('Inhibition ranking depends on readout');ax.legend(fontsize=8)
    ax=axs[1,1]
    for name,label in [('current_only','Current event'),('exact_cached_gradient','Cached-event oracle'),('trace_tau8','Accumulating trace tau8'),('trace_tau8_normmatched','Trace tau8, oracle norm matched')]:
        ax.plot([0,4,16],[summary['eligibility'][str(d)][name]['test']['accuracy']*100 for d in [0,4,16]],marker='o',label=label)
    ax.set_ylim(0,100);ax.set_xlabel('Reward delay (logical steps)');ax.set_ylabel('Test accuracy (%)');ax.set_title('Linear reference: trace credit interference');ax.legend(fontsize=8)
    fig.suptitle('Fly-inspired mechanism extraction: matched controls and limits',fontsize=15)
    fig.savefig(OUT/'mechanism-comparison.png',dpi=160);fig.savefig(OUT/'mechanism-comparison.svg');plt.close(fig)
    p=OUT/'mechanism-comparison.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    print('Wrote summary and mechanism-comparison figures')


if __name__=='__main__':main()
