"""Descriptive summaries of paired signal diagnostics; no independent-trial CIs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/signal-bridge"


def main():
    temporal = json.loads((OUT/'temporal.json').read_text())
    strength = json.loads((OUT/'strength.json').read_text())
    training = json.loads((OUT/'training-strength.json').read_text())
    summary = {"scope": "Descriptive means over three paired training runs, reused cases; no independent-trial confidence intervals", "temporal": []}
    for schedule in ['continuous','one_shot','periodic4']:
        for dose in ['per_pulse','equal_total']:
            for background in [0.,.25,1.]:
                for mode in temporal['design']['modes']:
                    rows = [r for r in temporal['runs'] if (r['schedule'],r['dose'],r['background'],r['mode']) == (schedule,dose,background,mode)]
                    row = dict(schedule=schedule,dose=dose,background=background,mode=mode)
                    for metric in ['all_ticks','cue_on_ticks','cue_off_ticks','post_initial_cue_ticks','first_phase','second_phase']:
                        row[metric] = None if rows[0][metric] is None else {k:float(np.mean([r[metric][k] for r in rows])) for k in rows[0][metric]}
                    row['accuracy_by_tick'] = np.mean([r['accuracy_by_tick'] for r in rows],axis=0).tolist()
                    summary['temporal'].append(row)
    summary['global_positive_gain_max_probability_change'] = strength['global_positive_max_probability_change']
    summary['training_strength'] = training['strength_mean']
    summary['one_shot_training'] = training['one_shot_mean']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axs = plt.subplots(2,3,figsize=(16,9),layout='constrained')
    modes=['direct','hold','ttl8','leaky4','leaky16','leaky16_cutoff']
    ax=axs[0,0]; xx=np.arange(len(modes)); width=.35
    for offset,bg in [(-width/2,0.),(width/2,.25)]:
        vals=[next(r['cue_off_ticks']['accuracy'] for r in summary['temporal'] if (r['schedule'],r['dose'],r['background'],r['mode'])==('one_shot','per_pulse',bg,m))*100 for m in modes]
        ax.bar(xx+offset,vals,width,label=f'Background {bg:g}')
    ax.set_xticks(xx,['Direct','Hold','TTL8','Leak4','Leak16','Leak16\n+cutoff']);ax.set_ylim(0,100);ax.set_ylabel('Cue-off accuracy (%)');ax.set_title('One-shot input: host memory / contamination');ax.legend(fontsize=8)
    ax=axs[0,1]
    for mode in ['direct','hold','leaky4','leaky16']:
        r=next(r for r in summary['temporal'] if (r['schedule'],r['dose'],r['background'],r['mode'])==('continuous','per_pulse',0.,mode))
        ax.plot(np.arange(24),np.array(r['accuracy_by_tick'])*100,label=mode)
    ax.axvline(12,color='black',linestyle=':',label='New need');ax.set_ylim(0,100);ax.set_xlabel('Logical tick');ax.set_ylabel('Accuracy (%)');ax.set_title('Continuous cue: old-state interference');ax.legend(fontsize=8)
    ax=axs[0,2]
    for mode in ['leaky16','leaky16_cutoff','ttl8']:
        rows=[r for r in strength['decay_silence'] if r['mode']==mode and r['gain']==1]
        ax.plot(rows[0]['delays'],np.mean([r['accuracy'] for r in rows],axis=0)*100,marker='o',label=mode)
    ax.set_ylim(0,100);ax.set_xlabel('Ticks after single cue');ax.set_ylabel('Accuracy (%)');ax.set_title('Scalar decay is normalized away');ax.legend(fontsize=8)
    ax=axs[1,0]
    gains=sorted({r['gain'] for r in strength['global_gain'] if r['gain']>0})
    ax.semilogx(gains,[np.mean([r['accuracy'] for r in strength['global_gain'] if r['gain']==g])*100 for g in gains],marker='o')
    ax.set_ylim(0,100);ax.set_xlabel('Common input multiplier');ax.set_ylabel('Accuracy (%)');ax.set_title('Positive global gain has no useful effect')
    ax=axs[1,1]
    ratios=sorted({r['distractor_ratio'] for r in strength['relative_mixture']})
    ax.plot(ratios,[np.mean([r['accuracy'] for r in strength['relative_mixture'] if r['distractor_ratio']==g])*100 for g in ratios],marker='o')
    ax.set_ylim(0,100);ax.set_xlabel('Distractor / target amplitude');ax.set_ylabel('Target accuracy (%)');ax.set_title('Relative signal direction does matter')
    ax=axs[1,2]
    for key,label in [('accuracy','Greedy accuracy'),('mean_correct_probability','Correct-action probability')]:
        ax.plot([0,.1,1,4],[training['strength_mean'][f'reward-{g:g}']['test'][key]*100 for g in [0,.1,1,4]],marker='o',label=label)
    ax.set_ylim(0,100);ax.set_xlabel('Reward multiplier, 4096 updates');ax.set_ylabel('Percent');ax.set_title('Reward strength: accuracy is non-monotonic');ax.legend(fontsize=8)
    fig.suptitle('Novigrad signal diagnostics — engineered state, not biological recall',fontsize=15)
    fig.savefig(OUT/'signals.png',dpi=160);fig.savefig(OUT/'signals.svg');plt.close(fig)
    p=OUT/'signals.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    print('Wrote summary.json, signals.png, signals.svg')


if __name__ == '__main__':
    main()
