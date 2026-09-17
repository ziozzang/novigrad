"""Descriptive seed summaries and static figures; no independent-trial CIs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/bio-bridge'

def read(name):return json.loads((OUT/(name+'.json')).read_text())

def main():
    functions={name:read('function-'+name)['overall'] for name in ['base','policy','adapter','adapter-policy']}
    embedding=read('embedding');context=read('context-memory');credit=read('delayed-credit');confirmation=read('sparsity-confirmation')
    sparse={}
    for d in [64,128]:
        sparse[str(d)]={}
        for f in [.02,.1,.2,.5]:
            selected=[r['known_accuracy'] for r in embedding['novi_runs'] if r['dimensions']==d and r['active_fraction']==f]
            sparse[str(d)][str(f)]={'mean':float(np.mean(selected)),'by_training_seed':selected}
    csummary={}
    for name,agg in context['aggregate'].items():
        csummary[name]={'A_after_A':agg['after_A']['context_A_identity']['test']['mean_accuracy'],'A_after_B':agg['after_B']['context_A_identity']['test']['mean_accuracy'],'B_after_B':agg['after_B']['context_B_rotate_plus_1']['test']['mean_accuracy'],'A_after_refresh':agg['after_A_refresh']['context_A_identity']['test']['mean_accuracy']}
    summary={'function':functions,'embedding_prototype':{k:{'threshold':v['threshold'],'forced':v['forced_known'],'calibrated':v['calibrated']} for k,v in embedding['prototype'].items()},'sparsity':sparse,'sparsity_confirmation':confirmation['summary'],'context':csummary,'batched_credit':credit['mean_across_seeds'],'amplitude_controls':credit['amplitude_control_mean_across_seeds'],'semantic_gate':{k:v['after'] for k,v in read('semantic-gate')['variants'].items()},'inference_scope':'Descriptive means. Shared authored items, related translations and fixedinitialization; no claim of independent biological replicates.'}
    if (OUT/'streaming-credit.json').exists():summary['streaming_credit']=read('streaming-credit')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axs=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    ax=axs[0,0];x=np.arange(4);w=.25
    for off,key,label in [(-w,'valid_accuracy','Valid exact call'),(0,'rejection_rate','Unsupported rejected'),(w,'balanced_accuracy','Balanced metric')]:
        ax.bar(x+off,[r[key]*100 for r in functions.values()],w,label=label)
    ax.set_xticks(x,['Base','Intent\npolicy','LoRA','LoRA +\npolicy']);ax.set_ylim(0,100);ax.set_ylabel('Percent');ax.set_title('FunctionGemma: positive-call gain, rejection loss');ax.legend(fontsize=9)
    ax=axs[0,1]
    for d,color in [('64','tab:orange'),('128','tab:blue')]:
        fractions=[.02,.1,.2,.5];means=[sparse[d][str(f)]['mean']*100 for f in fractions]
        ax.plot(np.array(fractions)*100,means,marker='o',color=color,label=d+' dimensions')
        for f in fractions:ax.scatter([f*100]*3,np.array(sparse[d][str(f)]['by_training_seed'])*100,color=color,alpha=.4,s=18)
    ax.set_xlabel('Top-k KC budget (%)');ax.set_ylabel('Known-class accuracy (%)');ax.set_title('Frozen embeddings / real connectome (3 seeds)');ax.legend()
    ax.text(.98,.98,'New 32-text confirmation, 128D:\n2%: 52.1% vs 20%: 37.5%\nPrototype reference: 81.3%',transform=ax.transAxes,ha='right',va='top',fontsize=9)
    ax=axs[1,0];names=['blind','concat','conjunctive','modular'];x=np.arange(4)
    for off,key,label in [(-w,'A_after_A','A after learning A'),(0,'A_after_B','A after learning B'),(w,'B_after_B','B after learning B')]:
        ax.bar(x+off,[csummary[n][key]*100 for n in names],w,label=label)
    ax.set_xticks(x,['Blind','Concat','Gated\nports','Two\nengines']);ax.set_ylim(0,100);ax.set_ylabel('Greedy accuracy (%)');ax.set_title('Context preservation ≠ high acquisition accuracy');ax.legend(fontsize=9)
    ax.text(.98,.97,'Unit-norm concat matched raw concat.\nTwo engines use twice the plastic weights.',transform=ax.transAxes,ha='right',va='top',fontsize=8)
    ax=axs[1,1]
    for cond,label in [('wrong_current_trial','Wrong current trial'),('exact_queued','Exact action ID'),('decayed_queued','Decayed action ID')]:
        ax.plot([0,4,16],[credit['mean_across_seeds'][str(d)][cond]['test']['accuracy']*100 for d in [0,4,16]],marker='o',label=label)
    ax.scatter([4,16],[credit['amplitude_control_mean_across_seeds'][str(d)]['test']['accuracy']*100 for d in [4,16]],marker='D',s=70,facecolors='none',edgecolors='black',label='Same scale, no delay')
    ax.set_ylim(0,100);ax.set_xlabel('Logical trial lag (32-action frozen rollouts)');ax.set_ylabel('Greedy accuracy (%)');ax.set_title('Batched credit: timing and reward-scale confounds');ax.legend(fontsize=9)
    fig.suptitle('Novigrad biological hypotheses: evidence and counterexamples — 2026-09-17',fontsize=14)
    fig.savefig(OUT/'biological-mechanisms.png',dpi=160);fig.savefig(OUT/'biological-mechanisms.svg');plt.close(fig)
    for svg in OUT.glob('*.svg'):svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps({'function_balanced':{k:v['balanced_accuracy'] for k,v in functions.items()},'sparsity_confirmation':confirmation['summary']},indent=2))
if __name__=='__main__':main()
