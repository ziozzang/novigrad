"""Build compact research tables and standalone plots from retained raw runs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/function-bridge-deep'

def read(name):return json.loads((OUT/name).read_text())

def main():
    names=['base','lora731','lora732','lora733']
    language={n:read(f'language-{n}.json') for n in names}
    reliability=read('reliability.json')
    short,long=read('internal-state.json'),read('internal-state-long.json')
    langsummary={n:r['groups'] for n,r in language.items()}
    mean_adapted=float(np.mean([language[n]['groups']['valid']['accuracy'] for n in names[1:]]))
    # Resample language cases, not 3*64 observations. Training-seed variability is separate.
    base=np.array([r['correct'] for r in language['base']['cases'] if r['expected'] is not None],float)
    adapted=np.mean([[r['correct'] for r in language[n]['cases'] if r['expected'] is not None] for n in names[1:]],axis=0)
    delta=adapted-base
    rng=np.random.default_rng(5701)
    means=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
    summary={'language':langsummary,'adapted_mean_valid_accuracy':mean_adapted,'adapted_mean_minus_base_valid_accuracy':{'difference':float(delta.mean()),'case_bootstrap_ci95':list(map(float,np.quantile(means,[.025,.975]))),'bootstrap_scope':'Case resampling of this authored set; three training runs averaged within each case, not 192 independent cases; not evidence of deployment population coverage.'},'grounded':{n:read(n+'.json')['aggregate'] for n in ['grounded-base','grounded']},'reliability':reliability['aggregate'],'internal_state':{'short':short['mean'],'long':long['mean']},'ensemble':read('ensemble.json')['summary'],'credit_assignment':read('credit-assignment.json')['mean'],'state_probe':read('state-probe.json')['mean']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    x=np.arange(4); width=.36
    axes[0].bar(x-width/2,[language[n]['groups']['valid']['accuracy']*100 for n in names],width,label='Valid exact calls')
    axes[0].bar(x+width/2,[language[n]['groups']['unsupported']['accuracy']*100 for n in names],width,label='Unsupported rejected')
    axes[0].set_xticks(x,['Base','LoRA\n731','LoRA\n732','LoRA\n733']);axes[0].set_ylim(0,100)
    axes[0].set_ylabel('Percent');axes[0].set_title('Task adaptation / over-calling tradeoff');axes[0].legend(fontsize=8)
    scenarios=['clean','abrupt_visual_bias','gradual_visual_drift','simultaneous_biases'];x=np.arange(4)
    for shift,method,label in [(-width/2,'fixed_trust','Fixed trust'),(width/2,'innovation_gate','Innovation gate')]:
        axes[1].bar(x+shift,[reliability['aggregate'][s][method]['mae_degrees']['mean'] for s in scenarios],width,label=label)
    axes[1].set_xticks(x,['Clean','Abrupt\nbias','Gradual\ndrift','Both\nbiased']);axes[1].set_ylabel('Heading MAE (degrees)');axes[1].set_title('No universal sensor-filter winner');axes[1].legend(fontsize=8)
    policies=['state_conditioned','no_hunger','linear','zero_reward'];x=np.arange(4)
    axes[2].bar(x-width/2,[short['mean'][p]['utility'] for p in policies],width,label='2,048 interactions')
    axes[2].bar(x+width/2,[long['mean'][p]['utility'] for p in policies],width,label='8,192 interactions')
    best=short['mean']['state_conditioned']['best_possible_utility']
    axes[2].axhline(best,color='black',ls='--',lw=1,label='Full-information optimum')
    axes[2].set_xticks(x,['novi +\nstate','novi -\nhunger','Raw\nlinear','No\nreward']);axes[2].set_ylabel('Mean host utility');axes[2].set_title('Reward learning and input ablation');axes[2].legend(fontsize=8)
    fig.suptitle('Novigrad: measured mechanisms and their failure modes — 2026-09-17')
    fig.savefig(OUT/'mechanisms.png',dpi=170);fig.savefig(OUT/'mechanisms.svg');plt.close(fig)
    credit=read('credit-assignment.json')
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for method,label in [('original_reinforce','Raw linear'),('centered_scaled_reinforce','Centered linear'),('actor_critic','Centered actor-critic'),('actor_critic_entropy','Actor-critic + entropy')]:
        curves=[r['learning_curves'][method] for r in credit['runs']]
        x=[v['interaction'] for v in curves[0]]
        y=np.array([[v['utility'] for v in c] for c in curves])
        ax.plot(x,y.mean(axis=0),marker='o',label=label)
    curves=[r['learning_curves']['state_conditioned'] for r in long['runs']]
    ax.plot([v['step'] for v in curves[0]],np.mean([[v['utility'] for v in c] for c in curves],axis=0),marker='s',label='novi + internal state')
    ax.axhline(best,color='black',ls='--',lw=1,label='Full-information optimum')
    ax.set_xlabel('Sampled training interactions (batch 32)');ax.set_ylabel('Frozen greedy utility on reused test contexts')
    ax.set_title('Input conditioning explains much of the failed linear baseline')
    ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.savefig(OUT/'credit-assignment.png',dpi=170);fig.savefig(OUT/'credit-assignment.svg');plt.close(fig)
    for svg in OUT.glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps({'mean_adapted':mean_adapted,'paired_delta':summary['adapted_mean_minus_base_valid_accuracy']},indent=2))

if __name__=='__main__':main()
