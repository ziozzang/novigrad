#!/usr/bin/env python3
"""Plot recorded validation curves only; requires requirements-reports.txt."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/'results/improvement'

def main():
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'novigrad'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    experiments=[('baseline','Original normalization', '#9a5967'),
                 ('no-homeostasis','No L1 normalization, 25 epochs','#d3932d'),
                 ('longer75','No L1 normalization, up to 75 epochs','#2875b5'),
                 ('active20','20% active hidden neurons, up to 75 epochs','#21866b')]
    for ax,task,title in zip(axes,['fashion','captcha'],['Clothing classification','Synthetic CAPTCHA characters']):
        for directory,label,color in experiments:
            with (ROOT/directory/task/'curves.csv').open() as f:
                rows=[r for r in csv.DictReader(f) if int(r['epoch'])>0]
            if directory == 'no-homeostasis':
                ax.scatter([int(rows[-1]['epoch'])],[100*float(rows[-1]['validation_accuracy'])],label='25-epoch checkpoint (same blue curve)',color=color,s=40,zorder=5,edgecolor='white')
            else:
                ax.plot([int(r['epoch']) for r in rows],[100*float(r['validation_accuracy']) for r in rows],label=label,color=color,linewidth=1.6)
        ax.set(title=title,xlabel='Completed training epochs',ylabel='Validation accuracy (%)')
        ax.grid(alpha=.2)
        ax.set_ylim((70,91) if task=='fashion' else (78,100))
        ax.spines[['top','right']].set_visible(False)
    axes[1].legend(loc='lower right',fontsize=8,frameon=False)
    fig.suptitle('Same biological edge topology; different engineered learning settings',fontsize=12)
    fig.savefig(ROOT/'learning-curves.svg',metadata={'Date':None})
    svg=ROOT/'learning-curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(ROOT/'learning-curves.png',dpi=160,metadata={'Software':'Novigrad / matplotlib'})
    plt.close(fig)

if __name__=='__main__':main()
