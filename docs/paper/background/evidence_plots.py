"""Plot ATFM's own measured retrieval-path medians (not analytical illustrations).

Numbers come from measured-evidence.json, transcribed from the dated stage C
report named there. plots.py remains the source of illustrative figures only.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'axes.labelcolor':'#263748'})
SERIES=[('cold','Cold prefill (recompute)','#b16626','-','o'),
        ('l2_on_demand','L2 load on demand','#245b85','--','s'),
        ('l1_warmed','L1 warmed ahead','#32714b','-','^'),
        ('warm','Warm lead time (off request path)','#6b6b6b',':','D')]


def main():
    data=json.loads((HERE/'measured-evidence.json').read_text())
    tokens=data['context_tokens']
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.3),sharey=True)
    for ax,(gpu,values) in zip(axes,data['gpus'].items()):
        for key,label,col,ls,mk in SERIES:
            ax.plot(tokens,values[key],color=col,ls=ls,marker=mk,ms=5,lw=1.8,label=label)
        ax.set_xscale('log',base=2);ax.set_yscale('log')
        ax.set_xticks(tokens,[f'{t:,}' for t in tokens])
        ax.minorticks_off()
        ax.set_yticks([0.1,0.3,1,3,10,30],['0.1','0.3','1','3','10','30'])
        ax.set(title=gpu,xlabel='Context length (tokens)')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Median time (s, log scale)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=9,bbox_to_anchor=(0.5,-0.13))
    fig.tight_layout()
    fig.savefig(HERE/'figures'/'retrieval-measured.pdf',bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
