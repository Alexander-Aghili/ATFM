"""Generate analytical illustrations, not measured hardware or ATFM results."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'axes.labelcolor':'#263748'})

def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.png',dpi=160,bbox_inches='tight')
    plt.close(fig)


def main():
    t=np.linspace(1024,32768,128)
    fig,ax=plt.subplots(figsize=(7.2,3.5))
    for heads,label,col in [(32,'MHA: 32 KV heads','#245b85'),(8,'GQA: 8 KV heads','#b16626'),(1,'MQA: 1 KV head','#32714b')]:
        ax.plot(t/1024,2*32*heads*128*2*t/2**30,label=label,color=col,lw=2)
    ax.set(xlabel='Context length (Ki tokens)',ylabel='KV memory per sequence (GiB)',title='Illustrative 32-layer model, head dimension 128, 2 bytes/element')
    ax.legend(frameon=False);ax.grid(alpha=.15);save(fig,'kv-growth')
    size=np.linspace(.05,8,160)
    fig,ax=plt.subplots(figsize=(7.2,3.5))
    for bw,col in [(8,'#b16626'),(25,'#245b85'),(100,'#32714b')]:
        ax.plot(size,1000*(.001+size/bw),label=f'{bw} GiB/s effective bandwidth',color=col)
    ax.axhline(150,color='#555',ls='--',label='Assumed 150 ms recomputation cost')
    ax.set(xlabel='Cache payload (GiB)',ylabel='Restore time (ms)',title='Illustrative transfer model: 1 ms fixed overhead, no queueing')
    ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.15);save(fig,'transfer')
    rho=np.linspace(.05,.97,150)
    fig,ax=plt.subplots(figsize=(7.2,3.5))
    ax.plot(rho,1/(1-rho),color='#245b85',lw=2)
    ax.set(xlabel='Utilization (arrival rate / service rate)',ylabel='Mean response / mean service',title='M/M/1 illustration; not a fit to a GPU serving engine',ylim=(0,35))
    ax.grid(alpha=.15);save(fig,'queueing')
    x=np.linspace(0,60,500)
    from math import erf,sqrt
    cdf=np.array([.5*(1+erf((u-20)/(5*sqrt(2)))) for u in x])
    cdf0=.5*(1+erf(-20/(5*sqrt(2))))
    cdf=(cdf-cdf0)/(1-cdf0)
    dt=x[1]-x[0]
    occupation=np.r_[0,np.cumsum((2-cdf[1:]-cdf[:-1])*.5*dt)]
    benefit=.2*cdf
    fig,ax=plt.subplots(figsize=(7.2,3.5))
    for rate,col in [(.002,'#245b85'),(.008,'#b16626')]:
        ax.plot(x,1000*(benefit-rate*occupation),label=f'Opportunity cost {rate*1000:g} ms per second held',color=col,lw=2)
    ax.axhline(0,color='#555',lw=.8)
    ax.set(xlabel='Retention limit (s)',ylabel='Expected net latency value (ms)',title='Illustrative return time: positive-truncated N(20 s, 5 s); miss cost 200 ms')
    ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.15);save(fig,'retention')
    facts={'kind':'analytical illustrations, not measured results','kv':{'layers':32,'head_dim':128,'element_bytes':2,'kv_heads':[32,8,1]},'transfer':{'fixed_overhead_ms':1,'effective_GiB_s':[8,25,100],'assumed_recompute_ms':150},'retention':{'return_distribution':'Normal(20,5) truncated below zero','miss_cost_s':.2,'opportunity_cost_s_per_s':[.002,.008]}}
    (HERE/'illustration-inputs.json').write_text(json.dumps(facts,indent=2)+'\n')

if __name__=='__main__':
    main()
