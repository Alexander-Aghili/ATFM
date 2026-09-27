"""Additional publication figures, using frozen CSVs or explicitly illustrative values."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'latex-fig'
QA=ROOT/'tmp/visuals'
COL={'B2':'#245b85','M1':'#b14b20','M2':'#35734d','rules':'#79659b'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
 'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
 'pdf.fonttype':42,'savefig.bbox':'tight','axes.axisbelow':True})
def save(fig,name):
    OUT.mkdir(exist_ok=True); QA.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/f'{name}.pdf'); fig.savefig(QA/f'{name}.png',dpi=160); plt.close(fig)
def paired(regime):
    return pd.read_csv(ROOT/f'results/h2sim_{regime}__paired.csv')
def row(df,arm,metric):
    return df[(df.arm==arm)&(df.metric==metric)].iloc[0]

def main():
    # Conceptual timing illustration; these are not observations.
    fig,ax=plt.subplots(figsize=(7,2.5),layout='constrained')
    phases=[[(0,8,'Model'),(8,44,'Tool'),(52,8,'Pending'),(60,12,'Model')],
            [(0,12,'Tool'),(12,6,'Pending'),(18,10,'Model'),(28,47,'Tool')],
            [(0,5,'Model'),(5,26,'Tool'),(31,5,'Pending'),(36,9,'Model'),(45,30,'Tool')]]
    colors={'Model':'#245b85','Tool':'#e0a56e','Pending':'#d2d8dd'}
    for y,parts in enumerate(phases):
        for start,duration,kind in parts: ax.broken_barh([(start,duration)],(y-.28,.56),facecolors=colors[kind],edgecolors='white')
    ax.axvline(15,color='black',ls='--',lw=1); ax.axvspan(15,45,color='#35734d',alpha=.09)
    ax.text(16,2.48,'Forecast at t = 15 s; horizon h = 30 s',fontsize=9)
    ax.set(yticks=[0,1,2],yticklabels=['Session A','Session B','Session C'],xlabel='Illustrative wall-clock time (s)',xlim=(0,78),ylim=(-.6,2.85))
    ax.legend(handles=[Patch(color=v,label=k) for k,v in colors.items()],ncol=3,loc='upper center',bbox_to_anchor=(.5,-.26),frameon=False)
    save(fig,'agent-timeline')

    fig,axs=plt.subplots(1,2,figsize=(7,2.8),layout='constrained')
    # Empirical support [10,20,40,80], observed surviving past 30 seconds.
    for ax,x,p,title in [(axs[0],[0,10,50],[.5,.25,.25],'B2: subtract elapsed time'),(axs[1],[10,50],[.5,.5],'M1: condition on survival')]:
        ax.vlines(x,0,p,color=COL['B2'] if ax is axs[0] else COL['M1'],lw=3)
        ax.scatter(x,p,color=COL['B2'] if ax is axs[0] else COL['M1'],zorder=3)
        ax.set(title=title,xlabel='Remaining tool time (s)',xlim=(-5,60),ylim=(0,.62),yticks=[0,.25,.5]); ax.grid(axis='y',alpha=.25)
    axs[0].set_ylabel('Probability mass'); save(fig,'survival-example')

    fig,axs=plt.subplots(1,2,figsize=(7,2.8),layout='constrained')
    p=np.linspace(.02,1,100)
    axs[0].plot(p,p,label='Linear: g(p) = p',color=COL['B2'])
    axs[0].plot(p,np.sqrt(p),label='Illustrative: g(p) = √p',color=COL['M2'],ls='--')
    axs[0].set(xlabel='Reported work fraction p',ylabel='Elapsed-time fraction g(p)',xlim=(0,1),ylim=(0,1)); axs[0].legend(fontsize=8,frameon=False)
    axs[1].plot(p,30*(1/p-1),color=COL['B2']); axs[1].plot(p,30*(1/np.sqrt(p)-1),color=COL['M2'],ls='--')
    axs[1].scatter([.25,.25],[90,30],color=[COL['B2'],COL['M2']]); axs[1].annotate('90 s',(.25,90),xytext=(.31,100)); axs[1].annotate('30 s',(.25,30),xytext=(.34,40))
    axs[1].set(xlabel='Reported work fraction p',ylabel='Estimated time remaining (s)',xlim=(0,1),ylim=(0,180),title='At elapsed time e = 30 s')
    for ax in axs: ax.grid(alpha=.2)
    save(fig,'progress-concept')

    fig,axs=plt.subplots(1,2,figsize=(7,2.9),layout='constrained')
    for suffix,label,style in [('', 'Uncalibrated','-'),('_cal','Dispersion calibrated','--')]:
        d=pd.read_csv(ROOT/f'results/h1_tracelab_r200{suffix}__metrics.csv')
        d=d[(d['class']=='interactive')&(d.target=='kv_blocks')&(d.model==('M1_survival+cal' if suffix else 'M1_survival'))].sort_values('h')
        assert len(d)==5, d
        axs[0].plot(d.h,100*d.cov90,style,marker='o',label=label)
        axs[1].plot(d.h,d.pinball90/1000,style,marker='o',label=label)
    axs[0].axhline(90,color='#444',ls=':',label='Nominal 90%'); axs[0].set(ylabel='Empirical interval coverage (%)',ylim=(0,100))
    axs[1].set_ylabel('q90 pinball loss (thousand KV blocks)')
    for ax in axs: ax.set_xscale('log'); ax.set_xticks([10,30,120,300,900],['10','30','120','300','900']); ax.set_xlabel('Forecast horizon (s)'); ax.grid(alpha=.2)
    axs[0].legend(fontsize=7.5,frameon=False,loc='lower right'); save(fig,'calibration')

    fig,axs=plt.subplots(1,2,figsize=(7,3.7),layout='constrained')
    regimes=['short_tool','long_tool','long_tool_loaded','interactive_long','long_tool_loaded_cap60']
    names=['Short tools\n600 s cap','Long tools\n600 s cap','Loaded long tools\n600 s cap','Long interactive\n60 s cap','Loaded long tools\n60 s cap']
    for k,(arm,label,color) in enumerate([('proxy_rules','Rules',COL['rules']),('forecast_M1','M1',COL['M1']),('forecast_M2','M2',COL['M2'])]):
        for ax,metric,scale in [(axs[0],'slo_attainment_sessions',100),(axs[1],'bg_jct_mean',1)]:
            rows=[row(paired(reg),arm,metric) for reg in regimes]
            x=np.array([r.diff_mean for r in rows])*scale; lo=np.array([r.diff_ci_lo for r in rows])*scale; hi=np.array([r.diff_ci_hi for r in rows])*scale
            ax.errorbar(x,np.arange(5)+(k-1)*.20,xerr=[x-lo,hi-x],fmt=['s','^','o'][k],ms=4,color=color,label=label,capsize=2,lw=1)
    for ax in axs: ax.axvline(0,color='#555',ls=':',lw=1); ax.set_yticks(range(5)); ax.invert_yaxis(); ax.grid(axis='x',alpha=.2)
    axs[0].set_yticklabels(names); axs[1].set_yticklabels([])
    axs[0].set_xlabel('SLO difference from native (pp)'); axs[1].set_xlabel('Background JCT difference (s)'); axs[1].legend(frameon=False,fontsize=8,loc='lower right')
    save(fig,'regime-effects')

    fig,axs=plt.subplots(1,2,figsize=(7,3),layout='constrained')
    for ax,metric,scale in [(axs[0],'slo_attainment_sessions',100),(axs[1],'bg_jct_mean',1)]:
        for j,(arm,label) in enumerate([('proxy_rules','Rules'),('forecast_M1','M1'),('forecast_M2','M2'),('working_set','Working set')]):
            a=row(paired('long_tool_loaded_cap60'),arm,metric)['mean']*scale
            b=row(paired('long_tool_loaded'),arm,metric)['mean']*scale
            ax.plot([a,b],[j,j],color='#a0a0a0',lw=2)
            ax.scatter(a,j,color='#245b85',marker='s',label='60 s cap' if j==0 else None,zorder=3)
            ax.scatter(b,j,color='#b14b20',marker='o',label='600 s cap' if j==0 else None,zorder=3)
        native=row(paired('long_tool_loaded'),'native',metric)['mean']*scale
        ax.axvline(native,color='#444',ls=':',label='Native'); ax.set_yticks(range(4)); ax.invert_yaxis(); ax.grid(axis='x',alpha=.2)
    axs[0].set_yticklabels(['Rules','M1','M2','Working set']); axs[1].set_yticklabels([])
    axs[0].set_xlabel('Session-weighted SLO (%)'); axs[1].set_xlabel('Mean background JCT (s)'); axs[1].legend(loc='lower right',fontsize=7,frameon=False)
    save(fig,'cap-ablation')

    fig,axs=plt.subplots(1,3,figsize=(7,3.5),layout='constrained')
    d=pd.read_csv(ROOT/'results/h2sim_long_tool_loaded__metrics.csv')
    arms=['native','proxy_rules','forecast_M1','forecast_M2','forecast_M2_nohold','oracle','working_set']
    labels=['Native','Rules','M1','M2','M2, no holds','Heap lookahead','Working set']
    for ax,(metric,scale,label) in zip(axs,[('recomputed_prefill_tokens',1e6,'Recomputed prefill\n(million tokens)'),('hold_kv_block_s',1e6,'Held KV\n(million block-seconds)'),('caps',1,'Holds reaching cap\n(count)')]):
        for j,arm in enumerate(arms):
            vals=d[d.arm==arm].sort_values('seed')[metric].to_numpy()/scale
            ax.scatter(vals,j+np.linspace(-.1,.1,len(vals)),s=18,facecolors='none',edgecolors='#245b85')
            ax.scatter(vals.mean(),j,marker='|',s=150,color='#222')
        ax.set_yticks(range(7)); ax.set_yticklabels(labels if ax is axs[0] else []); ax.invert_yaxis(); ax.set_xlabel(label); ax.grid(axis='x',alpha=.2)
    save(fig,'resource-costs')

if __name__=='__main__': main()
