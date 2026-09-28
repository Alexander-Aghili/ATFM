"""Deterministic illustrative probability figures and one frozen-data diagnostic."""
from math import lgamma,log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from build_visuals import save,ROOT,COL

def main():
    _hazard_residual()
    _rate_posterior()
    _dependence_risk()
    _demand_fan()
    _sampling_loss()
    _loss_matrix()


def _hazard_residual():
    fig,axs=plt.subplots(1,2,figsize=(7,2.9),layout='constrained')
    e=np.linspace(.5,120,240)
    for k,label,color,style in [(.6,'Decreasing hazard (k = 0.6)',COL['B2'],'-'),(1.,'Constant hazard (k = 1)',COL['M1'],'--'),(2.,'Increasing hazard (k = 2)',COL['M2'],':')]:
        axs[0].plot(e,k/60*(e/60)**(k-1),color=color,ls=style,label=label)
        med=60*((e/60)**k+log(2))**(1/k)-e
        axs[1].plot(e,med,color=color,ls=style)
    axs[0].set(xlabel='Elapsed tool time (s)',ylabel='Instantaneous completion hazard (1/s)')
    axs[1].set(xlabel='Elapsed tool time (s)',ylabel='Conditional median remaining time (s)')
    axs[0].legend(fontsize=7,frameon=False)
    for a in axs:a.grid(alpha=.2)
    save(fig,'hazard-residual')


def _rate_posterior():

    fig,axs=plt.subplots(1,2,figsize=(7,2.9),layout='constrained')
    v=np.linspace(.001,2,500); r=np.linspace(1,220,500);wleft=20
    for shape,rate,label,color,style in [(2,4,'Prior',COL['B2'],':'),(12,24,'10 units / 20 s',COL['M1'],'--'),(42,84,'40 units / 80 s',COL['M2'],'-')]:
        def pdf(x):return np.exp(shape*np.log(rate)-lgamma(shape)+(shape-1)*np.log(x)-rate*x)
        axs[0].plot(v,pdf(v),color=color,ls=style,label=label)
        axs[1].plot(r,pdf(wleft/r)*wleft/r**2,color=color,ls=style)
    axs[0].set(xlabel='Work rate v (units/s)',ylabel='Probability density (s/unit)')
    axs[1].set(xlabel='Time to finish 20 more units (s)',ylabel='Probability density (1/s)')
    axs[0].legend(fontsize=7,frameon=False)
    for a in axs:a.grid(alpha=.2)
    save(fig,'rate-posterior')


def _dependence_risk():

    fig,axs=plt.subplots(1,2,figsize=(7,2.8),layout='constrained')
    xs=np.array([0,100,200]); distributions=[([.25,.5,.25],'Independent',COL['B2']),([.5,0,.5],'Perfectly synchronized',COL['M1'])]
    for j,(pmf,label,color) in enumerate(distributions):
        axs[0].bar(xs+(j-.5)*20,pmf,width=19,label=label,color=color)
        cap=np.arange(0,221); risk=np.array([np.sum(np.array(pmf)[xs>c]) for c in cap])
        axs[1].step(cap,risk,where='post',color=color,ls=['-','--'][j],label=label)
    axs[0].set(xticks=xs,xlabel='Total first-return demand (blocks)',ylabel='Probability mass',ylim=(0,.6))
    axs[1].set(xlabel='Capacity threshold K (blocks)',ylabel='P(demand > K)',ylim=(0,.8));axs[0].legend(fontsize=7,frameon=False)
    for a in axs:a.grid(axis='y',alpha=.2)
    save(fig,'dependence-risk')


def _demand_fan():

    # Coherent hypothetical sample paths; analytic mean uses the same construction.
    h, total, sizes = _demand_samples()
    fig,axs=plt.subplots(1,2,figsize=(7,2.9),layout='constrained')
    q=np.quantile(total,[.05,.5,.95],axis=0)
    axs[0].fill_between(h,q[0],q[2],color='#dbe6ee',step='mid',label='Pointwise central 90%')
    axs[0].step(h,q[1],where='mid',color=COL['B2'],label='Median')
    axs[0].set(xlabel='Horizon h (s)',ylabel='First-return demand (blocks)');axs[0].legend(fontsize=7,frameon=False)
    expected=(sizes[:,None]*(1-np.exp(-h[None,:]/np.array([20,40,80])[:,None]))).sum(axis=0)
    axs[1].plot(h,expected,color=COL['M2'],label='Known sessions')
    axs[1].plot(h,100*.015*h,color=COL['M1'],ls='--',label='Future sessions')
    axs[1].plot(h,expected+100*.015*h,color=COL['B2'],ls=':',label='Total')
    axs[1].set(xlabel='Horizon h (s)',ylabel='Expected demand (blocks)');axs[1].legend(fontsize=7,frameon=False)
    for a in axs:a.grid(alpha=.2)
    save(fig,'demand-fan')


def _demand_samples():
    rng=np.random.default_rng(20260927);M=20000; h=np.arange(0,121,5)
    returns=rng.exponential([20,40,80],size=(M,3)); sizes=np.array([100,150,200])
    known=(returns[:,:,None]<=h).astype(float)*sizes[None,:,None]
    known=known.sum(axis=1)
    increments=rng.poisson(.015*5,size=(M,len(h)-1))
    future=np.column_stack([np.zeros(M),np.cumsum(increments,axis=1)])*100
    total=known+future
    return h, total, sizes


def _sampling_loss():

    fig,axs=plt.subplots(1,2,figsize=(7,2.8),layout='constrained')
    m=np.arange(32,1025);p=.1
    axs[0].plot(m,100*np.sqrt(p*(1-p)/m),color=COL['B2'])
    axs[0].scatter([128,256],100*np.sqrt(p*(1-p)/np.array([128,256])),color=COL['M1'])
    axs[0].set(xlabel='Independent Monte Carlo draws M',ylabel='Tail-probability standard error (pp)')
    axs[0].annotate('128', (128,100*np.sqrt(.09/128)),xytext=(210,3));axs[0].annotate('256',(256,100*np.sqrt(.09/256)),xytext=(410,2.25))
    err=np.linspace(-100,100,401)
    axs[1].plot(err,np.where(err>=0,.9*err,-.1*err),color=COL['M2'])
    axs[1].set(xlabel='Observation minus forecast quantile (blocks)',ylabel='q90 pinball loss (blocks)');axs[1].text(-95,60,'Overprediction\npenalty slope 0.1',fontsize=8);axs[1].text(35,6,'Underprediction\npenalty slope 0.9',fontsize=8)
    for a in axs:a.grid(alpha=.2)
    save(fig,'sampling-loss')


def _loss_matrix():

    d=pd.read_csv(ROOT/'results/h1_tracelab_r200__metrics.csv')
    d=d[(d['class']=='interactive')&(d.target=='kv_blocks')]
    names=['B0_constant','B1_kalman','B2_history','M1_survival','M2_progress']; hs=[10,30,120,300,900]
    matrix=d.pivot(index='model',columns='h',values='pinball90').loc[names,hs].to_numpy()
    ratio=matrix/matrix[1]
    fig,ax=plt.subplots(figsize=(7,2.8),layout='constrained')
    im=ax.pcolormesh(np.arange(6)-.5,np.arange(6)-.5,np.log2(ratio),shading='flat',cmap='RdBu_r',vmin=-3.2,vmax=3.2)
    ax.invert_yaxis()
    for i in range(5):
        for j in range(5):ax.text(j,i,f'{ratio[i,j]:.2f}×',ha='center',va='center',color='white' if abs(np.log2(ratio[i,j]))>2.2 else 'black',fontsize=9)
    ax.set(xticks=range(5),xticklabels=['10 s','30 s','2 min','5 min','15 min'],yticks=range(5),yticklabels=['B0 persistence','B1 Kalman','B2 history','M1 survival','M2 progress'],xlabel='Forecast horizon')
    cb=fig.colorbar(im,ax=ax,ticks=[-3,-2,-1,0,1,2,3]);cb.ax.set_yticklabels(['⅛','¼','½','1','2','4','8']);cb.set_label('Loss / B1 loss (log₂ color scale)')
    save(fig,'loss-matrix')

if __name__=='__main__': main()
