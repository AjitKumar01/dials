"""Fig. 3 (bundle budget) and Fig. 4 (learned same-label curves) of the TRIAD paper.

Inputs: bundlefuse/results/fair_prior/budget_comparison.json (experiments.fair_prior.budget_compare) and paper/figdata/datasets.json.
Training pairs per condition (sum over bundles of |b|(|b|-1)/2) were counted with the conditions' bundles on 2026-11-16.
python paper/figdata/make_figs.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[3]/'bundlefuse'/'results'
OUT=Path(__file__).resolve().parents[1]/'figs'
PAIRS={'electronic':{1211:7802,600:3964,300:1876,100:615,50:331},'food':{1232:7780,600:3885,300:1797,100:638,50:271},
 'clothing':{1294:7102,600:3449,300:1821,100:714,50:369},'pog_dense':{20780:100640,2078:10237,208:952},'pog':{14000:69633,1400:7165,140:664},
 'spotify_sparse':{8740:14785612,3000:4960995,1000:1637352,600:1094418,300:639719,100:159656,50:72510},
 'spotify':{14000:45910761,3000:9937026,1000:3477821,600:2170142,300:1089551,100:413145,50:228648}}
NAME={'electronic':'Electronic','food':'Food','clothing':'Clothing','pog_dense':'POG_DENSE','pog':'POG','spotify_sparse':'Spotify_sparse','spotify':'Spotify'}
COL={'electronic':'#2a78d6','food':'#eb6834','clothing':'#1baf7a','pog_dense':'#eda100','pog':'#e87ba4','spotify_sparse':'#008300','spotify':'#6250d6'}
MARK={'electronic':'o','food':'s','clothing':'^','pog_dense':'D','pog':'v','spotify_sparse':'P','spotify':'X'}
plt.rcParams.update({'font.size':7,'axes.linewidth':.5,'xtick.major.width':.5,'ytick.major.width':.5,'font.family':'sans-serif',
                     'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42})

def series():
    """TRIAD minus the best simple baseline (with the frequency prior allowed), per dataset and training-set size."""
    rows={}
    c=json.loads((R/'fair_prior'/'budget_comparison.json').read_text())
    for fam in c['families'].values():
        for t in fam['tests']:
            name,cond=t['dataset'],t['condition']
            if t['metric']!='ndcg20' or cond.startswith('pseudo'):continue
            sizes=sorted(PAIRS[name])[::-1]
            b=sizes[0] if cond=='full' else int(cond[1:]) if cond.startswith('n') else sizes[['frac10','frac1'].index(cond)+1]
            rows.setdefault(name,[]).append((b,PAIRS[name][b],t['delta'],*t['ci95']))
    return {k:sorted(v) for k,v in rows.items()}

S=series()
fig,axes=plt.subplots(1,2,figsize=(4.8,1.85),sharey=True)
for ax,j,xl in ((axes[0],0,'training bundles'),(axes[1],1,'item pairs in the training bundles')):
    ax.axhline(0,color='#8a8a85',lw=.6,zorder=0)
    for name,v in S.items():
        v=np.array(v);x=v[:,j]
        ax.fill_between(x,v[:,3],v[:,4],color=COL[name],alpha=.12,lw=0)
        ax.plot(x,v[:,2],color=COL[name],lw=1.1,marker=MARK[name],ms=3.2,mec='white',mew=.4,label=NAME[name])
    ax.set_xscale('log');ax.set_xlabel(xl)
axes[0].set_ylabel('$\\Delta$ NDCG@20\n(TRIAD $-$ best simple)')
axes[0].text(1500,-.040,'TRIAD behind',color='#8a8a85',fontsize=6);axes[0].text(45,.044,'TRIAD ahead',color='#8a8a85',fontsize=6)
axes[1].legend(loc='center left',bbox_to_anchor=(1.0,.5),fontsize=6,handlelength=1.6)
fig.tight_layout(pad=.3);fig.savefig(OUT/'fig3_budget.pdf');print('fig3 ok')

# Fig. 4: learned same-label score delta_l at m = 1 and the median query size, against label size
D=json.loads((Path(__file__).parent/'datasets.json').read_text())
fig,ax=plt.subplots(figsize=(3.0,1.85))
ax.axhline(0,color='#8a8a85',lw=.6,zorder=0)
for name in ['electronic','food','clothing','pog_dense','pog','spotify_sparse','spotify']:
    c=D[name]['delta_curves'][0];g=np.array(c['grid']);d=np.array(c['delta']);q90=c['label_size_q90']
    keep=g<=max(q90,2)*1.0
    lab=NAME[name]+(' (artist)' if name.startswith('spotify') else '')
    ax.fill_between(g[keep],d.min(0)[keep],d.max(0)[keep],color=COL[name],alpha=.15,lw=0)
    ax.plot(g[keep],d.mean(0)[keep],color=COL[name],lw=1.1,label=lab,marker=MARK[name],ms=2.6,markevery=6,mec='white',mew=.3)
ax.set_xscale('log');ax.set_xlabel('label size $n_g$ (items with the label)');ax.set_ylabel('same-label score $\\delta_l$')
ax.legend(loc='center left',bbox_to_anchor=(1.0,.5),fontsize=6,handlelength=1.6)
fig.tight_layout(pad=.3);fig.savefig(OUT/'fig4_delta.pdf');print('fig4 ok')

# Fig. 5: reading the other parameters (paper/figdata/parameters.json, first 512 validation queries, seeds 0-2)
P=json.loads((Path(__file__).parent/'parameters.json').read_text())
SEVEN=['electronic','food','clothing','pog_dense','pog','spotify_sparse','spotify']
fig,axes=plt.subplots(1,4,figsize=(7.2,2.25))
# (a) w_G rho against the base rate, per frequency bin, both centred per dataset (a softmax ignores a shift)
ax=axes[0]
for name in SEVEN:
    bins={}
    for r in P[name]:
        for b in r['rho_bins']:bins.setdefault(b['f'],[]).append((b['base'],b['prior']))
    order=[k for k in bins];v=np.array([np.mean(bins[k],0) for k in order])
    x=v[:,0]-v[:,0].mean();y=v[:,1]-v[:,1].mean()
    ax.plot(x,y,color=COL[name],lw=.6,marker=MARK[name],ms=2.8,mec='white',mew=.3,label=NAME[name])
lim=[-3.2,3.2];ax.plot(lim,lim,color='#8a8a85',lw=.5,ls='--',zorder=0);ax.set_xlim(lim);ax.set_ylim(lim)
ax.set_xlabel('base rate $b(f)$ (centred)');ax.set_ylabel('$w_G\\,\\rho$ (centred)');ax.set_title('(a) popularity prior',fontsize=7)
# (b) gate weight of content as the share of unseen seeds goes 0 -> 1, relative to its value at 0
ax=axes[1]
for name in SEVEN:
    w=np.array([[row[0] for row in r['gate_sweep']['unseen']['w']] for r in P[name]]);w=w/w[:,:1]
    at=P[name][0]['gate_sweep']['unseen']['at']
    ax.fill_between(at,w.min(0),w.max(0),color=COL[name],alpha=.12,lw=0)
    ax.plot(at,w.mean(0),color=COL[name],lw=1.1,marker=MARK[name],ms=2.8,mec='white',mew=.3)
ax.axhline(1,color='#8a8a85',lw=.5,zorder=0)
ax.set_xlabel('share of seeds unseen in training');ax.set_ylabel('content weight $w_C$\n(relative to all seen)');ax.set_title('(b) gate',fontsize=7)
# (c) the degree discount at the 90th-percentile degree, deg^-beta (candidate) and deg^-alpha (seed)
ax=axes[2];ys=np.arange(len(SEVEN))[::-1]
for y,name in zip(ys,SEVEN):
    c=np.array([r['cand_factor_deg90_vs_1'] for r in P[name]]);s=np.array([r['seed_factor_deg90_vs_1'] for r in P[name]])
    ax.plot([c.min(),c.max()],[y+.15]*2,color=COL[name],lw=2,alpha=.5,solid_capstyle='round');ax.plot(c.mean(),y+.15,'o',color=COL[name],ms=3)
    ax.plot([s.min(),s.max()],[y-.15]*2,color=COL[name],lw=2,alpha=.5,solid_capstyle='round');ax.plot(s.mean(),y-.15,'s',color=COL[name],ms=3,mfc='white',mew=.7)
ax.axvline(1,color='#8a8a85',lw=.5,zorder=0);ax.set_xscale('log');ax.set_yticks(ys);ax.set_yticklabels([NAME[n] for n in SEVEN],fontsize=6)
ax.set_xlabel('factor on a count at the\n90th-pct degree (log)');ax.set_title('(c) degree discount',fontsize=7,pad=13)
ax.plot([],[],'o',color='#555',ms=3,label='candidate $\\deg^{-\\beta}$');ax.plot([],[],'s',color='#555',ms=3,mfc='white',mew=.7,label='seed $\\deg^{-\\alpha}$')
ax.legend(fontsize=5.5,loc='lower center',ncol=2,handlelength=1,borderaxespad=0,bbox_to_anchor=(.5,1.01),columnspacing=.6)
# (d) how much each block of the content score moves the top 100 (std, gate weight included), share of the total
ax=axes[3];blocks=['zz','zh','hz','hh'];shade=['#3d3d3a','#7d7d78','#b3b3ad','#dcdcd6']
for y,name in zip(ys,SEVEN):
    v=np.mean([[r['content_block_spread_top100'][b] for b in blocks] for r in P[name]],0);v=v/v.sum();left=0
    for b,x,col in zip(blocks,v,shade):ax.barh(y,x,left=left,color=col,height=.7,edgecolor='white',lw=.4);left+=x
ax.set_yticks(ys);ax.set_yticklabels([]);ax.set_xlim(0,1);ax.set_xlabel('share of the content spread');ax.set_title('(d) content blocks',fontsize=7)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c,label=b) for b,c in zip(blocks,shade)],fontsize=5.5,ncol=4,loc='lower center',bbox_to_anchor=(.5,1.01),handlelength=.8,columnspacing=.6,borderaxespad=0)
ax.set_title('(d) content blocks',fontsize=7,pad=13)
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc='lower center',ncol=7,fontsize=6,handlelength=1.6,columnspacing=1.2,bbox_to_anchor=(.5,0))
fig.tight_layout(pad=.3,w_pad=.6,rect=(0,.08,1,1));fig.savefig(OUT/'fig5_params.pdf');print('fig5 ok')
