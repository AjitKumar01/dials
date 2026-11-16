"""Fig. 3 (bundle budget) and Fig. 4 (learned same-label curves) of the DIALS paper.

Inputs: bundlefuse/results/bundle_budget{,_retail}/comparison*.json and paper/figdata/datasets.json.
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
    rows={}
    for f,fam in [('bundle_budget_retail/comparison.json',None),('bundle_budget_retail/comparison_spotify.json',None),('bundle_budget/comparison.json','pog')]:
        c=json.loads((R/f).read_text())
        for t in c['tests']:
            if t['metric']!='ndcg20' or not t['comparison'].startswith('DIALS vs best simple'):continue
            name=t['dataset'];cond=t['condition']
            if name in ('pog','pog_dense') and cond not in ('full','frac10','frac1'):continue
            b=c['datasets'][name][cond].get('bundles')
            if b is None:b=sorted(PAIRS[name])[::-1][['full','frac10','frac1'].index(cond)]
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
axes[0].set_ylabel('$\\Delta$ NDCG@20\n(DIALS $-$ best simple)')
axes[0].text(2500,-.1,'DIALS behind',color='#8a8a85',fontsize=6);axes[0].text(45,.062,'DIALS ahead',color='#8a8a85',fontsize=6)
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
