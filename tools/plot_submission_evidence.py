"""Standalone publication figures from audited results; no model execution."""
import os
import argparse
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','D:/MyLab/FlagOS/tmp/matplotlib')
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final-clean',action='store_true');args=parser.parse_args()
    prefix='20261118' if args.final_clean else '20261116'
    repo=Path(__file__).resolve().parents[1];out=repo/'docs/figures';out.mkdir(exist_ok=True)
    new=json.loads((repo/'results/2026-11-16-v2-flagtree07-complete/independent-result.json').read_text())
    old=json.loads((repo/'results/2026-10-31-iluvatar-mixed-v2/independent-result.json').read_text())
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(7.2,3.4));x=np.arange(2);width=.23
    reference=np.array([2028.01,915.15])
    values=[reference,np.array([r['total_tokens_per_second'] for r in old['cases']]),np.array([r['total_tokens_per_second'] for r in new['cases']])]
    for i,(ys,label,color) in enumerate(zip(values,['Published reference','V2 / compiler 0.6.2a3','V2 / compiler 0.7.0'],['#a9b7c5','#3b6d91','#147e78'])):
        ratios=ys/reference
        bars=ax.bar(x+(i-1)*width,ratios,width,label=label,color=color)
        for bar,n in zip(bars,ys):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.025,f'{n:,.1f}',ha='center',fontsize=8)
    ax.set_xticks(x,['4K input / 1024 output','16K input / 1024 output']);ax.set_ylabel('Throughput / published reference')
    ax.set_ylim(0,2.4);ax.legend(frameon=False,loc='upper left',fontsize=8)
    ax.set_title('Complete original workloads: last-three-round means',loc='left',fontsize=11,pad=12)
    fig.text(.08,.025,'Official reference and measured V2 configurations; original workloads, last-three-round means.' if args.final_clean else 'Independent sessions. Not a causal A/B; compiler repair is not our code contribution.',fontsize=7.5,color='#555555')
    fig.subplots_adjust(left=.10,right=.98,top=.86,bottom=.22)
    for ext in ('png','svg'):fig.savefig(out/f'{prefix}-throughput.{ext}',dpi=180)
    plt.close(fig)
    audit=json.loads((repo/'results/2026-11-16-compiler-repair/independent-tensor-audit.json').read_text())
    cases=audit['cases'];x=np.arange(len(cases))
    a=[r['variants']['old/native']['max_abs'] for r in cases];b=[r['variants']['new/native']['max_abs'] for r in cases]
    fig,ax=plt.subplots(figsize=(7.2,3.5))
    for i,(v,w) in enumerate(zip(a,b)):ax.plot([i,i],[v,w],color='#ced6dc',linewidth=1,zorder=1)
    ax.scatter(x,a,label='Compiler 0.6.2a3',color='#bd6e36',marker='x',s=28,zorder=2)
    ax.scatter(x,b,label='Compiler 0.7.0',color='#147e78',s=25,zorder=3)
    ax.set_yscale('log');ax.set_ylabel('Max absolute error (sampled rows)')
    ax.set_xticks(x,[r['label'].replace('_tail','') for r in cases],rotation=50,ha='right',fontsize=7.3)
    ax.set_title('Shared inputs, unchanged attention source: CPU-reference errors',loc='left',fontsize=11,pad=12)
    ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15)
    fig.text(.10,.025,'All 16 cases shown. CPU float64 reference; correctness uses both atol=0.02 and rtol=0.02.',fontsize=7.4,color='#555555')
    fig.subplots_adjust(left=.12,right=.98,top=.86,bottom=.29)
    for ext in ('png','svg'):fig.savefig(out/f'{prefix}-compiler-error.{ext}',dpi=180)
    plt.close(fig)
    print('Two audited figures exported as PNG and SVG.')


if __name__=='__main__':main()
