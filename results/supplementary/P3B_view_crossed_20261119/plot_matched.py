"""Static research figures from saved predictions and split metrics."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
METHODS=[('raw','Original confidence','#777777'),('ours_unchecked_lr','Our features with fixed LR','#1764ab'),
         ('pg_same_lr','PhaseGuard features with fixed LR','#c45c16')]
def main():
    frame=pd.read_csv(ROOT/'matched_metrics.csv')
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained',sharey='row')
    for column,(policy,title) in enumerate([('original_views','Original four views'),('phase_views','Four two-frame phases')]):
        prediction=pd.read_csv(ROOT/'analysis'/policy/'split_00/test_predictions.csv',float_precision='round_trip')
        for method,label,color in METHODS:
            order=np.argsort(-prediction[method].to_numpy(),kind='mergesort')
            risk=np.cumsum(1-prediction.correct.to_numpy()[order])/np.arange(1,len(order)+1)
            axes[0,column].plot(np.arange(1,len(order)+1)/len(order),risk,label=label,color=color,lw=1.6)
        axes[0,column].set(title=title,xlabel='Coverage',ylabel='Risk',xlim=(0,1),ylim=(0,None))
        axes[0,column].grid(alpha=.2);axes[0,column].legend(fontsize=8,loc='upper left')
        selected=frame[(frame.policy==policy)&(frame.split>0)&(frame.subset=='all')]
        values=[selected[selected.method==method].sort_values('split').gain_percent.to_numpy() for method,_,_ in METHODS[1:]]
        assert all(len(v)==30 for v in values)
        boxes=axes[1,column].boxplot(values,labels=['Our features with LR','PhaseGuard features with LR'],patch_artist=True,showfliers=False)
        for patch,(_,_,color) in zip(boxes['boxes'],METHODS[1:]):patch.set(facecolor=color,alpha=.22)
        for index,value in enumerate(values,1):axes[1,column].scatter(np.full(len(value),index)+np.linspace(-.08,.08,len(value)),value,s=14,color=METHODS[index][2],alpha=.7,zorder=3)
        axes[1,column].axhline(0,color='#777777',lw=1,linestyle=':')
        axes[1,column].set(ylabel='Relative AURC reduction (%)',title='All 30 additional fixed splits')
        axes[1,column].grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'matched_comparison.png',dpi=180)
    fig.savefig(ROOT/'matched_comparison.pdf')
    plt.close(fig)
if __name__=='__main__':main()
