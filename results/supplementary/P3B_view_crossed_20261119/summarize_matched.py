"""Paired video bootstrap and all-split summaries, without favorable selection."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from analyze_matched import ROOT,aurc,dump

PAIRS=[('ours_checked','pg_full'),('ours_checked','pg_scope_selected'),
       ('ours_checked','pg_fixed_rf'),('ours_unchecked_lr','pg_same_lr'),
       ('ours_checked','ours_unchecked_lr')]

def main():
    tables=[];selections=[]
    for policy in ['original_views','phase_views']:
        for split in range(31):
            directory=ROOT/'analysis'/policy/f'split_{split:02d}'
            selections.append(json.loads((directory/'COMPLETE.json').read_text()))
            tables.append(pd.read_csv(directory/'metrics.csv'))
    all_metrics=pd.concat(tables,ignore_index=True)
    all_metrics.to_csv(ROOT/'matched_metrics.csv',index=False)
    repeated=all_metrics[all_metrics.split>0]
    summaries=[]
    for keys,g in repeated.groupby(['policy','subset','method'],sort=False):
        assert len(g)==30
        summaries.append(dict(zip(['policy','subset','method'],keys))|{'splits':30,
            'mean_gain':g.gain_percent.mean(),'p05_gain':g.gain_percent.quantile(.05),
            'p95_gain':g.gain_percent.quantile(.95),'min_gain':g.gain_percent.min(),
            'max_gain':g.gain_percent.max(),'harm_rate':g.harm.mean(),'mean_aurc':g.aurc.mean()})
    pd.DataFrame(summaries).to_csv(ROOT/'repeated_summary.csv',index=False)
    pd.DataFrame(selections).to_csv(ROOT/'selection_frequencies_source.csv',index=False)
    intervals=[]
    archived=np.load(ROOT.parent/'P0_recompute/Perception_resamples.npz')
    for policy in ['original_views','phase_views']:
        frame=pd.read_csv(ROOT/'analysis'/policy/'split_00/test_predictions.csv',float_precision='round_trip')
        groups=np.array(sorted(frame.group_id.unique()))
        assert np.array_equal(groups,archived['groups'])
        positions={v:np.flatnonzero(frame.group_id.to_numpy()==v) for v in groups}
        draws=archived['indices'];assert draws.shape==(10000,250)
        methods=['raw','ours_checked','pg_full','pg_scope_selected','pg_fixed_rf','ours_unchecked_lr','pg_same_lr']
        for subset in ['all','physics']:
            mask=np.ones(len(frame),bool) if subset=='all' else frame.task.astype(str).str.lower().eq('physics').to_numpy()
            selected=frame[mask];y=frame.correct.to_numpy();values={n:frame[n].to_numpy() for n in methods}
            boot={n:np.empty(len(draws)) for n in methods}
            for i,draw in enumerate(draws):
                take=np.concatenate([positions[groups[j]] for j in draw]);take=take[mask[take]]
                assert len(take)>0
                for n in methods:boot[n][i]=aurc(y[take],values[n][take])
            np.savez_compressed(ROOT/(policy+'_'+subset+'_bootstrap.npz'),**boot)
            point={n:aurc(y[mask],values[n][mask]) for n in methods}
            for n in methods:
                dist=boot[n];lo,hi=np.quantile(dist,[.025,.975])
                intervals.append({'policy':policy,'subset':subset,'quantity':'aurc','method':n,'estimate':point[n],'lo':lo,'hi':hi})
                gain=100*(1-dist/boot['raw']);lo,hi=np.quantile(gain,[.025,.975])
                intervals.append({'policy':policy,'subset':subset,'quantity':'relative_gain_percent','method':n,'estimate':100*(1-point[n]/point['raw']),'lo':lo,'hi':hi})
            for left,right in PAIRS:
                dist=boot[left]-boot[right];lo,hi=np.quantile(dist,[.025,.975])
                intervals.append({'policy':policy,'subset':subset,'quantity':'paired_aurc_difference_negative_favors_left','method':left+' minus '+right,'estimate':point[left]-point[right],'lo':lo,'hi':hi})
    pd.DataFrame(intervals).to_csv(ROOT/'original_split_intervals.csv',index=False)
    dump(ROOT/'SUMMARY_COMPLETE.json',{'policies':2,'splits_per_policy':31,'resamples':10000,
        'archived_original_split_draws_reused':True,'metric':'prefix-risk mean with archived stable ties',
        'split_quantiles_are_not_independent_dataset_CIs':True})

if __name__=='__main__':main()
