"""Aggregate validation curves and matched early clipping statistics, no sample IDs."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from src.research_io import ROOT,write_json,now


def summarize():
    cfg=json.loads((ROOT/'configs/parameterization_first_round_v2.json').read_text())
    root=ROOT/cfg['output_root'];runs={};window=cfg['clip_diagnostic_steps']
    for method in cfg['methods']:
        phase='clip_diagnostic' if method=='Xv' else 'convergence'
        folder=root/f'{phase}_{method}_p0_s{cfg["convergence_seed"]}'
        if not (folder/'training_log.jsonl').exists():continue
        rows={}
        for line in (folder/'training_log.jsonl').read_text().splitlines():
            if line.strip():
                item=json.loads(line)
                if item['update']<=window:rows[item['update']]=item
        if not rows:continue
        norms=np.array([r['gradient_norm_before_clip'] for r in rows.values()])
        scales=np.array([r['gradient_clip_scale'] for r in rows.values()])
        bins={}
        for row in rows.values():
            for item in row['time_loss_bins']:
                key=(item['t_min'],item['t_max']);dest=bins.setdefault(key,{'samples':0,**{k:0. for k in ['mse','unit','cosine','loss']}})
                dest['samples']+=item['samples']
                for k in ['mse','unit','cosine','loss']:
                    if item['samples']:dest[k]+=item[k]*item['samples']
        run={'matched_window_updates':window,'updates_recorded':len(rows),'window_completed':len(rows)==window,
             'gradient_clip_fraction':float(np.mean([r['gradient_clipped'] for r in rows.values()])),
             'gradient_norm_quantiles':{str(q):float(np.quantile(norms,q)) for q in [.5,.9,.99,1.]},
             'clip_scale_quantiles':{str(q):float(np.quantile(scales,q)) for q in [.1,.5,.9]},
             'time_loss_bins':[{'t_min':a,'t_max':b,'samples':r['samples'],
                 **{k:r[k]/r['samples'] if r['samples'] else None for k in ['mse','unit','cosine','loss']}}
                 for (a,b),r in bins.items()]}
        curve_file=folder/'learning_curve.json'
        if curve_file.exists():run['learning_curve']=json.loads(curve_file.read_text())['points']
        runs[method]=run
    value={'status':'diagnostic_progress','updated_at':now(),'runs':runs,
           'training_seed':cfg['convergence_seed'],'condition_dropout':0,'formal_grid_locked':True,
           'common_gradient_clip_threshold':cfg['gradient_clip_norm'],
           'comparison_ready':len(runs)==4 and all(r['window_completed'] for r in runs.values()),
           'note':'One training seed; loss units differ by parameterization. Review common-window clipping and per-domain validation before choosing shared formal budget.'}
    write_json(root/'diagnostic_progress.json',value)
    curves={m:r.get('learning_curve',[]) for m,r in runs.items() if r.get('learning_curve')}
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
        for ax,domain in zip(axes,['hypersim','arb','nyuv2']):
            for method,points in curves.items():
                ax.plot([p['update'] for p in points],[p['validation'][domain]['mean_mae_deg'] for p in points],marker='o',label=method)
            ax.set(title=domain,xlabel='Optimizer updates',ylabel='Mean angular error (degrees)');ax.grid(alpha=.2);ax.legend()
        fig.suptitle('Center-square v2 | seed 1247 | p=0 | paired inference draws 42,43')
        target=ROOT/'figures/P2_convergence_center_v2.png';target.parent.mkdir(exist_ok=True)
        fig.savefig(target,dpi=160);plt.close(fig)
    return value


if __name__=='__main__':
    result=summarize()
    print(json.dumps({'methods_recorded':list(result['runs']),'comparison_ready':result['comparison_ready']}))
