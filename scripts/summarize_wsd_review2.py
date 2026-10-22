"""Collect actual stable and cooled results; only cooled points select budgets."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
from src.research_io import ROOT,write_json,now
from src.wsd_protocol import decide_wsd
from scripts.train_wsd_review2 import run_name

CONFIG='configs/parameterization_first_round_v2_review2.json'


def summarize():
    cfg=json.loads((ROOT/CONFIG).read_text());rule=json.loads((ROOT/cfg['decision_rule']).read_text())
    root=ROOT/cfg['output_root'];stable={};cooled={};points={};training={}
    for method in rule['methods']:
        file=root/run_name('stable',method)/'learning_curve.json'
        if file.exists():
            stable[method]={p['update']:{d:r['mean_mae_deg'] for d,r in p['validation'].items()} for p in json.loads(file.read_text())['points']}
        for step in [*rule['initial_fork_steps'],50000]:
            file=root/run_name('cooldown',method,step)/'summary.json'
            if file.exists():
                result=json.loads(file.read_text())
                if result.get('status')=='completed':
                    cooled.setdefault(method,{})[step]={d:r['mean_mae_deg'] for d,r in result['validation'].items()}
                    training.setdefault(method,{})[step]={d:r['mean_mae_deg'] for d,r in result['training_probe'].items()}
    decision=decide_wsd(stable,cooled,rule);decision['updated_at']=now()
    write_json(root/'budget_decision.json',decision)
    write_json(root/'cooled_progress.json',{'status':'diagnostic_progress','created_at':now(),'stable':stable,'cooled':cooled,
        'training_probe':training,'budget_definition':'fork step +2500','formal_grid_locked':True})
    if cooled:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
        for ax,domain in zip(axes,rule['domains']):
            for method,rows in cooled.items():
                steps=sorted(rows);line=ax.plot([s+2500 for s in steps],[rows[s][domain] for s in steps],marker='o',label=method+' cooled val')[0]
                if domain in training[method][steps[0]]:
                    ax.plot([s+2500 for s in steps],[training[method][s][domain] for s in steps],linestyle='--',color=line.get_color(),label=method+' train probe')
            ax.set(title=domain,xlabel='Total optimizer updates (stable + cooldown)',ylabel='MAE (degrees)');ax.grid(alpha=.2);ax.legend()
        fig.suptitle('Review2 WSD | 2500-update linear cooldown forks | seed 1247, p=0')
        fig.savefig(ROOT/'figures/P2_wsd_review2_cooled.png',dpi=160);plt.close(fig)
    return decision


if __name__=='__main__':print(json.dumps(summarize()))
