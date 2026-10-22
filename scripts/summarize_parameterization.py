"""Aggregate independent training runs; keep inference and training variance separate."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,numpy as np
from src.research_io import ROOT,write_json,now

def main():
    cfg=json.loads((ROOT/'configs/parameterization_first_round.json').read_text())
    root=ROOT/'outputs/P2_parameterization_v1';groups={};completed=[]
    for seed in cfg['training_seeds']:
        for method in cfg['methods']:
            for dropout in cfg['condition_dropout']:
                path=root/f'formal_{method}_p{dropout:g}_s{seed}/summary.json'
                if not path.exists():continue
                value=json.loads(path.read_text())
                if value.get('status')!='completed':continue
                key=f'{method}_p{dropout:g}';groups.setdefault(key,[]).append(value);completed.append(value)
    result={}
    for key,runs in groups.items():
        domains={}
        for domain in ['hypersim','arb','nyuv2']:
            values=[r['validation'][domain]['mean_mae_deg'] for r in runs]
            domains[domain]={'mean_mae_deg':float(np.mean(values)),
                'training_seed_sd_ddof1_deg':float(np.std(values,ddof=1)) if len(values)>1 else None,
                'per_training_seed_mae_deg':{str(r['training_seed']):r['validation'][domain]['mean_mae_deg'] for r in runs},
                'mean_within_image_inference_mae_sd_deg':float(np.mean([r['validation'][domain]['mean_image_mae_sd_ddof1_deg'] for r in runs])),
                'mean_pixel_inference_angular_dispersion_deg':float(np.mean([r['validation'][domain]['mean_pixel_angular_dispersion_deg'] for r in runs]))}
        result[key]={'completed_training_seeds':len(runs),'three_training_seeds_complete':len(runs)==3,'domains':domains}
    summary={'status':'completed' if len(completed)==24 else 'partial','updated_at':now(),
        'completed_formal_runs':len(completed),'expected_formal_runs':24,'groups':result,
        'convergence_established':False,'scope':'fixed-update budget, each training-run score averages the same eight image-specific inference draws'}
    write_json(root/'summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
