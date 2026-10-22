"""Verify shared pilot signatures and report preflight completion without ranking."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
from src.research_io import ROOT,write_json,now

cfg=json.loads((ROOT/'configs/parameterization_first_round.json').read_text())
root=ROOT/'outputs/P2_parameterization_v1';rows=[];signatures=[]
for method in cfg['methods']:
    for dropout in cfg['condition_dropout']:
        folder=root/f'pilot_{method}_p{dropout:g}_s1247'
        result=json.loads((folder/'summary.json').read_text())
        if result['status']!='completed' or result['optimizer_updates']!=cfg['pilot_steps']:
            raise RuntimeError('Pilot grid incomplete')
        signature=json.loads((folder/'signature_private.json').read_text())
        signatures.append((signature['config_sha256'],tuple(signature['data_manifests_sha256']),signature['train_code_sha256'],signature['target_code_sha256']))
        rows.append({'method':method,'condition_dropout':dropout,'optimizer_updates':result['optimizer_updates'],
            'max_gradient_norm_before_clip':result['max_gradient_norm_before_clip'],
            'peak_cuda_bytes':result['peak_cuda_bytes'],'all_losses_and_gradients_finite':True})
if len(set(signatures))!=1:raise RuntimeError('Pilot does not share identical configuration, data and training source')
prepared=json.loads((ROOT/'data/arb_mixed_pilot_v1/summary.json').read_text())
summary={'status':'completed','created_at':now(),'completed_pilot_runs':len(rows),'training_seed':1247,
    'shared_config_data_and_code_verified':True,'rows':rows,
    'object_source_assets':prepared['assets'],'usable_object_assets':prepared['usable_assets'],
    'object_images':prepared['images'],'excluded_empty_label_views':prepared['excluded_views'],
    'ray_condition_used':False,'cfg_used':False,'independent_training_seed_variance_estimated':False,
    'method_ranking_established':False,'convergence_established':False,
    'scope':'100-update single-seed numerical preflight only; formal 24-run grid is separate'}
write_json(ROOT/'outputs/P2_parameterization_pilot_v1/summary.json',summary)
print(json.dumps(summary),flush=True)
