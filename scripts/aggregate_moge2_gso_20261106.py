
"""Reconcile the first completed external GSO-val scope, without opening held-out test."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,math
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 run=ROOT/'outputs/external_round_20261106/MoGe2/gso';mp=ROOT/'data/external_round_20261106/gso_primary.json';records=read(mp)['records'];frozen=read(run/'summary.json')
 assert frozen['status']=='completed' and len(records)==frozen['completed_images']==1860
 files=list((run/'metrics/gso_primary').glob('*.json'));assert len(files)==1860 and {x.stem for x in files}=={x['id'] for x in records}
 grouped=defaultdict(lambda:defaultdict(list))
 for r in records:
  assert '/test/' not in r['gt_path'];row=read(run/'metrics/gso_primary'/(r['id']+'.json'));assert row['image_key']==r['image_key'] and row['object']==r['object'] and row['elevation_bin']==r['elevation_bin'] and row['registration_sha256']==frozen['registration_sha256']
  value=row['metrics']['mean_deg'];assert math.isfinite(value);grouped[str(r['elevation_bin'])][r['object']].append(value)
 by={};difference=0.
 for pitch,objects in grouped.items():
  assert len(objects)==93 and all(len(x)==4 for x in objects.values())
  score=sum(sum(v)/len(v) for v in objects.values())/len(objects);by[pitch]=score;difference=max(difference,abs(score-frozen['panels']['gso_primary']['by_pitch_MAE_deg'][pitch]))
 mean=sum(by.values())/len(by);assert difference<1e-10 and abs(mean-frozen['panels']['gso_primary']['equal_pitch_object_macro_MAE_deg'])<1e-10
 old=ROOT/'outputs/D2_train_20261030/evaluation/summary.json';d2=read(old)['panels']['primary']['equal_pitch_object_macro_mae_deg']
 doc={'status':'completed_single_external_scope','model':'MoGe2','inference_seeds':[42],'panel':'GSO-val primary','objects':93,'images':1860,'views_per_object_per_angle':4,'by_pitch_MAE_deg':by,'equal_pitch_object_macro_MAE_deg':mean,'D2_frozen_reference_MAE_deg':d2,'D2_training_seed':1247,'D2_reference_inference_seeds':list(range(42,50)),'D2_reference_source_summary_sha256':sha(old),'measurement_registration_sha256':frozen['registration_sha256'],'source_commit':'925b8ed835a7a9cdb7578ba15c658a0afc969030','manifest_sha256':sha(mp),'test_images_scored':0,'full_external_comparison_completed':False,'validation':{'status':'passed','private_metric_rows_reconciled':1860,'exact_manifest_ID_set_verified':True,'all_MAE_values_finite':True,'max_abs_per_pitch_summary_difference_deg':difference,'reconciler_sha256':sha(Path(__file__))}}
 target=ROOT/'publish/external_real_snapshot_20261106/gso_moge2.json';target.write_text(json.dumps(doc,indent=2)+'\n');print('GSO_RECONCILED',mean,d2,flush=True)
if __name__=='__main__':main()
